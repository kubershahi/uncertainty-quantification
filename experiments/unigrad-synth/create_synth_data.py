"""
Create 3D synthetic registration samples from HCP T1w data.

Input layout (per subject):
  datasets/hcp/<subject_id>/T1w/
    - T1w_acpc_dc_restore_brain.nii.gz
    - brainmask_fs.nii.gz

Output layout:
  datasets/hcp_synth/{Train,Val,Test}/<subject_id>_<deformation>.npz

Each output npz contains:
  - source  : fixed/source image (float32, 3D, masked z-score from brain mask)
  - moving  : deformed image (float32, 3D, same normalization as source)
  - u       : displacement field (float32, shape (3, X, Y, Z), voxel units)
  - mask    : brain mask (bool, 3D)
  - source_affine, source_spacing, u_unit
  - deformation_class : one of {none, rigid_like, affine, non_rigid, affine_rigid_plus_non_rigid}
  - subject_id
  - qc_passed

TorchIO transforms run in physical space (mm) using the NIfTI affine; ``u`` is recovered
in voxel index space via the identity-grid trick (backward warp).

Split policy:
  - deterministic 70/15/15 by subject hash (Train/Val/Test)
  - balanced deformation mix in each split:
      5% none, 20% rigid-like, 25% affine, 25% non-rigid, 25% mixed (_ar)

Example:
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --workers 8
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --max-subjects 100 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

# =============================================================================
# CONFIG — 3D HCP synthetic generation
# =============================================================================

# Input filenames
T1_NAME = "T1w_acpc_dc_restore_brain.nii.gz"
MASK_NAME = "brainmask_fs.nii.gz"

# Split ratios (70/15/15)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test gets the remainder

# Deformation ratios (must sum to 1.0)
DEFORMATION_RATIOS = {
    "none": 0.05,
    "rigid_like": 0.20,
    "affine": 0.25,
    "non_rigid": 0.25,
    "affine_rigid_plus_non_rigid": 0.25,
}
DEFORMATION_SUFFIX = {
    "none": "none",
    "rigid_like": "rig",
    "affine": "aff",
    "non_rigid": "nr",
    "affine_rigid_plus_non_rigid": "ar",
}

# Rigid-like transform (rotation + translation only; TorchIO translation in mm)
RIGID_DEGREES = 6.0
RIGID_TRANSLATION_MM = 4.0

# Affine transform (includes scale/shear; TorchIO translation in mm)
AFFINE_SCALES = (0.97, 1.03)
AFFINE_DEGREES = 8.0
AFFINE_TRANSLATION_MM = 4.0

# Elastic transform
ELASTIC_NUM_CONTROL_POINTS = 7
ELASTIC_MAX_DISP_MM = 6.0

# QC checks
INTERIOR_MARGIN = 10
MAX_U_INTERIOR_VOX = 25.0
MAX_U_GLOBAL_VOX = 60.0
MIN_MOVING_MEAN_RATIO = 0.05
MAX_TRANSFORM_ATTEMPTS = 20


@dataclass(frozen=True)
class SubjectEntry:
    subject_id: str
    t1_path: str
    mask_path: str


@dataclass(frozen=True)
class Task:
    subject_id: str
    split: str
    deformation_class: str
    t1_path: str
    mask_path: str
    out_path: str
    seed: int


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def interior_valid_mask(shape_xyz: tuple[int, int, int], margin: int) -> np.ndarray:
    """True where voxel is at least `margin` from every volume boundary."""
    x, y, z = shape_xyz
    mask = np.zeros((x, y, z), dtype=bool)
    if x > 2 * margin and y > 2 * margin and z > 2 * margin:
        mask[margin : x - margin, margin : y - margin, margin : z - margin] = True
    else:
        mask[:] = True
    return mask


def passes_checks(
    u: np.ndarray,
    source: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    interior_margin: int,
    max_u_interior_vox: float | None,
    max_u_global_vox: float | None,
    min_moving_mean_ratio: float | None,
) -> bool:
    mag = displacement_magnitude(u.astype(np.float64))
    full_max = float(np.max(mag))

    interior = interior_valid_mask(source.shape, interior_margin)
    valid = interior & (mask > 0.5)
    if valid.any():
        region_max = float(np.max(mag[valid]))
    else:
        region_max = full_max

    if max_u_global_vox is not None and full_max > max_u_global_vox:
        return False

    if max_u_interior_vox is not None and region_max > max_u_interior_vox:
        return False

    if min_moving_mean_ratio is not None:
        src_region = source[mask > 0.5] if np.any(mask > 0.5) else source
        mov_region = moving[mask > 0.5] if np.any(mask > 0.5) else moving
        src_mean = float(np.mean(src_region))
        mov_mean = float(np.mean(mov_region))
        floor = max(1e-6, src_mean * min_moving_mean_ratio)
        if mov_mean < floor:
            return False

    return True


def build_transform(deformation_class: str) -> tio.Transform:
    if deformation_class == "none":
        return tio.Compose([])
    if deformation_class == "rigid_like":
        return tio.Compose(
            [
                tio.RandomAffine(
                    scales=(1.0, 1.0),
                    degrees=RIGID_DEGREES,
                    translation=(
                        RIGID_TRANSLATION_MM,
                        RIGID_TRANSLATION_MM,
                        RIGID_TRANSLATION_MM,
                    ),
                    default_pad_value="minimum",
                    p=1.0,
                )
            ]
        )
    if deformation_class == "affine":
        return tio.Compose(
            [
                tio.RandomAffine(
                    scales=AFFINE_SCALES,
                    degrees=AFFINE_DEGREES,
                    translation=(
                        AFFINE_TRANSLATION_MM,
                        AFFINE_TRANSLATION_MM,
                        AFFINE_TRANSLATION_MM,
                    ),
                    default_pad_value="minimum",
                    p=1.0,
                )
            ]
        )
    if deformation_class == "non_rigid":
        return tio.Compose(
            [
                tio.RandomElasticDeformation(
                    num_control_points=ELASTIC_NUM_CONTROL_POINTS,
                    max_displacement=(
                        ELASTIC_MAX_DISP_MM,
                        ELASTIC_MAX_DISP_MM,
                        ELASTIC_MAX_DISP_MM,
                    ),
                    locked_borders=2,
                    p=1.0,
                )
            ]
        )
    if deformation_class == "affine_rigid_plus_non_rigid":
        return tio.Compose(
            [
                tio.RandomAffine(
                    scales=AFFINE_SCALES,
                    degrees=AFFINE_DEGREES,
                    translation=(
                        AFFINE_TRANSLATION_MM,
                        AFFINE_TRANSLATION_MM,
                        AFFINE_TRANSLATION_MM,
                    ),
                    default_pad_value="minimum",
                    p=1.0,
                ),
                tio.RandomElasticDeformation(
                    num_control_points=ELASTIC_NUM_CONTROL_POINTS,
                    max_displacement=(
                        ELASTIC_MAX_DISP_MM,
                        ELASTIC_MAX_DISP_MM,
                        ELASTIC_MAX_DISP_MM,
                    ),
                    locked_borders=2,
                    p=1.0,
                ),
            ]
        )
    raise ValueError(f"Unknown deformation_class: {deformation_class}")


def _default_parallel_workers() -> int:
    return max(1, os.cpu_count() or 4)


def _pin_worker_cpu_threads() -> None:
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_k] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(dtype=np.float32))


def _load_nifti_with_meta(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NIfTI data plus physical-space metadata.

    Returns:
      data: float32 array
      affine: (4, 4) float32 voxel->world transform (typically mm)
      spacing: (3,) float32 voxel size in mm
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    affine = np.asarray(img.affine, dtype=np.float32)
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
    return data, affine, spacing


def zscore_with_mask(
    vol: np.ndarray,
    mask: np.ndarray,
    *,
    mu: float | None = None,
    sigma: float | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Masked z-score; outside mask set to 0. Optional fixed mu/sigma (e.g. from source)."""
    m = mask > 0.5
    if mu is None or sigma is None:
        in_vals = vol[m]
        in_vals = in_vals[np.isfinite(in_vals)]
        if in_vals.size == 0:
            return np.zeros_like(vol, dtype=np.float32), 0.0, 1.0
        mu = float(np.mean(in_vals))
        sigma = max(float(np.std(in_vals)), eps)
    else:
        sigma = max(float(sigma), eps)
    z = (vol.astype(np.float32) - float(mu)) / float(sigma)
    z[~m] = 0.0
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32), float(mu), float(sigma)


def _build_identity_grid(shape_xyz: tuple[int, int, int]) -> torch.Tensor:
    x, y, z = shape_xyz
    cx, cy, cz = torch.meshgrid(
        torch.arange(x, dtype=torch.float32),
        torch.arange(y, dtype=torch.float32),
        torch.arange(z, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([cx, cy, cz], dim=0).float()


def process_one_subject(
    task: Task,
    *,
    max_u_interior_vox: float | None,
    max_u_global_vox: float | None,
    min_moving_mean_ratio: float | None,
    pin_threads: bool,
) -> tuple[bool, str | None]:
    if pin_threads:
        _pin_worker_cpu_threads()

    source, source_affine, source_spacing = _load_nifti_with_meta(task.t1_path)
    mask = _load_nifti(task.mask_path)
    shape_xyz = (int(source.shape[0]), int(source.shape[1]), int(source.shape[2]))
    identity_grid = _build_identity_grid(shape_xyz)
    transform = build_transform(task.deformation_class)

    moving: np.ndarray | None = None
    u: np.ndarray | None = None
    qc_passed = False

    for attempt in range(MAX_TRANSFORM_ATTEMPTS):
        torch.manual_seed((task.seed + attempt * 100003) % (2**31))
        img_tensor = torch.from_numpy(source).unsqueeze(0).float()
        affine = source_affine.astype(np.float64)
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=img_tensor, affine=affine),
            grid=tio.ScalarImage(tensor=identity_grid.clone(), affine=affine),
        )
        transformed = transform(subject)
        cand_moving = transformed.mri.data.squeeze(0).numpy()
        cand_u = (
            transformed.grid.data.squeeze(0).numpy() - identity_grid.numpy()
        ).astype(np.float32)

        moving, u = cand_moving, cand_u

        if passes_checks(
            cand_u,
            source,
            cand_moving,
            mask,
            interior_margin=INTERIOR_MARGIN,
            max_u_interior_vox=max_u_interior_vox,
            max_u_global_vox=max_u_global_vox,
            min_moving_mean_ratio=min_moving_mean_ratio,
        ):
            qc_passed = True
            break

    assert moving is not None and u is not None
    mask_bin = mask > 0.5
    valid_mask = mask_bin & interior_valid_mask(shape_xyz, INTERIOR_MARGIN)
    source_z, norm_mu, norm_sigma = zscore_with_mask(source, mask_bin)
    moving_z, _, _ = zscore_with_mask(moving, mask_bin, mu=norm_mu, sigma=norm_sigma)
    np.savez_compressed(
        task.out_path,
        source=source_z,
        moving=moving_z,
        u=u.astype(np.float32),
        mask=mask_bin.astype(bool),
        source_affine=source_affine.astype(np.float32),
        source_spacing=source_spacing.astype(np.float32),
        u_unit=np.array("vox"),
        deformation_class=np.array(task.deformation_class),
        subject_id=np.array(task.subject_id),
        qc_passed=qc_passed,
    )

    if qc_passed:
        return True, None

    mag = displacement_magnitude(u.astype(np.float64))
    mx_valid = float(np.max(mag[valid_mask])) if valid_mask.any() else float("nan")
    mx_full = float(np.max(mag))
    lim_int = max_u_interior_vox if max_u_interior_vox is not None else "off"
    lim_glob = max_u_global_vox if max_u_global_vox is not None else "off"
    warn = (
        f"QC_FAIL (saved, qc_passed=False): {task.split}/{Path(task.out_path).name} | "
        f"max|u| valid_mask={mx_valid:.2f}  full_vol={mx_full:.2f}  "
        f"(limits: interior≤{lim_int}, global≤{lim_glob})"
    )
    return False, warn


def _worker_create_sample(task: Task) -> tuple[str, bool, str | None]:
    _pin_worker_cpu_threads()
    qc_ok, warn = process_one_subject(
        task,
        max_u_interior_vox=MAX_U_INTERIOR_VOX,
        max_u_global_vox=MAX_U_GLOBAL_VOX,
        min_moving_mean_ratio=MIN_MOVING_MEAN_RATIO,
        pin_threads=False,
    )
    rel = f"{task.split}/{Path(task.out_path).name}"
    return rel, qc_ok, warn


def stable_subject_hash(subject_id: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{subject_id}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def assign_split(subject_id: str, seed: int) -> str:
    r = stable_subject_hash(subject_id, seed) % 10000 / 10000.0
    if r < TRAIN_FRAC:
        return "Train"
    if r < TRAIN_FRAC + VAL_FRAC:
        return "Val"
    return "Test"


def compute_quotas(n: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = list(ratios.keys())
    raw = {k: n * ratios[k] for k in keys}
    base = {k: int(np.floor(raw[k])) for k in keys}
    rem = n - sum(base.values())
    if rem > 0:
        frac_sorted = sorted(keys, key=lambda k: (raw[k] - base[k], k), reverse=True)
        for k in frac_sorted[:rem]:
            base[k] += 1
    return base


def assign_deformation_classes(subject_ids: list[str], seed: int) -> dict[str, str]:
    if not subject_ids:
        return {}
    rng = np.random.default_rng(seed)
    order = subject_ids.copy()
    rng.shuffle(order)
    quotas = compute_quotas(len(order), DEFORMATION_RATIOS)
    cls_list: list[str] = []
    for cls in DEFORMATION_RATIOS:
        cls_list.extend([cls] * quotas[cls])
    if len(cls_list) < len(order):
        cls_list.extend(["affine"] * (len(order) - len(cls_list)))
    return {sid: cls for sid, cls in zip(order, cls_list)}


def collect_hcp_subjects(input_root: Path) -> list[SubjectEntry]:
    out: list[SubjectEntry] = []
    for subj_dir in sorted(input_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        t1 = subj_dir / "T1w" / T1_NAME
        m = subj_dir / "T1w" / MASK_NAME
        if t1.is_file() and m.is_file():
            out.append(SubjectEntry(subject_id=sid, t1_path=str(t1), mask_path=str(m)))
    return out


def create_synthetic_data(
    input_root: str,
    output_root: str,
    *,
    workers: int | None = None,
    base_seed: int = 42,
    max_subjects: int | None = None,
) -> None:
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, workers if workers is not None else _default_parallel_workers())
    flagged_rel_paths: list[str] = []

    subjects = collect_hcp_subjects(in_root)
    if not subjects:
        print(f"No HCP subjects found in: {in_root}")
        return
    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]

    split_subjects: dict[str, list[SubjectEntry]] = {"Train": [], "Val": [], "Test": []}
    for s in subjects:
        split_subjects[assign_split(s.subject_id, base_seed)].append(s)

    tasks: list[Task] = []
    split_summary: dict[str, dict[str, int]] = {}
    for split, entries in split_subjects.items():
        entries = sorted(entries, key=lambda e: e.subject_id)
        sid_to_class = assign_deformation_classes(
            [e.subject_id for e in entries], seed=base_seed + stable_subject_hash(split, base_seed)
        )
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        c = Counter()
        for idx, e in enumerate(entries):
            cls = sid_to_class[e.subject_id]
            suf = DEFORMATION_SUFFIX[cls]
            out_name = f"{e.subject_id}_{suf}.npz"
            out_path = out_dir / out_name
            seed = (base_seed + stable_subject_hash(e.subject_id, base_seed) + idx * 100003) % (
                2**31
            )
            tasks.append(
                Task(
                    subject_id=e.subject_id,
                    split=split,
                    deformation_class=cls,
                    t1_path=e.t1_path,
                    mask_path=e.mask_path,
                    out_path=str(out_path),
                    seed=seed,
                )
            )
            c[cls] += 1
        split_summary[split] = dict(c)

    print(
        f"Parallel workers: {n_workers} (default = all logical CPUs; each worker uses 1 OpenMP thread)"
    )

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Create 3D HCP synth"):
            qc_ok, warn = process_one_subject(
                t,
                max_u_interior_vox=MAX_U_INTERIOR_VOX,
                max_u_global_vox=MAX_U_GLOBAL_VOX,
                min_moving_mean_ratio=MIN_MOVING_MEAN_RATIO,
                pin_threads=False,
            )
            if not qc_ok:
                rel_flag = f"{t.split}/{Path(t.out_path).name}"
                flagged_rel_paths.append(rel_flag)
                if warn:
                    tqdm.write(warn)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_create_sample, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Create 3D HCP synth"):
                rel_flag_ret, qc_ok, warn = fut.result()
                if not qc_ok:
                    flagged_rel_paths.append(rel_flag_ret)
                    if warn:
                        tqdm.write(warn)

    # save split/deformation summary
    meta = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "seed": base_seed,
        "n_subjects": len(subjects),
        "split_counts": {k: len(v) for k, v in split_subjects.items()},
        "deformation_ratios_target": DEFORMATION_RATIOS,
        "deformation_counts_actual": split_summary,
        "field_names": ["source", "moving", "u", "mask"],
    }
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    manifest = os.path.join(output_root, "qc_flagged_paths.txt")
    if flagged_rel_paths:
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("# Samples with qc_passed=False — delete or regenerate\n")
            for line in sorted(flagged_rel_paths):
                f.write(f"{line}\n")
        print(
            f"Warning: {len(flagged_rel_paths)} sample(s) failed QC after "
            f"{MAX_TRANSFORM_ATTEMPTS} attempts (saved with qc_passed=False). "
            f"List: {manifest}"
        )
    else:
        if os.path.isfile(manifest):
            os.remove(manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create 3D HCP synthetic registration samples with balanced deformation classes."
    )
    p.add_argument(
        "--input-path",
        type=str,
        default="datasets/hcp",
        help="HCP root: datasets/hcp/<subject>/T1w/*.nii.gz",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="datasets/hcp_synth",
        help="Output root for split folders with *_<suffix>.npz files",
    )
    p.add_argument(
        "--workers",
        "--worker",
        type=int,
        metavar="N",
        default=None,
        help="Parallel processes (default: all logical CPUs). Each worker pins 1 OpenMP thread.",
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap for smoke runs (e.g., 100).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-file seeds are derived for reproducible parallel runs.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_synthetic_data(
        args.input_path,
        args.output_path,
        workers=args.workers,
        base_seed=args.seed,
        max_subjects=args.max_subjects,
    )
    print("Finished! 3D HCP synthetic data is ready.")
