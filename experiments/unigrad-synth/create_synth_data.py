"""
Create synthetic registration samples from HCP T1w data.

Input layout (per subject):
  datasets/hcp/<subject_id>/T1w/
    - T1w_acpc_dc_restore_brain.nii.gz
    - brainmask_fs.nii.gz

Output layout:
  datasets/hcp_synth/{Train,Val,Test}/<subject_id>_<suffix>.npz  (qc_passed=True)
  datasets/hcp_synth_qc_fail/{Train,Val,Test}/...                 (qc_passed=False)

Each output npz contains:
  - source  : fixed/source image (float32 volume, masked z-score from brain mask)
  - moving  : deformed image (float32 volume, same normalization as source)
  - u       : displacement field (float32, shape (3, X, Y, Z), voxel units)
  - source_mask    : fixed brain mask (bool; for visualization only)
  - moving_mask    : source_mask warped with the same transform (bool; viz only)
  - identity_grid_mask : in-bounds mask for the displacement field (bool; viz only)
  - source_affine
  - deformation_class : one of {none, rigid, affine, elastic, affine_elastic}
  - u_max_interior, u_mean_interior  (full-volume max/mean ‖u‖; names kept for compat)
  - subject_id, qc_passed

Note: ``source_mask`` / ``moving_mask`` / ``identity_grid_mask`` are written for
visualization only. They are not mixed together and are not used for ‖u‖ stats/QC.

TorchIO transforms run in physical space (mm) using the NIfTI affine; ``u`` is recovered
in voxel index space via the identity-grid trick (backward warp).

Split policy (full run):
  - deterministic 70/15/15 by subject hash (Train/Val/Test)
  - balanced deformation mix in each split:
      5% none, 20% rigid, 25% affine, 25% elastic, 25% affine+elastic

Modes:
  - Full cohort: omit ``--dry-run`` → Train/Val/Test under ``--output-path``
  - Dry run: ``--dry-run [N]`` → flat folder, 5 classes × N samples (default N=5)

Examples:
# Full cohort
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --qc-fail-path datasets/hcp_synth_qc_fail --workers 16
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --max-subjects 30 --workers 16
# Dry run (25 samples)
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth_dryrun2 --dry-run 5 --no-qc --workers 16
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
# CONFIG — HCP synthetic generation
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
    "rigid": 0.20,
    "affine": 0.25,
    "elastic": 0.25,
    "affine_elastic": 0.25,
}
DEFORMATION_SUFFIX = {
    "none": "none",
    "rigid": "rig",
    "affine": "aff",
    "elastic": "ela",
    "affine_elastic": "aela",
}

DEFAULT_QC_FAIL_ROOT = "datasets/hcp_synth_qc_fail"

DRY_RUN_CLASSES = ("none", "rigid", "affine", "elastic", "affine_elastic")

# Single TorchIO sampling envelope per class (no low/mid/high tiers).
# Rigid / affine / affine_elastic: between previous mid and high.
# Pure elastic: above previous high (9 mm) for clearer deformation.
PARAM_RIGID = {"degrees": 8.0, "translation_mm": 5.0}
PARAM_AFFINE = {"scales": (0.96, 1.04), "degrees": 10.0, "translation_mm": 5.5}
PARAM_ELASTIC = {"max_disp_mm": 12.0}
# Milder elastic when composed with affine so the combo stays mid–high overall.
PARAM_ELASTIC_IN_AFFINE = {"max_disp_mm": 8.0}

ELASTIC_NUM_CONTROL_POINTS = 7

# QC checks — max caps (all classes)
INTERIOR_MARGIN = 10
MAX_U_INTERIOR_VOX = 25.0
MAX_U_GLOBAL_VOX = 60.0
MIN_MOVING_MEAN_RATIO = 0.05
MAX_TRANSFORM_ATTEMPTS = 20
MAX_U_NONE_VOX = 0.5

# Per-class minimum interior ‖u‖ (voxels); none uses MAX_U_NONE_VOX instead
MIN_U_MAX_INTERIOR_BY_CLASS: dict[str, float | None] = {
    "none": None,
    "rigid": 1.5,
    "affine": 2.0,
    "elastic": 1.5,
    "affine_elastic": 3.0,
}
MIN_U_MEAN_INTERIOR_BY_CLASS: dict[str, float | None] = {
    "none": None,
    "rigid": 0.5,
    "affine": 0.7,
    "elastic": 0.5,
    "affine_elastic": 1.0,
}


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
    qc_fail_out_path: str
    seed: int
    skip_qc: bool = False


@dataclass(frozen=True)
class SampleStats:
    subject_id: str
    split: str
    deformation_class: str
    qc_passed: bool
    u_max_interior: float
    u_mean_interior: float
    rel_path: str


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


def u_volume_stats(u: np.ndarray) -> tuple[float, float]:
    """Full-volume max and mean ‖u‖ (all voxels; no mask)."""
    mag = displacement_magnitude(u.astype(np.float64))
    return float(np.max(mag)), float(np.mean(mag))


def passes_checks(
    u: np.ndarray,
    source: np.ndarray,
    moving: np.ndarray,
    brain_mask: np.ndarray,
    deformation_class: str,
    max_u_interior_vox: float | None,
    max_u_global_vox: float | None,
    min_moving_mean_ratio: float | None,
) -> bool:
    """QC on full-volume ‖u‖; brain_mask only for intensity mean-ratio check."""
    mag = displacement_magnitude(u.astype(np.float64))
    full_max = float(np.max(mag))
    full_mean = float(np.mean(mag))

    if deformation_class == "none":
        if full_max >= MAX_U_NONE_VOX:
            return False
    else:
        min_max = MIN_U_MAX_INTERIOR_BY_CLASS.get(deformation_class)
        min_mean = MIN_U_MEAN_INTERIOR_BY_CLASS.get(deformation_class)
        if min_max is not None and full_max < min_max:
            return False
        if min_mean is not None and full_mean < min_mean:
            return False

    if max_u_global_vox is not None and full_max > max_u_global_vox:
        return False

    if max_u_interior_vox is not None and full_max > max_u_interior_vox:
        return False

    if min_moving_mean_ratio is not None:
        # Intensity sanity only (FreeSurfer brainmask); not mixed with NPZ viz masks.
        src_region = source[brain_mask > 0.5] if np.any(brain_mask > 0.5) else source
        mov_region = moving[brain_mask > 0.5] if np.any(brain_mask > 0.5) else moving
        src_mean = float(np.mean(src_region))
        mov_mean = float(np.mean(mov_region))
        floor = max(1e-6, src_mean * min_moving_mean_ratio)
        if mov_mean < floor:
            return False

    return True


def dry_run_filename(subject_id: str, deformation_class: str, replicate: int) -> str:
    """e.g. 100206_rigid.npz or 100206_rigid_02.npz when replicate > 0."""
    base = f"{subject_id}_{deformation_class}"
    if replicate > 0:
        return f"{base}_{replicate:02d}.npz"
    return f"{base}.npz"


def _affine_transform(
    *,
    scales: tuple[float, float],
    degrees: float,
    translation_mm: float,
) -> tio.RandomAffine:
    t = translation_mm
    return tio.RandomAffine(
        scales=scales,
        degrees=degrees,
        translation=(t, t, t),
        default_pad_value="minimum",
        p=1.0,
    )


def _elastic_transform(*, max_disp_mm: float) -> tio.RandomElasticDeformation:
    d = max_disp_mm
    return tio.RandomElasticDeformation(
        num_control_points=ELASTIC_NUM_CONTROL_POINTS,
        max_displacement=(d, d, d),
        locked_borders=2,
        p=1.0,
    )


def build_transform(deformation_class: str) -> tio.Transform:
    if deformation_class == "none":
        return tio.Compose([])
    if deformation_class == "rigid":
        p = PARAM_RIGID
        return tio.Compose(
            [
                tio.RandomAffine(
                    scales=(1.0, 1.0),
                    degrees=p["degrees"],
                    translation=(p["translation_mm"],) * 3,
                    default_pad_value="minimum",
                    p=1.0,
                )
            ]
        )
    if deformation_class == "affine":
        return tio.Compose([_affine_transform(**PARAM_AFFINE)])
    if deformation_class == "elastic":
        return tio.Compose([_elastic_transform(**PARAM_ELASTIC)])
    if deformation_class == "affine_elastic":
        return tio.Compose(
            [_affine_transform(**PARAM_AFFINE), _elastic_transform(**PARAM_ELASTIC_IN_AFFINE)]
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


def _load_nifti_with_meta(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI data plus physical-space metadata.

    Returns:
      data: float32 array
      affine: (4, 4) float32 voxel->world transform (typically mm)
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    affine = np.asarray(img.affine, dtype=np.float32)
    return data, affine


def zscore_brain_pair(
    source: np.ndarray,
    moving: np.ndarray,
    source_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Masked z-score from source brain; fill outside each mask with 0 after scaling."""
    src_m = source_mask > 0.5
    mov_m = moving_mask > 0.5
    in_vals = source[src_m]
    in_vals = in_vals[np.isfinite(in_vals)]
    if in_vals.size == 0:
        return (
            np.zeros_like(source, dtype=np.float32),
            np.zeros_like(moving, dtype=np.float32),
            0.0,
            1.0,
        )
    mu = float(np.mean(in_vals))
    sigma = max(float(np.std(in_vals)), eps)
    source_z = (source.astype(np.float32) - mu) / sigma
    moving_z = (moving.astype(np.float32) - mu) / sigma
    source_z[~src_m] = 0.0
    moving_z[~mov_m] = 0.0
    source_z[~np.isfinite(source_z)] = 0.0
    moving_z[~np.isfinite(moving_z)] = 0.0
    return source_z.astype(np.float32), moving_z.astype(np.float32), mu, sigma


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
) -> SampleStats:
    if pin_threads:
        _pin_worker_cpu_threads()

    source, source_affine = _load_nifti_with_meta(task.t1_path)
    mask = _load_nifti(task.mask_path)
    shape_xyz = (int(source.shape[0]), int(source.shape[1]), int(source.shape[2]))
    identity_grid = _build_identity_grid(shape_xyz)
    transform = build_transform(task.deformation_class)

    moving: np.ndarray | None = None
    moving_mask_bin: np.ndarray | None = None
    u: np.ndarray | None = None
    qc_passed = task.skip_qc
    n_attempts = 1 if task.skip_qc else MAX_TRANSFORM_ATTEMPTS
    source_mask_bin = mask > 0.5

    identity_grid_mask_bin: np.ndarray | None = None

    for attempt in range(n_attempts):
        torch.manual_seed((task.seed + attempt * 100003) % (2**31))
        img_tensor = torch.from_numpy(source).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(source_mask_bin.astype(np.float32)).unsqueeze(0)
        ones_tensor = torch.ones_like(img_tensor, dtype=torch.float32)
        affine = source_affine.astype(np.float64)
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=img_tensor, affine=affine),
            grid=tio.ScalarImage(tensor=identity_grid.clone(), affine=affine),
            brain_mask=tio.LabelMap(tensor=mask_tensor, affine=affine),
            valid_mask=tio.LabelMap(tensor=ones_tensor, affine=affine),
        )
        transformed = transform(subject)
        cand_moving = transformed.mri.data.squeeze(0).numpy()
        cand_moving_mask = transformed.brain_mask.data.squeeze(0).numpy() > 0.5
        cand_valid_mask = transformed.valid_mask.data.squeeze(0).numpy()
        cand_u = (
            transformed.grid.data.squeeze(0).numpy() - identity_grid.numpy()
        ).astype(np.float32)
        identity_grid_mask_bin = cand_valid_mask > 0.99
        invalid = ~identity_grid_mask_bin
        cand_u[:, invalid] = 0.0

        moving, moving_mask_bin, u = cand_moving, cand_moving_mask, cand_u

        if task.skip_qc or passes_checks(
            cand_u,
            source,
            cand_moving,
            mask,
            task.deformation_class,
            max_u_interior_vox=max_u_interior_vox,
            max_u_global_vox=max_u_global_vox,
            min_moving_mean_ratio=min_moving_mean_ratio,
        ):
            qc_passed = True
            break

    assert (
        moving is not None
        and moving_mask_bin is not None
        and identity_grid_mask_bin is not None
        and u is not None
    )
    u_max_int, u_mean_int = u_volume_stats(u)
    source_z, moving_z, _, _ = zscore_brain_pair(
        source, moving, source_mask_bin, moving_mask_bin
    )

    save_path = task.out_path if (qc_passed or task.skip_qc) else task.qc_fail_out_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        source=source_z,
        moving=moving_z,
        u=u.astype(np.float32),
        source_mask=source_mask_bin.astype(bool),
        moving_mask=moving_mask_bin.astype(bool),
        identity_grid_mask=identity_grid_mask_bin.astype(bool),
        source_affine=source_affine.astype(np.float32),
        deformation_class=np.array(task.deformation_class),
        u_max_interior=np.float32(u_max_int),
        u_mean_interior=np.float32(u_mean_int),
        subject_id=np.array(task.subject_id),
        qc_passed=qc_passed,
    )

    rel = Path(save_path).name if not task.split else f"{task.split}/{Path(save_path).name}"
    return SampleStats(
        subject_id=task.subject_id,
        split=task.split,
        deformation_class=task.deformation_class,
        qc_passed=qc_passed,
        u_max_interior=u_max_int,
        u_mean_interior=u_mean_int,
        rel_path=rel,
    )


def _qc_fail_warning(stats: SampleStats) -> str:
    cls = stats.deformation_class
    min_max = MIN_U_MAX_INTERIOR_BY_CLASS.get(cls)
    min_mean = MIN_U_MEAN_INTERIOR_BY_CLASS.get(cls)
    return (
        f"QC_FAIL (saved to qc_fail): {stats.rel_path} | "
        f"class={cls} | "
        f"u_max_int={stats.u_max_interior:.2f} u_mean_int={stats.u_mean_interior:.2f} | "
        f"(floors: max≥{min_max}, mean≥{min_mean}; caps: interior≤{MAX_U_INTERIOR_VOX}, "
        f"global≤{MAX_U_GLOBAL_VOX})"
    )


def _worker_create_sample(task: Task) -> SampleStats:
    _pin_worker_cpu_threads()
    return process_one_subject(
        task,
        max_u_interior_vox=MAX_U_INTERIOR_VOX,
        max_u_global_vox=MAX_U_GLOBAL_VOX,
        min_moving_mean_ratio=MIN_MOVING_MEAN_RATIO,
        pin_threads=False,
    )


def build_deformation_stats(samples: list[SampleStats]) -> dict[str, dict[str, float | int]]:
    """Per-class counts and u_max_interior / u_mean_interior p50 summaries."""
    buckets: dict[str, list[SampleStats]] = {}
    for s in samples:
        if not s.qc_passed:
            continue
        buckets.setdefault(s.deformation_class, []).append(s)
    out: dict[str, dict[str, float | int]] = {}
    for cls, items in sorted(buckets.items()):
        max_vals = np.asarray([s.u_max_interior for s in items], dtype=np.float64)
        mean_vals = np.asarray([s.u_mean_interior for s in items], dtype=np.float64)
        out[cls] = {
            "count": int(len(items)),
            "u_max_p50": float(np.percentile(max_vals, 50)) if max_vals.size else 0.0,
            "u_mean_p50": float(np.percentile(mean_vals, 50)) if mean_vals.size else 0.0,
        }
    return out


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


def build_dry_run_tasks(
    subjects: list[SubjectEntry],
    out_root: Path,
    *,
    replicates: int,
    base_seed: int,
    skip_qc: bool,
) -> list[Task]:
    """Dry run: N samples per class; distinct subjects within each class when possible."""
    if len(subjects) < replicates:
        raise ValueError(
            f"Dry run needs ≥{replicates} subjects for distinct subjects within "
            f"each class; found {len(subjects)}"
        )
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = out_root / "_unused_qc_fail"
    # Deterministic subject order; walk without replacement across the whole dry run
    # so classes also get different subjects when enough subjects are available.
    order = sorted(
        subjects,
        key=lambda s: (stable_subject_hash(s.subject_id, base_seed), s.subject_id),
    )
    tasks: list[Task] = []
    task_idx = 0
    for cls_i, cls in enumerate(DRY_RUN_CLASSES):
        for rep in range(replicates):
            subj = order[(cls_i * replicates + rep) % len(order)]
            rep_suffix = rep + 1 if replicates > 1 else 0
            out_name = dry_run_filename(subj.subject_id, cls, rep_suffix)
            seed = (
                base_seed
                + stable_subject_hash(f"{cls}:{rep}:{subj.subject_id}", base_seed)
                + task_idx * 100003
            ) % (2**31)
            tasks.append(
                Task(
                    subject_id=subj.subject_id,
                    split="",
                    deformation_class=cls,
                    t1_path=subj.t1_path,
                    mask_path=subj.mask_path,
                    out_path=str(out_dir / out_name),
                    qc_fail_out_path=str(fail_dir / out_name),
                    seed=seed,
                    skip_qc=skip_qc,
                )
            )
            task_idx += 1
    return tasks


def compute_class_u_stats_table(
    out_root: Path,
    samples: list[SampleStats],
) -> list[dict[str, float | int | str]]:
    """
    Per-class ‖u‖ summary over the full volume (all voxels) pooled across dry-run samples.

    Returns rows with min / Q1 / mean / Q3 / max, plus sample count.
    """
    by_class: dict[str, list[Path]] = {cls: [] for cls in DRY_RUN_CLASSES}
    for s in samples:
        fp = out_root / Path(s.rel_path).name if not s.split else out_root / s.rel_path
        if not fp.is_file():
            fp = out_root / Path(s.rel_path).name
        if fp.is_file() and s.deformation_class in by_class:
            by_class[s.deformation_class].append(fp)

    rows: list[dict[str, float | int | str]] = []
    for cls in DRY_RUN_CLASSES:
        mags: list[np.ndarray] = []
        for fp in by_class[cls]:
            with np.load(fp) as z:
                u = np.asarray(z["u"], dtype=np.float64)
                mags.append(displacement_magnitude(u).ravel())
        if not mags:
            rows.append(
                {
                    "deformation_class": cls,
                    "n_samples": 0,
                    "n_voxels": 0,
                    "min": float("nan"),
                    "q1": float("nan"),
                    "mean": float("nan"),
                    "q3": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        vals = np.concatenate(mags)
        rows.append(
            {
                "deformation_class": cls,
                "n_samples": len(by_class[cls]),
                "n_voxels": int(vals.size),
                "min": float(np.min(vals)),
                "q1": float(np.percentile(vals, 25)),
                "mean": float(np.mean(vals)),
                "q3": float(np.percentile(vals, 75)),
                "max": float(np.max(vals)),
            }
        )
    return rows


def print_and_save_class_u_stats(out_root: Path, samples: list[SampleStats]) -> Path:
    rows = compute_class_u_stats_table(out_root, samples)
    print("\nPer-class ‖u‖ stats (full volume, voxels):")
    print(
        f"{'class':<16} {'n':>3} {'min':>8} {'Q1':>8} {'mean':>8} {'Q3':>8} {'max':>8}"
    )
    for r in rows:
        print(
            f"{r['deformation_class']:<16} {r['n_samples']:>3} "
            f"{r['min']:8.3f} {r['q1']:8.3f} {r['mean']:8.3f} "
            f"{r['q3']:8.3f} {r['max']:8.3f}"
        )
    json_path = out_root / "dryrun_class_u_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {json_path}")
    return json_path


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
    qc_fail_root: str = DEFAULT_QC_FAIL_ROOT,
    workers: int | None = None,
    base_seed: int = 42,
    max_subjects: int | None = None,
    skip_qc: bool = False,
    dry_run: bool = False,
    dry_run_replicates: int = 5,
) -> None:
    in_root = Path(input_root)
    out_root = Path(output_root)
    fail_root = Path(qc_fail_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not skip_qc:
        fail_root.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, workers if workers is not None else _default_parallel_workers())
    flagged_rel_paths: list[str] = []
    all_stats: list[SampleStats] = []

    subjects = collect_hcp_subjects(in_root)
    if not subjects:
        print(f"No HCP subjects found in: {in_root}")
        return
    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]

    if dry_run:
        n_tasks = len(DRY_RUN_CLASSES) * dry_run_replicates
        print(
            f"Dry run: {len(DRY_RUN_CLASSES)} classes × "
            f"{dry_run_replicates} samples = {n_tasks} NPZs"
        )
        print(f"Classes: {', '.join(DRY_RUN_CLASSES)}")
        print(
            f"Params: rigid={PARAM_RIGID}, affine={PARAM_AFFINE}, "
            f"elastic={PARAM_ELASTIC}, affine_elastic_elastic={PARAM_ELASTIC_IN_AFFINE}"
        )
        if skip_qc:
            print("QC disabled (--no-qc): single random draw per task, all saved to output-path")
        tasks = build_dry_run_tasks(
            subjects,
            out_root,
            replicates=dry_run_replicates,
            base_seed=base_seed,
            skip_qc=skip_qc,
        )
        split_summary: dict[str, dict[str, int]] = {}
        split_subjects = {"": subjects}
    else:
        split_subjects = {"Train": [], "Val": [], "Test": []}
        for s in subjects:
            split_subjects[assign_split(s.subject_id, base_seed)].append(s)

        tasks = []
        split_summary = {}
        for split, entries in split_subjects.items():
            entries = sorted(entries, key=lambda e: e.subject_id)
            sid_to_class = assign_deformation_classes(
                [e.subject_id for e in entries],
                seed=base_seed + stable_subject_hash(split, base_seed),
            )
            out_dir = out_root / split
            fail_dir = fail_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            fail_dir.mkdir(parents=True, exist_ok=True)
            c = Counter()
            for idx, e in enumerate(entries):
                cls = sid_to_class[e.subject_id]
                suf = DEFORM_SUFFIX[cls]
                out_name = f"{e.subject_id}_{suf}.npz"
                out_path = out_dir / out_name
                fail_path = fail_dir / out_name
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
                        qc_fail_out_path=str(fail_path),
                        seed=seed,
                        skip_qc=skip_qc,
                    )
                )
                c[cls] += 1
            split_summary[split] = dict(c)

    print(
        f"Parallel workers: {n_workers} (default = all logical CPUs; each worker uses 1 OpenMP thread)"
    )

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Create HCP synth"):
            stats = process_one_subject(
                t,
                max_u_interior_vox=MAX_U_INTERIOR_VOX,
                max_u_global_vox=MAX_U_GLOBAL_VOX,
                min_moving_mean_ratio=MIN_MOVING_MEAN_RATIO,
                pin_threads=False,
            )
            all_stats.append(stats)
            if not stats.qc_passed:
                flagged_rel_paths.append(stats.rel_path)
                tqdm.write(_qc_fail_warning(stats))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_create_sample, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Create HCP synth"):
                stats = fut.result()
                all_stats.append(stats)
                if not stats.qc_passed:
                    flagged_rel_paths.append(stats.rel_path)
                    tqdm.write(_qc_fail_warning(stats))

    passed = sum(1 for s in all_stats if s.qc_passed)
    failed = len(all_stats) - passed
    meta = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "qc_fail_root": str(fail_root) if not skip_qc else None,
        "seed": base_seed,
        "skip_qc": skip_qc,
        "dry_run": dry_run,
        "dry_run_replicates": dry_run_replicates if dry_run else None,
        "dry_run_classes": list(DRY_RUN_CLASSES) if dry_run else None,
        "transform_params": {
            "rigid": PARAM_RIGID,
            "affine": PARAM_AFFINE,
            "elastic": PARAM_ELASTIC,
            "affine_elastic": {"affine": PARAM_AFFINE, "elastic": PARAM_ELASTIC_IN_AFFINE},
        },
        "n_subjects": len(subjects),
        "n_tasks": len(tasks),
        "qc_passed_count": passed,
        "qc_failed_count": failed,
        "split_counts": {k: len(v) for k, v in split_subjects.items()},
        "deformation_ratios_target": DEFORM_RATIOS if not dry_run else None,
        "deformation_counts_actual": split_summary,
        "deformation_stats": build_deformation_stats(all_stats),
        "field_names": [
            "source",
            "moving",
            "u",
            "source_mask",
            "moving_mask",
            "identity_grid_mask",
            "u_max_interior",
            "u_mean_interior",
        ],
    }
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if dry_run:
        print_and_save_class_u_stats(out_root, all_stats)

    manifest = os.path.join(qc_fail_root, "qc_flagged_paths.txt")
    if flagged_rel_paths and not skip_qc:
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("# Samples with qc_passed=False — stored under qc_fail_root\n")
            for line in sorted(flagged_rel_paths):
                f.write(f"{line}\n")
        print(
            f"Warning: {failed} sample(s) failed QC after {MAX_TRANSFORM_ATTEMPTS} attempts "
            f"({failed / max(len(all_stats), 1) * 100:.1f}%). Saved under {fail_root}. "
            f"List: {manifest}"
        )
    else:
        if os.path.isfile(manifest):
            os.remove(manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create HCP synthetic registration samples with balanced deformation classes."
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
        help="Output root for qc_passed samples (split folders with *_<suffix>.npz)",
    )
    p.add_argument(
        "--qc-fail-path",
        type=str,
        default=DEFAULT_QC_FAIL_ROOT,
        help="Output root for qc_passed=False samples",
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
    p.add_argument(
        "--no-qc",
        action="store_true",
        help="Disable QC checks and retries; save first transform draw to output-path.",
    )
    p.add_argument(
        "--dry-run",
        nargs="?",
        const=5,
        default=None,
        type=int,
        metavar="N",
        help=(
            "Dry run mode: write a flat folder with N samples per deformation class "
            "(5 classes × N; default N=5 → 25). Omit this flag for a full Train/Val/Test run."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dry_run = args.dry_run is not None
    dry_run_replicates = args.dry_run if dry_run else 5
    create_synthetic_data(
        args.input_path,
        args.output_path,
        qc_fail_root=args.qc_fail_path,
        workers=args.workers,
        base_seed=args.seed,
        max_subjects=args.max_subjects,
        skip_qc=args.no_qc,
        dry_run=dry_run,
        dry_run_replicates=dry_run_replicates,
    )
    print("Finished! HCP synthetic data is ready.")
