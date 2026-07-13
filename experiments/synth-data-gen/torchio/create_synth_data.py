"""
Create synthetic registration samples from HCP T1w data.

Input layout (per subject):
  datasets/hcp/<subject_id>/T1w/
    - T1w_acpc_dc_restore_brain.nii.gz
    - brainmask_fs.nii.gz

Output layout:
  datasets/synth-data/torchio/hcp/{Train,Val,Test}/<subject_id>_<suffix>.npz
  Dry run: datasets/synth-data/torchio/hcp_dryrun3/<subject_id>_<class>[_NN].npz (flat)

Each output npz contains:
  - source  : fixed/source image (float32 volume, masked z-score from brain mask)
  - moving  : deformed image (float32 volume, same normalization as source)
  - u       : displacement field (float32, shape (3, X, Y, Z), voxel units);
              OOB zeroed via identity_grid_mask, then 12-voxel border zeroed, then p99.9 clip
  - source_mask    : fixed brain mask (bool; for visualization only)
  - moving_mask    : source_mask warped with the same transform (bool; viz only)
  - identity_grid_mask : in-bounds mask for the displacement field (bool; viz only)
  - source_affine
  - deformation_class : one of {none, rigid, affine, elastic, affine_elastic}
  - subject_id

Note: ``source_mask`` / ``moving_mask`` / ``identity_grid_mask`` are written for
visualization only. They are not mixed together and are not used for ‖u‖ stats.

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
# Dry run (25 samples)
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp_dryrun --dry-run 5 --workers 16

# Full cohort (all subjects)
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp --workers 16

# Subset full run (100 subjects → ~70/15/15 + per-class ratios in each split)
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp_100 --max-subjects 100 --workers 16
"""

from __future__ import annotations

import argparse
import csv
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
DEFORM_RATIOS = {
    "none": 0.05,
    "rigid": 0.20,
    "affine": 0.25,
    "elastic": 0.25,
    "affine_elastic": 0.25,
}
DEFORM_SUFFIX = {
    "none": "none",
    "rigid": "rig",
    "affine": "aff",
    "elastic": "ela",
    "affine_elastic": "aela",
}

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

# After identity_grid_mask zeros OOB voxels, also zero this many voxels from each face.
U_BOUNDARY_MARGIN = 12
# Then clip ‖u‖ outliers at this percentile (over nonzero voxels only).
U_CLIP_PERCENTILE = 99.9

# Kept for legacy 2D helpers that import this module.
INTERIOR_MARGIN = 10


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


@dataclass(frozen=True)
class SampleStats:
    subject_id: str
    split: str
    deformation_class: str
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


def zero_u_boundary(u: np.ndarray, margin: int = U_BOUNDARY_MARGIN) -> np.ndarray:
    """Zero displacement vectors within ``margin`` voxels of any volume face."""
    out = u.astype(np.float32, copy=True)
    shape_xyz = (int(out.shape[1]), int(out.shape[2]), int(out.shape[3]))
    keep = interior_valid_mask(shape_xyz, margin)
    out[:, ~keep] = 0.0
    return out


def clip_u_at_percentile(u: np.ndarray, percentile: float = U_CLIP_PERCENTILE) -> np.ndarray:
    """
    Scale vectors with ‖u‖ above the given percentile down to that threshold.

    Call last, after OOB + boundary zeroing. Percentile is over nonzero ‖u‖ only
    so the zeroed border does not collapse the threshold.
    """
    out = u.astype(np.float32, copy=True)
    mag = displacement_magnitude(out.astype(np.float64))
    positive = mag > 0.0
    if not np.any(positive):
        return out
    thresh = float(np.percentile(mag[positive], percentile))
    if thresh <= 0.0 or not np.isfinite(thresh):
        return out
    over = mag > thresh
    if not np.any(over):
        return out
    scale = np.ones_like(mag, dtype=np.float64)
    scale[over] = thresh / mag[over]
    out *= scale.astype(np.float32)
    return out


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


def process_one_subject(task: Task, *, pin_threads: bool) -> SampleStats:
    if pin_threads:
        _pin_worker_cpu_threads()

    source, source_affine = _load_nifti_with_meta(task.t1_path)
    mask = _load_nifti(task.mask_path)
    shape_xyz = (int(source.shape[0]), int(source.shape[1]), int(source.shape[2]))
    identity_grid = _build_identity_grid(shape_xyz)
    transform = build_transform(task.deformation_class)
    source_mask_bin = mask > 0.5

    torch.manual_seed(task.seed % (2**31))
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
    moving = transformed.mri.data.squeeze(0).numpy()
    moving_mask_bin = transformed.brain_mask.data.squeeze(0).numpy() > 0.5
    cand_valid_mask = transformed.valid_mask.data.squeeze(0).numpy()
    u = (
        transformed.grid.data.squeeze(0).numpy() - identity_grid.numpy()
    ).astype(np.float32)
    # identity_grid_mask False → OOB; zero those displacements (‖u‖ = 0).
    identity_grid_mask_bin = cand_valid_mask > 0.99
    u[:, ~identity_grid_mask_bin] = 0.0
    u = zero_u_boundary(u, U_BOUNDARY_MARGIN)
    u = clip_u_at_percentile(u, U_CLIP_PERCENTILE)

    source_z, moving_z, _, _ = zscore_brain_pair(
        source, moving, source_mask_bin, moving_mask_bin
    )

    Path(task.out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.out_path,
        source=source_z,
        moving=moving_z,
        u=u.astype(np.float32),
        source_mask=source_mask_bin.astype(bool),
        moving_mask=moving_mask_bin.astype(bool),
        identity_grid_mask=identity_grid_mask_bin.astype(bool),
        source_affine=source_affine.astype(np.float32),
        deformation_class=np.array(task.deformation_class),
        subject_id=np.array(task.subject_id),
    )

    rel = Path(task.out_path).name if not task.split else f"{task.split}/{Path(task.out_path).name}"
    return SampleStats(
        subject_id=task.subject_id,
        split=task.split,
        deformation_class=task.deformation_class,
        rel_path=rel,
    )


def _worker_create_sample(task: Task) -> SampleStats:
    _pin_worker_cpu_threads()
    return process_one_subject(task, pin_threads=False)


def build_deformation_stats(samples: list[SampleStats]) -> dict[str, dict[str, int]]:
    """Per-class sample counts."""
    buckets: dict[str, int] = {}
    for s in samples:
        buckets[s.deformation_class] = buckets.get(s.deformation_class, 0) + 1
    return {cls: {"count": n} for cls, n in sorted(buckets.items())}


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
    quotas = compute_quotas(len(order), DEFORM_RATIOS)
    cls_list: list[str] = []
    for cls in DEFORM_RATIOS:
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
) -> list[Task]:
    """Dry run: N samples per class; distinct subjects within each class when possible."""
    if len(subjects) < replicates:
        raise ValueError(
            f"Dry run needs ≥{replicates} subjects for distinct subjects within "
            f"each class; found {len(subjects)}"
        )
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
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
                    seed=seed,
                )
            )
            task_idx += 1
    return tasks


def compute_class_u_stats_table(
    out_root: Path,
    samples: list[SampleStats],
) -> list[dict[str, float | int | str]]:
    """
    Per-class ‖u‖ summary: compute min/Q1/mean/Q3/max on each sample's full
    volume, then average those scalars across samples in the class.
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
        per_sample: list[dict[str, float]] = []
        for fp in by_class[cls]:
            with np.load(fp) as z:
                vals = displacement_magnitude(np.asarray(z["u"], dtype=np.float64)).ravel()
            per_sample.append(
                {
                    "min": float(np.min(vals)),
                    "q1": float(np.percentile(vals, 25)),
                    "mean": float(np.mean(vals)),
                    "q3": float(np.percentile(vals, 75)),
                    "max": float(np.max(vals)),
                }
            )
        if not per_sample:
            rows.append(
                {
                    "deformation_class": cls,
                    "n_samples": 0,
                    "min": float("nan"),
                    "q1": float("nan"),
                    "mean": float("nan"),
                    "q3": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "deformation_class": cls,
                "n_samples": len(per_sample),
                "min": float(np.mean([s["min"] for s in per_sample])),
                "q1": float(np.mean([s["q1"] for s in per_sample])),
                "mean": float(np.mean([s["mean"] for s in per_sample])),
                "q3": float(np.mean([s["q3"] for s in per_sample])),
                "max": float(np.mean([s["max"] for s in per_sample])),
            }
        )
    return rows


def print_and_save_class_u_stats(out_root: Path, samples: list[SampleStats]) -> Path:
    rows = compute_class_u_stats_table(out_root, samples)
    print("\nPer-class ‖u‖ stats (per-sample full-volume stats, then mean over samples):")
    print(
        f"{'class':<16} {'n':>3} {'min':>8} {'Q1':>8} {'mean':>8} {'Q3':>8} {'max':>8}"
    )
    for r in rows:
        print(
            f"{r['deformation_class']:<16} {r['n_samples']:>3} "
            f"{r['min']:8.3f} {r['q1']:8.3f} {r['mean']:8.3f} "
            f"{r['q3']:8.3f} {r['max']:8.3f}"
        )
    csv_path = out_root / "dryrun_class_u_stats.csv"
    fieldnames = ["deformation_class", "n_samples", "min", "q1", "mean", "q3", "max"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path}")
    return csv_path


def compute_split_class_u_stats_table(
    out_root: Path,
    samples: list[SampleStats],
) -> list[dict[str, float | int | str]]:
    """
    Full-run ‖u‖ summary: per (split, class), compute min/Q1/mean/Q3/max on each
    sample, then average those scalars across samples in that bucket.
    """
    splits = ("Train", "Val", "Test")
    by_key: dict[tuple[str, str], list[Path]] = {
        (sp, cls): [] for sp in splits for cls in DRY_RUN_CLASSES
    }
    for s in samples:
        key = (s.split, s.deformation_class)
        if key not in by_key:
            continue
        fp = out_root / s.rel_path
        if fp.is_file():
            by_key[key].append(fp)

    rows: list[dict[str, float | int | str]] = []
    for sp in splits:
        for cls in DRY_RUN_CLASSES:
            paths = by_key[(sp, cls)]
            per_sample: list[dict[str, float]] = []
            for fp in paths:
                with np.load(fp) as z:
                    vals = displacement_magnitude(
                        np.asarray(z["u"], dtype=np.float64)
                    ).ravel()
                per_sample.append(
                    {
                        "min": float(np.min(vals)),
                        "q1": float(np.percentile(vals, 25)),
                        "mean": float(np.mean(vals)),
                        "q3": float(np.percentile(vals, 75)),
                        "max": float(np.max(vals)),
                    }
                )
            if not per_sample:
                rows.append(
                    {
                        "split": sp,
                        "deformation_class": cls,
                        "n_samples": 0,
                        "min": float("nan"),
                        "q1": float("nan"),
                        "mean": float("nan"),
                        "q3": float("nan"),
                        "max": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "split": sp,
                    "deformation_class": cls,
                    "n_samples": len(per_sample),
                    "min": float(np.mean([x["min"] for x in per_sample])),
                    "q1": float(np.mean([x["q1"] for x in per_sample])),
                    "mean": float(np.mean([x["mean"] for x in per_sample])),
                    "q3": float(np.mean([x["q3"] for x in per_sample])),
                    "max": float(np.mean([x["max"] for x in per_sample])),
                }
            )
    return rows


def print_and_save_split_class_u_stats(
    out_root: Path, samples: list[SampleStats]
) -> Path:
    rows = compute_split_class_u_stats_table(out_root, samples)
    print(
        "\nPer-split × class ‖u‖ stats "
        "(per-sample full-volume stats, then mean over samples):"
    )
    print(
        f"{'split':<6} {'class':<16} {'n':>3} "
        f"{'min':>8} {'Q1':>8} {'mean':>8} {'Q3':>8} {'max':>8}"
    )
    for r in rows:
        print(
            f"{r['split']:<6} {r['deformation_class']:<16} {r['n_samples']:>3} "
            f"{r['min']:8.3f} {r['q1']:8.3f} {r['mean']:8.3f} "
            f"{r['q3']:8.3f} {r['max']:8.3f}"
        )
    csv_path = out_root / "split_class_u_stats.csv"
    fieldnames = [
        "split",
        "deformation_class",
        "n_samples",
        "min",
        "q1",
        "mean",
        "q3",
        "max",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path}")
    return csv_path


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
    dry_run: bool = False,
    dry_run_replicates: int = 5,
) -> None:
    in_root = Path(input_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, workers if workers is not None else _default_parallel_workers())
    all_stats: list[SampleStats] = []

    subjects = collect_hcp_subjects(in_root)
    if not subjects:
        print(f"No HCP subjects found in: {in_root}")
        return
    if max_subjects is not None and max_subjects > 0:
        subjects = subjects[:max_subjects]
    print(f"Subjects: {len(subjects)} from {in_root} → {out_root}")

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
        print(
            f"u cleanup: identity_grid_mask OOB → {U_BOUNDARY_MARGIN}-voxel border → "
            f"p{U_CLIP_PERCENTILE:g} clip"
        )
        tasks = build_dry_run_tasks(
            subjects,
            out_root,
            replicates=dry_run_replicates,
            base_seed=base_seed,
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
            out_dir.mkdir(parents=True, exist_ok=True)
            c = Counter()
            for idx, e in enumerate(entries):
                cls = sid_to_class[e.subject_id]
                suf = DEFORM_SUFFIX[cls]
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
            "Full run splits: "
            + ", ".join(f"{sp}={len(split_subjects[sp])}" for sp in ("Train", "Val", "Test"))
        )
        for sp in ("Train", "Val", "Test"):
            counts = split_summary.get(sp, {})
            if counts:
                mix = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"  {sp} classes: {mix}")
        print(
            f"u cleanup: identity_grid_mask OOB → {U_BOUNDARY_MARGIN}-voxel border → "
            f"p{U_CLIP_PERCENTILE:g} clip"
        )

    print(f"Generating {len(tasks)} NPZs with {n_workers} worker(s)…")

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Create HCP synth"):
            all_stats.append(process_one_subject(t, pin_threads=False))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_create_sample, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Create HCP synth"):
                all_stats.append(fut.result())

    print(f"Wrote {len(all_stats)} NPZs")
    meta = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "seed": base_seed,
        "dry_run": dry_run,
        "dry_run_replicates": dry_run_replicates if dry_run else None,
        "dry_run_classes": list(DRY_RUN_CLASSES) if dry_run else None,
        "u_boundary_margin": U_BOUNDARY_MARGIN,
        "u_clip_percentile": U_CLIP_PERCENTILE,
        "transform_params": {
            "rigid": PARAM_RIGID,
            "affine": PARAM_AFFINE,
            "elastic": PARAM_ELASTIC,
            "affine_elastic": {"affine": PARAM_AFFINE, "elastic": PARAM_ELASTIC_IN_AFFINE},
        },
        "n_subjects": len(subjects),
        "n_tasks": len(tasks),
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
            "source_affine",
            "deformation_class",
            "subject_id",
        ],
    }
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if dry_run:
        print_and_save_class_u_stats(out_root, all_stats)
    else:
        print_and_save_split_class_u_stats(out_root, all_stats)


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
        default="datasets/synth-data/torchio/hcp",
        help="Output root (Train/Val/Test, or flat folder for --dry-run).",
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
        workers=args.workers,
        base_seed=args.seed,
        max_subjects=args.max_subjects,
        dry_run=dry_run,
        dry_run_replicates=dry_run_replicates,
    )
    print("Finished! HCP synthetic data is ready.")
