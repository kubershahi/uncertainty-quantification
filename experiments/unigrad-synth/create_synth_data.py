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
  - source_mask    : fixed brain mask (bool volume)
  - moving_mask    : source_mask warped with the same transform (bool volume)
  - moving_grid        : transformed binary grid-lines mask for QC visualization (bool volume)
  - source_grid        : undeformed binary grid-lines mask for QC visualization (bool volume)
  - identity_grid_mask : binary in-bounds mask for displacement field (bool volume)
  - source_affine
  - deformation_class : one of {none, rigid, affine, elastic, affine_elastic}
  - magnitude_range : low | mid | high | none (TorchIO sampling envelope, not realized ‖u‖)
  - u_max_interior, u_mean_interior
  - subject_id, qc_passed

TorchIO transforms run in physical space (mm) using the NIfTI affine; ``u`` is recovered
in voxel index space via the identity-grid trick (backward warp).

Split policy:
  - deterministic 70/15/15 by subject hash (Train/Val/Test)
  - balanced deformation mix in each split:
      5% none, 20% rigid, 25% affine, 25% elastic, 25% affine+elastic

Example:
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --qc-fail-path datasets/hcp_synth_qc_fail --workers 8
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --max-subjects 30 --workers 4
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth_dryrun --range-grid --no-qc --workers 4
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth_dryrun --range-grid 5 --no-qc
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

# Magnitude ranges (within each deformation class except none).
# Each range sets TorchIO *sampling bounds*; RandomAffine/Elastic still draw uniformly inside them.
MAGNITUDE_RANGES = ("low", "mid", "high")
MAGNITUDE_RANGE_FRAC = {"low": 0.30, "mid": 0.40, "high": 0.30}

# Per-range transform parameter bounds (TorchIO: degrees, translation mm, scales, elastic mm)
RANGE_RIGID = {
    "low": {"degrees": 3.0, "translation_mm": 2.0},
    "mid": {"degrees": 6.0, "translation_mm": 4.0},
    "high": {"degrees": 10.0, "translation_mm": 6.0},
}
RANGE_AFFINE = {
    "low": {"scales": (0.98, 1.02), "degrees": 5.0, "translation_mm": 3.0},
    "mid": {"scales": (0.97, 1.03), "degrees": 8.0, "translation_mm": 4.0},
    "high": {"scales": (0.95, 1.05), "degrees": 12.0, "translation_mm": 7.0},
}
RANGE_ELASTIC = {
    "low": {"max_disp_mm": 3.0},
    "mid": {"max_disp_mm": 6.0},
    "high": {"max_disp_mm": 9.0},
}

ELASTIC_NUM_CONTROL_POINTS = 7

# QC checks — max caps (all classes)
INTERIOR_MARGIN = 10
MAX_U_INTERIOR_VOX = 25.0
MAX_U_GLOBAL_VOX = 60.0
MIN_MOVING_MEAN_RATIO = 0.05
MAX_TRANSFORM_ATTEMPTS = 20
MAX_U_NONE_VOX = 0.5
# Smooth valid-boundary vectors to reduce sharp seams in u_true vs u_pred comparisons.
U_BOUNDARY_SMOOTH_BAND = 2

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
    magnitude_range: str
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
    magnitude_range: str
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


def u_interior_stats(
    u: np.ndarray,
    mask: np.ndarray,
    shape_xyz: tuple[int, int, int],
    interior_margin: int,
) -> tuple[float, float]:
    mag = displacement_magnitude(u.astype(np.float64))
    interior = interior_valid_mask(shape_xyz, interior_margin)
    valid = interior & (mask > 0.5)
    if valid.any():
        vals = mag[valid]
        return float(np.max(vals)), float(np.mean(vals))
    return float(np.max(mag)), float(np.mean(mag))


def passes_checks(
    u: np.ndarray,
    source: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    deformation_class: str,
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
        region_mean = float(np.mean(mag[valid]))
    else:
        region_max = full_max
        region_mean = full_max

    if deformation_class == "none":
        if region_max >= MAX_U_NONE_VOX:
            return False
    else:
        min_max = MIN_U_MAX_INTERIOR_BY_CLASS.get(deformation_class)
        min_mean = MIN_U_MEAN_INTERIOR_BY_CLASS.get(deformation_class)
        if min_max is not None and region_max < min_max:
            return False
        if min_mean is not None and region_mean < min_mean:
            return False

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


def iter_class_range_combinations() -> list[tuple[str, str]]:
    """All (deformation_class, magnitude_range) pairs for range-grid dry runs."""
    combos: list[tuple[str, str]] = [("none", "none")]
    for cls in ("rigid", "affine", "elastic", "affine_elastic"):
        for mag_range in MAGNITUDE_RANGES:
            combos.append((cls, mag_range))
    return combos


def range_grid_filename(subject_id: str, deformation_class: str, mag_range: str, replicate: int) -> str:
    """e.g. 100206_rigid_low.npz or 100206_rigid_low_02.npz when replicate > 0."""
    base = f"{subject_id}_{deformation_class}_{mag_range}"
    if replicate > 0:
        return f"{base}_{replicate:02d}.npz"
    return f"{base}.npz"


def assign_magnitude_range(subject_id: str, deformation_class: str, seed: int) -> str:
    if deformation_class == "none":
        return "none"
    h = stable_subject_hash(f"{subject_id}:{deformation_class}", seed) % 10000 / 10000.0
    if h < MAGNITUDE_RANGE_FRAC["low"]:
        return "low"
    if h < MAGNITUDE_RANGE_FRAC["low"] + MAGNITUDE_RANGE_FRAC["mid"]:
        return "mid"
    return "high"


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


def build_transform(deformation_class: str, mag_range: str = "mid") -> tio.Transform:
    if deformation_class == "none":
        return tio.Compose([])
    if deformation_class == "rigid":
        p = RANGE_RIGID[mag_range]
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
        p = RANGE_AFFINE[mag_range]
        return tio.Compose([_affine_transform(**p)])
    if deformation_class == "elastic":
        p = RANGE_ELASTIC[mag_range]
        return tio.Compose([_elastic_transform(**p)])
    if deformation_class == "affine_elastic":
        ap = RANGE_AFFINE[mag_range]
        ep = RANGE_ELASTIC[mag_range]
        return tio.Compose([_affine_transform(**ap), _elastic_transform(**ep)])
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


def _build_binary_grid_mask(shape_xyz: tuple[int, int, int], stride: int = 12) -> np.ndarray:
    """Grid-line volume (1 on lines, 0 elsewhere) in voxel index space."""
    x, y, z = shape_xyz
    gx = (np.arange(x) % stride) == 0
    gy = (np.arange(y) % stride) == 0
    gz = (np.arange(z) % stride) == 0
    line = (
        gx[:, None, None]
        | gy[None, :, None]
        | gz[None, None, :]
    )
    return line.astype(np.float32)


def smooth_u_near_boundary(
    u: np.ndarray, valid_mask: np.ndarray, *, band_vox: int = U_BOUNDARY_SMOOTH_BAND
) -> np.ndarray:
    """
    Smooth displacement magnitude in a narrow valid-band near OOB boundary.

    Steps:
    - keep invalid vectors exactly 0
    - keep far-interior vectors unchanged
    - only in boundary band, smooth |u| then rescale vectors to match smoothed magnitude
    """
    if band_vox <= 0:
        return u
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except Exception:
        return u

    valid = valid_mask.astype(bool)
    if not np.any(valid):
        return u
    dist_to_invalid = distance_transform_edt(valid)
    boundary_band = valid & (dist_to_invalid <= float(band_vox))
    if not np.any(boundary_band):
        return u

    out = u.copy()
    mag = displacement_magnitude(out.astype(np.float64))
    mag_smooth = gaussian_filter(mag, sigma=1.0, mode="nearest")
    scale = np.ones_like(mag, dtype=np.float32)
    denom = np.maximum(mag, 1e-6)
    scale[boundary_band] = (mag_smooth[boundary_band] / denom[boundary_band]).astype(np.float32)

    out = out * scale[None, ...]
    out[:, ~valid] = 0.0
    return out.astype(np.float32)


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
    transform = build_transform(task.deformation_class, task.magnitude_range)

    moving: np.ndarray | None = None
    moving_mask_bin: np.ndarray | None = None
    moving_grid_bin: np.ndarray | None = None
    u: np.ndarray | None = None
    qc_passed = task.skip_qc
    n_attempts = 1 if task.skip_qc else MAX_TRANSFORM_ATTEMPTS
    source_mask_bin = mask > 0.5
    # Must match visualize_synth_data's default grid stride for consistent overlays.
    binary_grid = _build_binary_grid_mask(shape_xyz, stride=20)

    identity_grid_mask_bin: np.ndarray | None = None

    for attempt in range(n_attempts):
        torch.manual_seed((task.seed + attempt * 100003) % (2**31))
        img_tensor = torch.from_numpy(source).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(source_mask_bin.astype(np.float32)).unsqueeze(0)
        ones_tensor = torch.ones_like(img_tensor, dtype=torch.float32)
        gridline_tensor = torch.from_numpy(binary_grid).unsqueeze(0).float()
        affine = source_affine.astype(np.float64)
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=img_tensor, affine=affine),
            grid=tio.ScalarImage(tensor=identity_grid.clone(), affine=affine),
            brain_mask=tio.LabelMap(tensor=mask_tensor, affine=affine),
            valid_mask=tio.LabelMap(tensor=ones_tensor, affine=affine),
            grid_lines=tio.LabelMap(tensor=gridline_tensor, affine=affine),
        )
        transformed = transform(subject)
        cand_moving = transformed.mri.data.squeeze(0).numpy()
        cand_moving_mask = transformed.brain_mask.data.squeeze(0).numpy() > 0.5
        cand_valid_mask = transformed.valid_mask.data.squeeze(0).numpy()
        cand_moving_grid = transformed.grid_lines.data.squeeze(0).numpy() > 0.5
        cand_u = (
            transformed.grid.data.squeeze(0).numpy() - identity_grid.numpy()
        ).astype(np.float32)
        identity_grid_mask_bin = cand_valid_mask > 0.99
        invalid = ~identity_grid_mask_bin
        cand_u[:, invalid] = 0.0
        cand_u = smooth_u_near_boundary(cand_u, identity_grid_mask_bin, band_vox=U_BOUNDARY_SMOOTH_BAND)
        cand_moving_grid[invalid] = False

        moving, moving_mask_bin, moving_grid_bin, u = (
            cand_moving,
            cand_moving_mask,
            cand_moving_grid,
            cand_u,
        )

        if task.skip_qc or passes_checks(
            cand_u,
            source,
            cand_moving,
            mask,
            task.deformation_class,
            interior_margin=INTERIOR_MARGIN,
            max_u_interior_vox=max_u_interior_vox,
            max_u_global_vox=max_u_global_vox,
            min_moving_mean_ratio=min_moving_mean_ratio,
        ):
            qc_passed = True
            break

    assert (
        moving is not None
        and moving_mask_bin is not None
        and moving_grid_bin is not None
        and identity_grid_mask_bin is not None
        and u is not None
    )
    u_max_int, u_mean_int = u_interior_stats(u, mask, shape_xyz, INTERIOR_MARGIN)
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
        moving_grid=moving_grid_bin.astype(bool),
        source_grid=binary_grid.astype(bool),
        identity_grid_mask=identity_grid_mask_bin.astype(bool),
        source_affine=source_affine.astype(np.float32),
        deformation_class=np.array(task.deformation_class),
        magnitude_range=np.array(task.magnitude_range),
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
        magnitude_range=task.magnitude_range,
        qc_passed=qc_passed,
        u_max_interior=u_max_int,
        u_mean_interior=u_mean_int,
        rel_path=rel,
    )


def _qc_fail_warning(stats: SampleStats) -> str:
    cls = stats.deformation_class
    mag_range = stats.magnitude_range
    min_max = MIN_U_MAX_INTERIOR_BY_CLASS.get(cls)
    min_mean = MIN_U_MEAN_INTERIOR_BY_CLASS.get(cls)
    return (
        f"QC_FAIL (saved to qc_fail): {stats.rel_path} | "
        f"class={cls} range={mag_range} | "
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


def build_deformation_stats(samples: list[SampleStats]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-class/range counts and u_max_interior p50 summaries."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        if not s.qc_passed:
            continue
        buckets.setdefault(s.deformation_class, {}).setdefault(s.magnitude_range, []).append(
            s.u_max_interior
        )
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for cls, ranges in sorted(buckets.items()):
        out[cls] = {}
        for mag_range, vals in sorted(ranges.items()):
            arr = np.asarray(vals, dtype=np.float64)
            out[cls][mag_range] = {
                "count": int(arr.size),
                "u_max_p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
                "u_mean_p50": float(
                    np.percentile(
                        [s.u_mean_interior for s in samples if s.qc_passed and s.deformation_class == cls and s.magnitude_range == mag_range],
                        50,
                    )
                )
                if arr.size
                else 0.0,
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


def build_range_grid_tasks(
    subjects: list[SubjectEntry],
    out_root: Path,
    *,
    replicates: int,
    base_seed: int,
    skip_qc: bool,
) -> list[Task]:
    """One task per (class, range) × replicate; filenames include class and range labels."""
    combos = iter_class_range_combinations()
    out_dir = out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = out_root / "_unused_qc_fail"
    tasks: list[Task] = []
    task_idx = 0
    for cls, mag_range in combos:
        # Keep source subject fixed within a (class, range) group so replicate rows are comparable.
        subj_idx = stable_subject_hash(f"{cls}:{mag_range}", base_seed) % len(subjects)
        subj = subjects[subj_idx]
        for rep in range(replicates):
            rep_suffix = rep + 1 if replicates > 1 else 0
            out_name = range_grid_filename(subj.subject_id, cls, mag_range, rep_suffix)
            seed = (
                base_seed
                + stable_subject_hash(f"{cls}:{mag_range}:{rep}:{subj.subject_id}", base_seed)
                + task_idx * 100003
            ) % (2**31)
            tasks.append(
                Task(
                    subject_id=subj.subject_id,
                    split="",
                    deformation_class=cls,
                    magnitude_range=mag_range,
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
    range_grid: bool = False,
    range_grid_replicates: int = 3,
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

    if range_grid:
        combos = iter_class_range_combinations()
        n_tasks = len(combos) * range_grid_replicates
        print(
            f"Range-grid dry run: {len(combos)} (class, range) combinations × "
            f"{range_grid_replicates} replicates = {n_tasks} samples"
        )
        print(f"Combinations: {', '.join(f'{c}_{r}' for c, r in combos)}")
        if skip_qc:
            print("QC disabled (--no-qc): single random draw per task, all saved to output-path")
        tasks = build_range_grid_tasks(
            subjects,
            out_root,
            replicates=range_grid_replicates,
            base_seed=base_seed,
            skip_qc=skip_qc,
        )
        split_summary: dict[str, dict[str, int]] = {}
        range_summary: dict[str, dict[str, int]] = {}
        split_subjects = {"": subjects}
    else:
        split_subjects: dict[str, list[SubjectEntry]] = {"Train": [], "Val": [], "Test": []}
        for s in subjects:
            split_subjects[assign_split(s.subject_id, base_seed)].append(s)

        tasks = []
        split_summary = {}
        range_summary = {}
        for split, entries in split_subjects.items():
            entries = sorted(entries, key=lambda e: e.subject_id)
            sid_to_class = assign_deformation_classes(
                [e.subject_id for e in entries], seed=base_seed + stable_subject_hash(split, base_seed)
            )
            out_dir = out_root / split
            fail_dir = fail_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            fail_dir.mkdir(parents=True, exist_ok=True)
            c = Counter()
            tc = Counter()
            for idx, e in enumerate(entries):
                cls = sid_to_class[e.subject_id]
                mag_range = assign_magnitude_range(e.subject_id, cls, base_seed)
                suf = DEFORMATION_SUFFIX[cls]
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
                        magnitude_range=mag_range,
                        t1_path=e.t1_path,
                        mask_path=e.mask_path,
                        out_path=str(out_path),
                        qc_fail_out_path=str(fail_path),
                        seed=seed,
                        skip_qc=skip_qc,
                    )
                )
                c[cls] += 1
                tc[f"{cls}:{mag_range}"] += 1
            split_summary[split] = dict(c)
            range_summary[split] = dict(tc)

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
        "range_grid": range_grid,
        "range_grid_replicates": range_grid_replicates if range_grid else None,
        "range_grid_combinations": (
            [f"{c}_{r}" for c, r in iter_class_range_combinations()] if range_grid else None
        ),
        "n_subjects": len(subjects),
        "n_tasks": len(tasks),
        "qc_passed_count": passed,
        "qc_failed_count": failed,
        "split_counts": {k: len(v) for k, v in split_subjects.items()},
        "deformation_ratios_target": DEFORMATION_RATIOS if not range_grid else None,
        "deformation_counts_actual": split_summary,
        "magnitude_range_counts": range_summary,
        "deformation_stats": build_deformation_stats(all_stats),
        "field_names": [
            "source",
            "moving",
            "u",
            "mask",
            "source_mask",
            "moving_mask",
            "moving_grid",
            "magnitude_range",
            "u_max_interior",
            "u_mean_interior",
        ],
    }
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

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
        "--range-grid",
        nargs="?",
        const=3,
        default=None,
        type=int,
        metavar="N",
        help=(
            "Dry run: all 13 (class, range) combos into output-path/ (flat, no split folders). "
            "Optional N = replicates per combo (default 3 when flag is given alone → 39 samples)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    range_grid = args.range_grid is not None
    range_grid_replicates = args.range_grid if range_grid else 3
    create_synthetic_data(
        args.input_path,
        args.output_path,
        qc_fail_root=args.qc_fail_path,
        workers=args.workers,
        base_seed=args.seed,
        max_subjects=args.max_subjects,
        skip_qc=args.no_qc,
        range_grid=range_grid,
        range_grid_replicates=range_grid_replicates,
    )
    print("Finished! HCP synthetic data is ready.")
