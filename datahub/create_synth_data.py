"""
Build synthetic registration triplets (fixed, warped, phi) for IXI_2D slices.

phi_true is warped_grid - identity_grid with no edits — same random sample as `warped`.

If QC checks never pass within MAX_TRANSFORM_ATTEMPTS, the last sample is still saved with `qc_passed=False` in the .npz.
Paths are also listed in qc_flagged_paths.txt under output_path for easy cleanup.

Example:
  python create_synthetic_data.py
  python create_synthetic_data.py --workers 64
  python create_synthetic_data.py --max-phi-int 25
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

# =============================================================================
# CONFIG — edit here (TorchIO + retry / acceptance checks)
# =============================================================================

# RandomAffine
AFFINE_SCALES = (0.97, 1.03)
AFFINE_DEGREES = 8.0
AFFINE_TRANSLATION_PX = 4.0
AFFINE_P = 0.5

# RandomElasticDeformation
ELASTIC_NUM_CONTROL_POINTS = 7
ELASTIC_MAX_DISP_MM = 6.0
ELASTIC_P = 0.6

# Retry until phi / warped pass; if still failing, save last draw with qc_passed=False
INTERIOR_MARGIN = 10
# Max |phi| on valid_mask only (pixels inset from image boundary by INTERIOR_MARGIN). None = skip.
MAX_PHI_INTERIOR_PX = 25.0
# Max |phi| over the entire slice (includes boundary / padding artifacts). None = skip.
MAX_PHI_GLOBAL_PX = 60.0
# Reject if mean(warped) < MIN_WARPED_MEAN_RATIO * mean(image) — a *lower floor*, not ±5%.
# Example 0.05: warped mean must be ≥ 5% of fixed-image mean (blocks mostly black warps).
# None = do not check.
MIN_WARPED_MEAN_RATIO = 0.05
# Maximum number of attempts to generate a valid triplet
MAX_TRANSFORM_ATTEMPTS = 20


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def interior_valid_mask(w: int, h: int, margin: int) -> np.ndarray:
    """True where pixel is at least `margin` away from the image boundary (for checks + Phase 3 loss)."""
    mask = np.zeros((w, h), dtype=bool)
    if w > 2 * margin and h > 2 * margin:
        mask[margin : w - margin, margin : h - margin] = True
    else:
        mask[:] = True
    return mask


def passes_checks(
    phi: np.ndarray,
    image: np.ndarray,
    warped: np.ndarray,
    interior_margin: int,
    max_phi_interior_px: float | None,
    max_phi_global_px: float | None,
    min_warped_mean_ratio: float | None,
) -> bool:
    mag = phi_magnitude(phi.astype(np.float64))
    slice_max = float(np.max(mag))

    m = interior_valid_mask(image.shape[0], image.shape[1], interior_margin)
    if m.any():
        region_max = float(np.max(mag[m]))
    else:
        region_max = slice_max

    if max_phi_global_px is not None and slice_max > max_phi_global_px:
        return False

    if max_phi_interior_px is not None and region_max > max_phi_interior_px:
        return False

    if min_warped_mean_ratio is not None:
        imu = float(np.mean(image))
        wmu = float(np.mean(warped))
        floor = max(1e-6, imu * min_warped_mean_ratio)
        if wmu < floor:
            return False

    return True


def build_transform(
    *,
    affine_p: float | None = None,
    elastic_p: float | None = None,
) -> tio.Compose:
    """
    RandomAffine then RandomElasticDeformation with shared geometry from CONFIG.

    ``affine_p`` / ``elastic_p`` default to ``AFFINE_P`` / ``ELASTIC_P``; pass
    e.g. ``1.0`` to force both transforms to apply every call (no identity skip).
    """
    ap = AFFINE_P if affine_p is None else affine_p
    ep = ELASTIC_P if elastic_p is None else elastic_p
    return tio.Compose(
        [
            tio.RandomAffine(
                scales=AFFINE_SCALES,
                degrees=AFFINE_DEGREES,
                translation=(AFFINE_TRANSLATION_PX, AFFINE_TRANSLATION_PX, 0),
                default_pad_value="minimum",
                p=ap,
            ),
            tio.RandomElasticDeformation(
                num_control_points=ELASTIC_NUM_CONTROL_POINTS,
                max_displacement=(ELASTIC_MAX_DISP_MM, ELASTIC_MAX_DISP_MM, 0),
                locked_borders=2,
                p=ep,
            ),
        ]
    )


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


def process_one_triplet_file(
    in_path: str,
    out_path: str,
    rel_flag: str,
    *,
    max_phi_interior_px: float | None,
    max_phi_global_px: float | None,
    min_warped_mean_ratio: float | None,
    seed: int,
    pin_threads: bool,
) -> tuple[bool, str | None]:
    """
    Load ``in_path`` *.npy, sample transforms, write ``out_path`` *_triplet.npz.

    Returns ``(qc_passed, warn_line_or_none)`` for QC-fail logging.
    """
    if pin_threads:
        _pin_worker_cpu_threads()

    img_2d = np.load(in_path)
    w, h = img_2d.shape
    grid_x, grid_y = torch.meshgrid(
        torch.arange(w, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        indexing="ij",
    )
    identity_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(-1).float()

    transform = build_transform()
    warped_img: np.ndarray | None = None
    phi_true: np.ndarray | None = None
    qc_passed = False

    for attempt in range(MAX_TRANSFORM_ATTEMPTS):
        torch.manual_seed((seed + attempt * 100003) % (2**31))
        img_tensor = torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(-1).float()
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=img_tensor),
            grid=tio.ScalarImage(tensor=identity_grid.clone()),
        )
        transformed = transform(subject)
        cand_warped = transformed.mri.data.squeeze().numpy()
        cand_phi = (
            transformed.grid.data.squeeze().numpy() - identity_grid.squeeze().numpy()
        ).astype(np.float32)

        warped_img, phi_true = cand_warped, cand_phi

        if passes_checks(
            cand_phi,
            img_2d,
            cand_warped,
            interior_margin=INTERIOR_MARGIN,
            max_phi_interior_px=max_phi_interior_px,
            max_phi_global_px=max_phi_global_px,
            min_warped_mean_ratio=min_warped_mean_ratio,
        ):
            qc_passed = True
            break

    assert warped_img is not None and phi_true is not None

    valid_mask = interior_valid_mask(w, h, INTERIOR_MARGIN)
    np.savez_compressed(
        out_path,
        image=img_2d,
        warped=warped_img,
        phi=phi_true,
        valid_mask=valid_mask,
        qc_passed=qc_passed,
    )

    if qc_passed:
        return True, None

    mag = phi_magnitude(phi_true.astype(np.float64))
    mi = interior_valid_mask(w, h, INTERIOR_MARGIN)
    mx_valid = float(np.max(mag[mi])) if mi.any() else float("nan")
    mx_full = float(np.max(mag))
    lim_int = max_phi_interior_px if max_phi_interior_px is not None else "off"
    lim_glob = max_phi_global_px if max_phi_global_px is not None else "off"
    warn = (
        f"QC_FAIL (saved, qc_passed=False): {rel_flag} | "
        f"max|phi| valid_mask={mx_valid:.2f}  full_slice={mx_full:.2f}  "
        f"(limits: interior≤{lim_int} on valid_mask, global≤{lim_glob} on full slice)"
    )
    return False, warn


def _worker_create_triplet(
    task: tuple[str, str, str, float | None, float | None, float | None, int],
) -> tuple[str, bool, str | None]:
    """Picklable: (in_path, out_path, rel_flag, max_phi_interior, max_phi_global, min_warped_ratio, seed)."""
    in_path, out_path, rel_flag, max_phi_i, max_phi_g, min_warp, seed = task
    _pin_worker_cpu_threads()
    qc_ok, warn = process_one_triplet_file(
        in_path,
        out_path,
        rel_flag,
        max_phi_interior_px=max_phi_i,
        max_phi_global_px=max_phi_g,
        min_warped_mean_ratio=min_warp,
        seed=seed,
        pin_threads=False,
    )
    return rel_flag, qc_ok, warn


def create_synthetic_data(
    input_root: str,
    output_root: str,
    *,
    max_phi_interior_px_override: float | None = None,
    workers: int | None = None,
    base_seed: int = 42,
) -> None:
    max_phi_interior = (
        max_phi_interior_px_override
        if max_phi_interior_px_override is not None
        else MAX_PHI_INTERIOR_PX
    )
    n_workers = max(1, workers if workers is not None else _default_parallel_workers())

    splits = ["Train", "Val", "Test", "Atlas"]
    flagged_rel_paths: list[str] = []

    tasks: list[tuple[str, str, str, float | None, float | None, float | None, int]] = []
    task_index = 0
    for split in splits:
        input_dir = os.path.join(input_root, split)
        if not os.path.isdir(input_dir):
            print(f"Skipping {split} — missing directory {input_dir}")
            continue

        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))

        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Queueing {split} split ({len(files)} files)...")

        for fname in files:
            in_path = os.path.join(input_dir, fname)
            out_name = fname.replace(".npy", "_triplet.npz")
            out_path = os.path.join(output_dir, out_name)
            rel_flag = os.path.join(split, out_name)
            seed = (
                base_seed + task_index * 100003 + hash(in_path) % 999983
            ) % (2**31)
            tasks.append(
                (
                    in_path,
                    out_path,
                    rel_flag,
                    max_phi_interior,
                    MAX_PHI_GLOBAL_PX,
                    MIN_WARPED_MEAN_RATIO,
                    seed,
                )
            )
            task_index += 1

    if not tasks:
        print("No .npy files found under input splits.")
        manifest = os.path.join(output_root, "qc_flagged_paths.txt")
        if os.path.isfile(manifest):
            os.remove(manifest)
        return

    print(
        f"Parallel workers: {n_workers} (default = all logical CPUs; each worker uses 1 OpenMP thread)"
    )

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Create triplets"):
            in_path, out_path, rel_flag, max_i, max_g, min_w, seed = t
            qc_ok, warn = process_one_triplet_file(
                in_path,
                out_path,
                rel_flag,
                max_phi_interior_px=max_i,
                max_phi_global_px=max_g,
                min_warped_mean_ratio=min_w,
                seed=seed,
                pin_threads=False,
            )
            if not qc_ok:
                flagged_rel_paths.append(rel_flag)
                if warn:
                    tqdm.write(warn)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_create_triplet, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Create triplets"):
                rel_flag_ret, qc_ok, warn = fut.result()
                if not qc_ok:
                    flagged_rel_paths.append(rel_flag_ret)
                    if warn:
                        tqdm.write(warn)

    manifest = os.path.join(output_root, "qc_flagged_paths.txt")
    if flagged_rel_paths:
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("# Triplets with qc_passed=False — delete or regenerate\n")
            for line in sorted(flagged_rel_paths):
                f.write(f"{line}\n")
        print(
            f"Warning: {len(flagged_rel_paths)} triplet(s) failed QC after "
            f"{MAX_TRANSFORM_ATTEMPTS} attempts (saved with qc_passed=False). "
            f"List: {manifest}"
        )
    else:
        if os.path.isfile(manifest):
            os.remove(manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic triplets: edit CONFIG in file; paths optional on CLI."
    )
    p.add_argument(
        "--input-path",
        type=str,
        default="./data/IXI_2D/",
        help="IXI_2D root with Train/, Val/, Test/, Atlas/ each containing *.npy",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="./data/IXI_2D_synth_trip/",
        help="Output root for *_triplet.npz",
    )
    p.add_argument(
        "--max-phi-int",
        type=float,
        default=None,
        help="Override CONFIG MAX_PHI_INTERIOR_PX for this run (interior-band max |phi|).",
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
        max_phi_interior_px_override=args.max_phi_int,
        workers=args.workers,
        base_seed=args.seed,
    )
    print("Finished! Synthetic data is ready.")
