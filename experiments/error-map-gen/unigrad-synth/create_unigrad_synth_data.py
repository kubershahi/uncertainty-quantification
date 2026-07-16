"""
Generate error-map NPZ from HCP TorchIO synthetic registration pairs.

Reads ``Train|Val|Test/*.npz`` from ``create_synth_data.py``; runs UniGradICON
(moving → source) so ``u_pred`` lives on the same **source/fixed** grid as Phase I ``u_gt``;
writes augmented NPZ under ``--output-path/{Train,Val,Test}/``.

Input NPZ keys (required from Phase I)
--------------------------------------
  - source             : fixed image (float32, ``(X, Y, Z)``); masked z-score
  - moving             : warped image (float32, ``(X, Y, Z)``); same lattice as source
  - u_gt               : GT registration displacement on **source** grid
                         (float32, ``(3, X, Y, Z)``): ``moving(x + u_gt(x)) ≈ source(x)``
  - source_mask        : fixed brain mask (bool)
  - moving_mask        : warped brain mask (bool)
  - identity_grid_mask : in-bounds mask for ``u_gt`` (bool)
  - source_affine      : NIfTI voxel → world (float32, ``(4, 4)``)
  - deformation_class  : ``none`` | ``rigid`` | ``affine`` | ``elastic`` | ``affine_elastic``
  - subject_id         : HCP subject ID (str scalar)

Output NPZ keys
---------------
  - source, moving, source_mask, moving_mask, source_affine, deformation_class, subject_id
                         — copied from input unchanged
  - u_gt                 — copied from input (source-lattice GT)
  - u_gt_igm             — input ``identity_grid_mask``
  - u_pred               — UniGradICON predicted displacement on the **source** grid
                         (``(3, X, Y, Z)`` voxels): ``moving(x + u_pred(x)) ≈ source(x)``;
                         zeroed where ``u_gt_igm`` is false, 12-voxel face border zeroed,
                         then ‖u‖ clipped at p99.9 (nonzero voxels)
  - u_error_map          — ``‖u_gt - u_pred‖`` per voxel (float32, ``(X, Y, Z)``)
  - error_map_mask       — ``u_gt_igm & interior(12-voxel margin)`` (bool); use for U-Net loss

Registration: ``net(moving, source)`` so both ``u_gt`` and ``u_pred`` share
``moving(x + u(x)) ≈ source(x)`` on the source lattice.
``u_pred`` cleanup matches Phase I: OOB → border → p99.9 clip.

Throughput: default ``np.savez`` (use ``--compress`` for zlib), prefetch next NPZ on a
background thread while the current sample runs, no per-sample ``empty_cache``.

Examples:
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp --output-path datasets/error-map/unigrad-synth/hcp --device cuda
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp --output-path datasets/error-map/unigrad-synth/hcp --splits Train --max-per-split 15 --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unigradicon import get_unigradicon

HCP_SYNTH_REQUIRED_KEYS = frozenset(
    {
        "source",
        "moving",
        "u_gt",
        "source_mask",
        "moving_mask",
        "identity_grid_mask",
        "source_affine",
        "deformation_class",
        "subject_id",
    }
)
HCP_SYNTH_PASS_KEYS = (
    "source",
    "moving",
    "source_mask",
    "moving_mask",
    "source_affine",
    "deformation_class",
    "subject_id",
)
FULL_SPLITS = ("Train", "Val", "Test")
U_BOUNDARY_MARGIN = 12  # match create_synth_data.py
U_CLIP_PERCENTILE = 99.9  # match create_synth_data.py


def resolve_device(mode: str) -> torch.device:
    """Pick torch.device. 'auto' probes CUDA; falls back to CPU if kernels fail."""
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.randn(1, 1, 8, 8, 8, device="cuda")
        y = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        torch.cuda.synchronize()
        del x, y
        return torch.device("cuda")
    except Exception as exc:
        print(
            f"CUDA reported available but a probe failed ({exc!r}); "
            "using CPU. Pass --device cpu to skip this message."
        )
        return torch.device("cpu")


def preprocess_volume_for_unigrad(vol_5d: torch.Tensor) -> torch.Tensor:
    """Clip to p99 and resize to 175³ for UniGradICON.

    ``torch.quantile`` fails on large CUDA tensors (HCP ~21M voxels exceeds the
    ~16M-element limit), so p99 is computed with NumPy on the host.
    """
    im_min = torch.min(vol_5d)
    im_max_val = float(np.quantile(vol_5d.detach().float().cpu().numpy(), 0.99))
    im_max = torch.as_tensor(im_max_val, device=vol_5d.device, dtype=vol_5d.dtype)
    denom = torch.clamp(im_max - im_min, min=1e-5)
    img = torch.clip(vol_5d, im_min, im_max)
    img = (img - im_min) / denom
    return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)


def phi_vectorfield_to_volume_voxels(
    net: torch.nn.Module, orig_d: int, orig_h: int, orig_w: int
) -> np.ndarray:
    identity = net.identity_map
    phi_disp_175 = net.phi_AB_vectorfield - identity
    phi_rescaled = F.interpolate(
        phi_disp_175,
        [orig_d, orig_h, orig_w],
        mode="trilinear",
        align_corners=True,
    )
    p = phi_rescaled[0].cpu().numpy()
    out = np.zeros((3, orig_d, orig_h, orig_w), dtype=np.float32)
    out[0] = p[0] * (orig_d - 1)
    out[1] = p[1] * (orig_h - 1)
    out[2] = p[2] * (orig_w - 1)
    return out


def hcp_volume_xyz_to_torch5d(vol_xyz: np.ndarray) -> torch.Tensor:
    """HCP synth volumes are ``(X, Y, Z)``; UniGradICON expects ``(1, 1, D, H, W)``."""
    t = torch.from_numpy(np.asarray(vol_xyz, dtype=np.float32))
    return t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


def phi_dhw_to_u_xyz(phi_dhw: np.ndarray) -> np.ndarray:
    """Map UniGradICON phi ``(3, D, H, W)`` with ``D=Z, H=X, W=Y`` to ``u`` ``(3, X, Y, Z)``."""
    if phi_dhw.ndim != 4 or phi_dhw.shape[0] != 3:
        raise ValueError(f"expected phi (3, D, H, W), got {phi_dhw.shape}")
    return np.stack(
        [
            np.transpose(phi_dhw[1], (1, 2, 0)),
            np.transpose(phi_dhw[2], (1, 2, 0)),
            np.transpose(phi_dhw[0], (1, 2, 0)),
        ],
        axis=0,
    ).astype(np.float32)


def u_error_map_from_gt_pred(u_gt: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
    diff = u_gt.astype(np.float64) - u_pred.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=0)).astype(np.float32)


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def clip_u_at_percentile(u: np.ndarray, percentile: float = U_CLIP_PERCENTILE) -> np.ndarray:
    """Scale vectors with ‖u‖ above percentile down to that threshold (nonzero voxels only)."""
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


def _interior_valid_mask(shape_xyz: tuple[int, int, int], margin: int) -> np.ndarray:
    x, y, z = shape_xyz
    mask = np.zeros((x, y, z), dtype=bool)
    if x > 2 * margin and y > 2 * margin and z > 2 * margin:
        mask[margin : x - margin, margin : y - margin, margin : z - margin] = True
    else:
        mask[:] = True
    return mask


def cleanup_u_pred(
    u_pred: np.ndarray,
    u_gt_igm: np.ndarray,
    *,
    boundary_margin: int = U_BOUNDARY_MARGIN,
    clip_percentile: float = U_CLIP_PERCENTILE,
) -> np.ndarray:
    """Match Phase I ``u`` cleanup: OOB mask → face border → p99.9 ‖u‖ clip."""
    out = u_pred.astype(np.float32, copy=True)
    igm = np.asarray(u_gt_igm, dtype=bool)
    if igm.shape != out.shape[1:]:
        raise ValueError(f"u_gt_igm {igm.shape} vs u_pred spatial {out.shape[1:]}")
    out[:, ~igm] = 0.0
    shape_xyz = (int(out.shape[1]), int(out.shape[2]), int(out.shape[3]))
    keep = _interior_valid_mask(shape_xyz, boundary_margin)
    out[:, ~keep] = 0.0
    return clip_u_at_percentile(out, clip_percentile)


def error_map_mask_from_igm(
    u_gt_igm: np.ndarray,
    shape_xyz: tuple[int, int, int],
    *,
    boundary_margin: int = U_BOUNDARY_MARGIN,
) -> np.ndarray:
    """``u_gt_igm`` AND interior ``boundary_margin`` (exclude border from U-Net loss)."""
    igm = np.asarray(u_gt_igm, dtype=bool)
    if igm.shape != shape_xyz:
        raise ValueError(f"u_gt_igm {igm.shape} vs volume {shape_xyz}")
    interior = _interior_valid_mask(shape_xyz, boundary_margin)
    return (igm & interior).astype(bool)


def load_hcp_synth_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = HCP_SYNTH_REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{path.name} missing {sorted(missing)}")
        return {k: np.asarray(data[k]) for k in data.files}


def run_hcp_error_map_generation(
    input_root: Path,
    output_root: Path,
    *,
    splits: list[str],
    max_per_split: int | None = None,
    device: torch.device | None = None,
    overwrite: bool = False,
    compress: bool = False,
) -> None:
    if device is None:
        device = resolve_device("auto")
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Flat job list so we can prefetch the next NPZ while the current one runs on GPU.
    jobs: list[tuple[str, str, Path, Path]] = []
    for split in splits:
        in_dir = input_root / split
        if not in_dir.is_dir():
            print(f"Skip {split}: missing {in_dir}")
            continue
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npz"))
        if max_per_split is not None:
            files = files[:max_per_split]
        if not overwrite:
            done = {f for f in os.listdir(out_dir) if f.endswith(".npz")}
            files = [f for f in files if f not in done]
        print(f"{split}: {len(files)} sample(s) to process")
        for fname in files:
            jobs.append((split, fname, in_dir / fname, out_dir / fname))

    total = len(jobs)
    if total == 0:
        print("Nothing to process.")
        return

    save_fn = np.savez_compressed if compress else np.savez
    print(
        f"Loading UniGradICON on {device} ({total} volume(s)); "
        f"save={'compressed' if compress else 'uncompressed'}; prefetch=on"
    )
    net = get_unigradicon().to(device)
    net.eval()

    n_done = 0
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        tqdm(total=total, desc="UniGradICON HCP synth", dynamic_ncols=True) as pbar,
    ):
        next_fut: Future | None = pool.submit(load_hcp_synth_npz, jobs[0][2])
        for i, (split, fname, _in_path, out_path) in enumerate(jobs):
            pbar.set_postfix_str(f"{n_done + 1}/{total} {split}/{fname}", refresh=True)
            assert next_fut is not None
            sample = next_fut.result()
            next_fut = (
                pool.submit(load_hcp_synth_npz, jobs[i + 1][2])
                if i + 1 < total
                else None
            )

            source = sample["source"]
            moving = sample["moving"]
            u_gt = np.asarray(sample["u_gt"], dtype=np.float32)
            u_gt_igm = np.asarray(sample["identity_grid_mask"], dtype=bool)

            nx, ny, nz = (int(source.shape[0]), int(source.shape[1]), int(source.shape[2]))
            if moving.shape != source.shape:
                raise ValueError(f"{fname}: moving {moving.shape} vs source {source.shape}")
            if u_gt.shape != (3, nx, ny, nz):
                raise ValueError(f"{fname}: u_gt {u_gt.shape} vs expected (3, {nx}, {ny}, {nz})")

            moving_5d = hcp_volume_xyz_to_torch5d(moving).to(device)
            source_5d = hcp_volume_xyz_to_torch5d(source).to(device)
            moving_175 = preprocess_volume_for_unigrad(moving_5d)
            source_175 = preprocess_volume_for_unigrad(source_5d)
            del moving_5d, source_5d

            with torch.no_grad():
                # moving → source: phi / u_pred on the source (fixed) grid (matches Phase I u_gt).
                net(moving_175, source_175)
                phi_dhw = phi_vectorfield_to_volume_voxels(net, nz, nx, ny)
            del moving_175, source_175

            u_pred = phi_dhw_to_u_xyz(phi_dhw)
            u_pred = cleanup_u_pred(u_pred, u_gt_igm)
            u_err = u_error_map_from_gt_pred(u_gt, u_pred)
            err_mask = error_map_mask_from_igm(u_gt_igm, (nx, ny, nz))

            out_payload: dict[str, np.ndarray] = {}
            for key in HCP_SYNTH_PASS_KEYS:
                out_payload[key] = sample[key]
            out_payload["u_gt"] = u_gt
            out_payload["u_gt_igm"] = u_gt_igm
            out_payload["u_pred"] = u_pred
            out_payload["u_error_map"] = u_err
            out_payload["error_map_mask"] = err_mask

            save_fn(out_path, **out_payload)
            n_done += 1
            pbar.update(1)

    print(f"Done: wrote {n_done} NPZ(s) under {output_root.resolve()}")
    print(
        "  Per sample: HCP synth keys + u_gt, u_gt_igm, u_pred, u_error_map, error_map_mask"
    )


def parse_args() -> argparse.Namespace:
    examples = """
Examples:
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp --output-path datasets/error-map/unigrad-synth/hcp
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp_100 --output-path datasets/error-map/unigrad-synth/hcp_100 --max-per-split 2
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --device cpu --overwrite
""".strip()

    p = argparse.ArgumentParser(
        description=(
            "UniGradICON moving→source on HCP synth pairs: add u_pred (source grid) and u_error_map."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-path",
        type=Path,
        default=Path("datasets/synth-data/torchio/hcp"),
        help="HCP synth root with Train/Val/Test/*.npz.",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/error-map/unigrad-synth/hcp"),
        help="Output root (mirrors split subfolders).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="Train,Val,Test",
        help="Comma-separated splits to process.",
    )
    p.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N files per split (sorted by name).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if output NPZ already exists.",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed (slower writes, smaller files). Default: uncompressed np.savez.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    device = resolve_device(args.device)
    run_hcp_error_map_generation(
        args.input_path,
        args.output_path,
        splits=splits,
        max_per_split=args.max_per_split,
        device=device,
        overwrite=args.overwrite,
        compress=args.compress,
    )
