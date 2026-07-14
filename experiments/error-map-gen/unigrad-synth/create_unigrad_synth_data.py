"""
Generate error-map NPZ from HCP TorchIO synthetic registration pairs.

Reads ``Train|Val|Test/*.npz`` from ``create_synth_data.py``; runs UniGradICON
(moving → fixed); writes augmented NPZ under ``--output-path/{Train,Val,Test}/``.

Input NPZ keys (required from Phase I)
--------------------------------------
  - source             : fixed image (float32, ``(X, Y, Z)``); masked z-score
  - moving             : warped image (float32, ``(X, Y, Z)``); same grid as source
  - u                  : ground-truth displacement (float32, ``(3, X, Y, Z)`` voxels)
  - source_mask        : fixed brain mask (bool)
  - moving_mask        : warped brain mask (bool)
  - identity_grid_mask : in-bounds backward-warp mask (bool); OOB voxels invalid for u
  - source_affine      : NIfTI voxel → world (float32, ``(4, 4)``)
  - deformation_class  : ``none`` | ``rigid`` | ``affine`` | ``elastic`` | ``affine_elastic``
  - subject_id         : HCP subject ID (str scalar)

Output NPZ keys
---------------
  - source, moving, source_mask, moving_mask, source_affine, deformation_class, subject_id
                         — copied from input unchanged
  - u_gt                 — input ``u`` (GT displacement on fixed grid, voxels)
  - u_gt_igm             — input ``identity_grid_mask``
  - u_pred               — UniGradICON predicted displacement (``(3, X, Y, Z)`` voxels);
                         zeroed where ``u_gt_igm`` is false and within 12-voxel face border
  - u_error_map          — ``‖u_gt - u_pred‖`` per voxel (float32, ``(X, Y, Z)``)
  - error_map_mask       — ``u_gt_igm & interior(12-voxel margin)`` (bool); use for U-Net loss

Registration: ``net(moving, source)``; displacement lives on the **fixed** (source) grid.
``u_pred`` cleanup matches Phase I border/OOB handling (no p99.9 clip on predictions).

Examples:
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp --output-path datasets/error-map/unigrad-synth/hcp --device cuda
python experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py --input-path datasets/synth-data/torchio/hcp --output-path datasets/error-map/unigrad-synth/hcp --splits Train --max-per-split 15 --device cuda
"""

from __future__ import annotations

import argparse
import os
import time
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
        "u",
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
    im_min = torch.min(vol_5d)
    im_max = torch.quantile(vol_5d.reshape(-1), 0.99)
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
) -> np.ndarray:
    """Match Phase I ``u`` cleanup (without p99.9 clip): OOB mask then face border."""
    out = u_pred.astype(np.float32, copy=True)
    igm = np.asarray(u_gt_igm, dtype=bool)
    if igm.shape != out.shape[1:]:
        raise ValueError(f"u_gt_igm {igm.shape} vs u_pred spatial {out.shape[1:]}")
    out[:, ~igm] = 0.0
    shape_xyz = (int(out.shape[1]), int(out.shape[2]), int(out.shape[3]))
    keep = _interior_valid_mask(shape_xyz, boundary_margin)
    out[:, ~keep] = 0.0
    return out


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


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_hcp_error_map_generation(
    input_root: Path,
    output_root: Path,
    *,
    splits: list[str],
    max_per_split: int | None = None,
    device: torch.device | None = None,
    overwrite: bool = False,
) -> None:
    if device is None:
        device = resolve_device("auto")
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Count files first so we can explain cost before the model download stalls the terminal.
    split_files: dict[str, list[str]] = {}
    total = 0
    for split in splits:
        in_dir = input_root / split
        if not in_dir.is_dir():
            print(f"Skip {split}: missing {in_dir}", flush=True)
            continue
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npz"))
        if max_per_split is not None:
            files = files[:max_per_split]
        if not overwrite:
            done = {f for f in os.listdir(out_dir) if f.endswith(".npz")}
            files = [f for f in files if f not in done]
        split_files[split] = files
        print(f"{split}: {len(files)} sample(s) to process", flush=True)
        total += len(files)

    if total == 0:
        print("Nothing to process.", flush=True)
        return

    print(
        f"\nAbout to run UniGradICON on {total} volume pair(s) on {device}.\n"
        "  Cost per sample (rough): load NPZ → resize to 175³ → ICON forward → "
        "upsample phi → save compressed NPZ.\n"
        "  Expect ~seconds/sample on GPU, often minutes/sample on CPU "
        "(full 3D registration network).\n",
        flush=True,
    )
    if device.type == "cpu":
        print(
            "WARNING: device=cpu — each forward pass is very slow. "
            "Prefer --device cuda if a GPU is available.\n",
            flush=True,
        )

    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] Loading UniGradICON weights on {device}...", flush=True)
    net = get_unigradicon().to(device)
    net.eval()
    print(
        f"[{time.strftime('%H:%M:%S')}] Model ready ({time.perf_counter() - t0:.1f}s). "
        "Starting samples...\n",
        flush=True,
    )

    n_done = 0
    sample_times: list[float] = []
    with tqdm(total=total, desc="UniGradICON HCP synth", dynamic_ncols=True) as pbar:
        for split in splits:
            files = split_files.get(split, [])
            if not files:
                continue
            in_dir = input_root / split
            out_dir = output_root / split
            for fname in files:
                sample_i = n_done + 1
                pbar.set_postfix_str(f"{split}/{fname}", refresh=True)
                print(
                    f"\n[{sample_i}/{total}] {split}/{fname}",
                    flush=True,
                )
                t_sample = time.perf_counter()

                print("  [1/5] load NPZ...", flush=True)
                t_step = time.perf_counter()
                in_path = in_dir / fname
                sample = load_hcp_synth_npz(in_path)
                source = sample["source"]
                moving = sample["moving"]
                u_gt = np.asarray(sample["u"], dtype=np.float32)
                u_gt_igm = np.asarray(sample["identity_grid_mask"], dtype=bool)

                nx, ny, nz = (int(source.shape[0]), int(source.shape[1]), int(source.shape[2]))
                if moving.shape != source.shape:
                    raise ValueError(f"{fname}: moving {moving.shape} vs source {source.shape}")
                if u_gt.shape != (3, nx, ny, nz):
                    raise ValueError(f"{fname}: u {u_gt.shape} vs expected (3, {nx}, {ny}, {nz})")
                print(
                    f"        shape ({nx},{ny},{nz}) in {time.perf_counter() - t_step:.1f}s",
                    flush=True,
                )

                print("  [2/5] preprocess → 175³ ...", flush=True)
                t_step = time.perf_counter()
                moving_5d = hcp_volume_xyz_to_torch5d(moving).to(device)
                source_5d = hcp_volume_xyz_to_torch5d(source).to(device)
                moving_175 = preprocess_volume_for_unigrad(moving_5d)
                source_175 = preprocess_volume_for_unigrad(source_5d)
                del moving_5d, source_5d
                _sync_device(device)
                print(f"        done in {time.perf_counter() - t_step:.1f}s", flush=True)

                print(
                    "  [3/5] UniGradICON forward (net(moving, source)) — usually the slow step...",
                    flush=True,
                )
                t_step = time.perf_counter()
                with torch.no_grad():
                    net(moving_175, source_175)
                    _sync_device(device)
                print(
                    f"        forward done in {time.perf_counter() - t_step:.1f}s",
                    flush=True,
                )

                print("  [4/5] upsample phi → native u_pred, build error map...", flush=True)
                t_step = time.perf_counter()
                with torch.no_grad():
                    phi_dhw = phi_vectorfield_to_volume_voxels(net, nz, nx, ny)
                del moving_175, source_175
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                u_pred = phi_dhw_to_u_xyz(phi_dhw)
                u_pred = cleanup_u_pred(u_pred, u_gt_igm)
                u_err = u_error_map_from_gt_pred(u_gt, u_pred)
                err_mask = error_map_mask_from_igm(u_gt_igm, (nx, ny, nz))
                print(f"        done in {time.perf_counter() - t_step:.1f}s", flush=True)

                print("  [5/5] save compressed NPZ...", flush=True)
                t_step = time.perf_counter()
                out_payload: dict[str, np.ndarray] = {}
                for key in HCP_SYNTH_PASS_KEYS:
                    out_payload[key] = sample[key]
                out_payload["u_gt"] = u_gt
                out_payload["u_gt_igm"] = u_gt_igm
                out_payload["u_pred"] = u_pred
                out_payload["u_error_map"] = u_err
                out_payload["error_map_mask"] = err_mask

                np.savez_compressed(out_dir / fname, **out_payload)
                elapsed = time.perf_counter() - t_sample
                sample_times.append(elapsed)
                mean_s = sum(sample_times) / len(sample_times)
                remaining = total - sample_i
                eta_s = mean_s * remaining
                print(
                    f"        saved in {time.perf_counter() - t_step:.1f}s | "
                    f"sample total {elapsed:.1f}s | "
                    f"avg {mean_s:.1f}s | ETA ~{eta_s / 60.0:.1f} min",
                    flush=True,
                )
                n_done += 1
                pbar.update(1)

    print(f"\nDone: wrote {n_done} NPZ(s) under {output_root.resolve()}", flush=True)
    if sample_times:
        print(
            f"  Timing: min {min(sample_times):.1f}s / "
            f"mean {sum(sample_times) / len(sample_times):.1f}s / "
            f"max {max(sample_times):.1f}s per sample",
            flush=True,
        )
    print(
        "  Per sample: HCP synth keys + u_gt, u_gt_igm, u_pred, u_error_map, error_map_mask",
        flush=True,
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
            "UniGradICON on HCP synth pairs: copy synth NPZ fields, add u_pred and u_error_map."
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
    )
