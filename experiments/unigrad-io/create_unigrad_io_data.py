"""
Build atlas-vs-subject UniGradICON 3D registration data (one ``.npz`` per IXI volume).

Reads ``Train|Val|Test/*.pkl`` subject volumes + ``atlas.pkl``, runs zero-shot registration
and **50-step** instance optimization (IO: Adam, lr ``2e-5``, LNCC), writes compressed NPZ.

Dataset layout
--------------
::

  <output-path>/
    atlas_valid_mask.npz     # atlas (H,W,D) + valid_mask (D,H,W) — shared, outside splits
    Train/<subject>.npz
    Val/<subject>.npz
    Test/<subject>.npz

Per-subject NPZ keys
--------------------
- ``source``: ``(H, W, D)`` float32
- ``phi_pred``, ``phi_predio``: ``(3, D, H, W)`` float32 — voxel shifts
- ``error_map``: ``(D, H, W)`` float32 — ``||phi_predio - phi_pred||_2``
- ``io_iterations``: int32

Examples (from repo root)::

python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root datasets/IXI --atlas-pkl datasets/IXI/atlas.pkl --output-path datasets/IXI_unigrad_io/

python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root datasets/IXI --atlas-pkl datasets/IXI/atlas.pkl --output-path datasets/IXI_unigrad_io/ --splits Val --max-per-split 2
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unigradicon import get_unigradicon, make_sim

ATLAS_MASK_FILENAME = "atlas_valid_mask.npz"
DEFAULT_FG_PERCENTILE = 5.0


def _atlas_foreground_mask_hw_d(
    atlas_hw_d: np.ndarray, *, fg_percentile: float
) -> tuple[np.ndarray, float]:
    """Mask from atlas>0 percentile (zeros excluded from threshold)."""
    nonzero = atlas_hw_d[atlas_hw_d > 0]
    if nonzero.size == 0:
        raise ValueError("atlas has no positive voxels")
    t = float(np.percentile(nonzero.astype(np.float64), fg_percentile))
    return (atlas_hw_d > t).astype(np.bool_), t


def _save_atlas_and_mask_npz(
    out_path: Path, atlas_hw_d: np.ndarray, *, fg_percentile: float
) -> tuple[float, float]:
    mask_hw, threshold = _atlas_foreground_mask_hw_d(atlas_hw_d, fg_percentile=fg_percentile)
    mask_dhw = np.transpose(mask_hw, (2, 0, 1))
    frac = float(mask_dhw.mean())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        atlas=atlas_hw_d.astype(np.float32, copy=False),
        valid_mask=mask_dhw,
        threshold=np.float32(threshold),
        fg_percentile=np.float32(fg_percentile),
    )
    return threshold, frac


def pkload(path: Path):
    with path.open("rb") as f, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
        )
        return pickle.load(f)


def load_ixi_image_volume_from_pkl(pkl_path: Path) -> np.ndarray:
    payload = pkload(pkl_path)
    if isinstance(payload, tuple) and len(payload) >= 1:
        raw = payload[0]
    elif isinstance(payload, np.ndarray):
        raw = payload
    else:
        raise ValueError(f"Unexpected pickle structure in {pkl_path}: {type(payload)}")
    img = np.array(raw, dtype=np.float32, copy=True)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D volume in {pkl_path}, got shape {img.shape}")
    return img


def numpy_volume_hw_d_to_torch5d(vol_hw_d: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(vol_hw_d.astype(np.float32))
    return t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


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


def apply_displacement_3d(
    moving_hw_d: np.ndarray, phi_dhw: np.ndarray, device: torch.device
) -> np.ndarray:
    h, w, d = int(moving_hw_d.shape[0]), int(moving_hw_d.shape[1]), int(moving_hw_d.shape[2])
    od, oh, ow = phi_dhw.shape[1], phi_dhw.shape[2], phi_dhw.shape[3]
    if (od, oh, ow) != (d, h, w):
        raise ValueError(f"phi spatial shape {(od, oh, ow)} vs volume {(d, h, w)} (D,H,W)")

    vol = (
        torch.from_numpy(moving_hw_d)
        .to(device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    phi_t = torch.from_numpy(phi_dhw).to(device, dtype=torch.float32)

    zs = torch.arange(d, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")

    src_x = grid_x + phi_t[2]
    src_y = grid_y + phi_t[1]
    src_z = grid_z + phi_t[0]
    src_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    src_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    src_z = 2.0 * src_z / max(d - 1, 1) - 1.0
    grid = torch.stack([src_x, src_y, src_z], dim=-1).unsqueeze(0)

    warped = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def run_io_then_extract_phi(
    net: torch.nn.Module,
    source_175: torch.Tensor,
    target_175: torch.Tensor,
    *,
    steps: int,
    lr: float,
    orig_d: int,
    orig_h: int,
    orig_w: int,
    optimizer_name: str = "adam",
) -> np.ndarray:
    state0_cpu = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    if steps > 0:
        if optimizer_name == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name!r}")
        was_training = net.training
        net.train()
        try:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss_tuple = net(source_175, target_175)
                loss_tuple[0].backward()
                opt.step()
                del loss_tuple
        finally:
            if not was_training:
                net.eval()
            del opt
            torch.cuda.empty_cache()
    with torch.no_grad():
        net(source_175, target_175)
        phi = phi_vectorfield_to_volume_voxels(net, orig_d, orig_h, orig_w)
    net.load_state_dict(state0_cpu)
    net.eval()
    del state0_cpu
    torch.cuda.empty_cache()
    return phi


def _plan_split_volumes(
    ixi_root: Path,
    output_root: Path,
    *,
    splits: list[str],
    max_per_split: int | None,
    shard_id: int,
    num_shards: int,
    overwrite: bool,
) -> list[tuple[str, Path, Path, list[str]]]:
    """Return ``(split, in_dir, out_dir, pkl_filenames)`` for each split to process."""
    plan: list[tuple[str, Path, Path, list[str]]] = []
    for split in splits:
        in_dir = ixi_root / split
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.is_dir():
            print(f"Skip {split}: missing {in_dir}")
            plan.append((split, in_dir, out_dir, []))
            continue
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".pkl"))
        if max_per_split is not None:
            files = files[:max_per_split]
        if num_shards > 1:
            files = [f for i, f in enumerate(files) if i % num_shards == shard_id]
        if not overwrite:
            already_done = {f for f in os.listdir(out_dir) if f.endswith(".npz")}
            files = [f for f in files if f"{Path(f).stem}.npz" not in already_done]
        plan.append((split, in_dir, out_dir, files))
    return plan


def run_atlas_io_generation(
    ixi_root: Path,
    output_root: Path,
    *,
    atlas_pkl: Path,
    splits: list[str],
    max_per_split: int | None,
    shard_id: int,
    num_shards: int,
    io_iterations: int,
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
    fg_percentile: float = DEFAULT_FG_PERCENTILE,
    overwrite: bool = False,
) -> None:
    device = torch.device("cuda")
    if not atlas_pkl.is_file():
        raise FileNotFoundError(f"--atlas-pkl not found: {atlas_pkl}")

    output_root.mkdir(parents=True, exist_ok=True)
    atlas_vol = load_ixi_image_volume_from_pkl(atlas_pkl)
    oh, ow, od = int(atlas_vol.shape[0]), int(atlas_vol.shape[1]), int(atlas_vol.shape[2])
    print(f"Atlas {atlas_pkl} shape (H,W,D)=({oh},{ow},{od})")

    mask_path = output_root / ATLAS_MASK_FILENAME
    if overwrite or not mask_path.is_file():
        thr, fg_frac = _save_atlas_and_mask_npz(
            mask_path, atlas_vol, fg_percentile=fg_percentile
        )
        print(
            f"Wrote {mask_path.name}: atlas {atlas_vol.shape}  threshold={thr:.6g}  "
            f"foreground={fg_frac * 100:.2f}%  (p{fg_percentile:g} of atlas>0)"
        )
    else:
        print(f"Reusing {mask_path.name} (--overwrite to rebuild atlas + mask)")

    print(f"Loading UniGradICON (IO similarity={io_sim}) on {device}...")
    net = get_unigradicon(loss_fn=make_sim(io_sim)).to(device)
    net.eval()

    atlas_5d = numpy_volume_hw_d_to_torch5d(atlas_vol).to(device)
    target_175 = preprocess_volume_for_unigrad(atlas_5d)
    del atlas_5d

    plan = _plan_split_volumes(
        ixi_root,
        output_root,
        splits=splits,
        max_per_split=max_per_split,
        shard_id=shard_id,
        num_shards=num_shards,
        overwrite=overwrite,
    )
    total_volumes = sum(len(files) for _, _, _, files in plan)
    print(f"Total: {total_volumes} volume(s) across {len(splits)} split(s) (IO steps={io_iterations})")
    for split, _, _, files in plan:
        print(f"  {split}: {len(files)} volume(s)")

    with tqdm(total=total_volumes, desc="total", position=0) as total_pbar:
        for split, in_dir, out_dir, files in plan:
            if not files:
                continue
            for fname in tqdm(files, desc=split, position=1, leave=False):
                subj_path = in_dir / fname
                source_vol = load_ixi_image_volume_from_pkl(subj_path)
                sh, sw, sd = int(source_vol.shape[0]), int(source_vol.shape[1]), int(source_vol.shape[2])
                if source_vol.shape != atlas_vol.shape:
                    raise ValueError(
                        f"Shape mismatch {subj_path} {source_vol.shape} vs atlas {atlas_vol.shape}"
                    )

                vol_5d = numpy_volume_hw_d_to_torch5d(source_vol).to(device)
                source_175 = preprocess_volume_for_unigrad(vol_5d)
                del vol_5d

                with torch.no_grad():
                    net(source_175, target_175)
                    phi_pred = phi_vectorfield_to_volume_voxels(net, sd, sh, sw)

                phi_predio = run_io_then_extract_phi(
                    net,
                    source_175,
                    target_175,
                    steps=io_iterations,
                    lr=io_lr,
                    orig_d=sd,
                    orig_h=sh,
                    orig_w=sw,
                    optimizer_name=io_optimizer,
                )
                del source_175
                torch.cuda.empty_cache()

                error_map = np.sqrt(
                    np.sum((phi_predio - phi_pred) ** 2, axis=0)
                ).astype(np.float32)

                np.savez_compressed(
                    out_dir / f"{Path(fname).stem}.npz",
                    source=source_vol,
                    phi_pred=phi_pred,
                    phi_predio=phi_predio,
                    error_map=error_map,
                    io_iterations=np.int32(io_iterations),
                )
                total_pbar.update(1)

    print(f"Done. {output_root.resolve()}")
    print(f"  Shared: {ATLAS_MASK_FILENAME} (atlas + valid_mask)")
    print("  Per subject: source, phi_pred, phi_predio, error_map, io_iterations")


def parse_args() -> argparse.Namespace:
    ex = """
Examples:
python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root datasets/IXI --atlas-pkl datasets/IXI/atlas.pkl --output-path datasets/IXI_unigrad_io/
python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root datasets/IXI --atlas-pkl datasets/IXI/atlas.pkl --output-path datasets/IXI_unigrad_io/ --splits Train --max-per-split 3
""".strip()
    p = argparse.ArgumentParser(
        description="IXI 3D volume UniGradICON IO data: one .npz per subject .pkl.",
        epilog=ex,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ixi-root",
        type=Path,
        default=Path("./datasets/IXI"),
        help="Folder with Train/Val/Test/*.pkl volumes.",
    )
    p.add_argument(
        "--atlas-pkl",
        type=Path,
        default=None,
        help="Atlas volume pickle (default: <ixi-root>/atlas.pkl).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("./datasets/IXI_unigrad_io"),
        help="Output root; mirrors Train/Val/Test with <subject_stem>.npz per volume.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="Train,Val,Test",
        help="Comma-separated splits.",
    )
    p.add_argument("--max-per-split", type=int, default=None, metavar="N")
    p.add_argument("--shard-id", type=int, default=0, metavar="K")
    p.add_argument("--num-shards", type=int, default=1, metavar="N")
    p.add_argument(
        "--io-iterations",
        type=int,
        default=50,
        help="IO Adam steps per pair (default 50).",
    )
    p.add_argument("--io-lr", type=float, default=2e-5)
    p.add_argument("--io-sim", type=str, default="lncc", choices=["lncc", "lncc2", "mind"])
    p.add_argument("--io-optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument(
        "--fg-percentile",
        type=float,
        default=DEFAULT_FG_PERCENTILE,
        help="Atlas mask: percentile of atlas>0 voxels (default 5).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(
            f"--shard-id must satisfy 0 <= shard-id < num-shards (got {args.shard_id}, {args.num_shards})"
        )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    atlas_pkl = (
        args.atlas_pkl.resolve()
        if args.atlas_pkl is not None
        else (args.ixi_root / "atlas.pkl").resolve()
    )
    run_atlas_io_generation(
        args.ixi_root.resolve(),
        args.output_path.resolve(),
        atlas_pkl=atlas_pkl,
        splits=splits,
        max_per_split=args.max_per_split,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        io_iterations=args.io_iterations,
        io_lr=args.io_lr,
        io_sim=args.io_sim,
        io_optimizer=args.io_optimizer,
        fg_percentile=args.fg_percentile,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
