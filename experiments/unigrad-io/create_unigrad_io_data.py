"""
Build atlas-vs-subject UniGradICON registration data (one ``.npz`` per subject slice).

Nomenclature (matches the UniGradICON paper Fig. 2 columns)
----------------------------------------------------------
- ``source``        : subject slice (moving, gets warped) from Train / Val / Test ``.npy``.
- ``target``        : atlas slice (fixed reference, ``Atlas/atlas_slice_111.npy``).
- ``phi_pred``      : UniGradICON zero-shot displacement field (pixels, channel order [col, row]).
- ``warped_pred``   : ``source`` warped by ``phi_pred`` (should look like ``target``).
- ``phi_predio``    : same model after instance optimization (IO) on this pair.
- ``warped_predio`` : ``source`` warped by ``phi_predio``.
- ``error_map``     : per-pixel L2 norm of ``phi_predio - phi_pred`` (scalar, shape (H, W)).

Default IO protocol matches the official UniGradICON setup:
  Adam optimizer, lr = 2e-5, LNCC similarity, 50 iterations
(see ``icon_registration.itk_wrapper.finetune_execute`` and ``unigradicon-register``).

Example (DataHub) -- official protocol::

  python create_unigrad_io_data.py --ixi-root ./data/IXI_2D/ \\
      --output-path ./data/IXI_2D_unigrad_io/

Optional smoke test with fewer slices::

  python create_unigrad_io_data.py --ixi-root ./data/IXI_2D/ \\
      --output-path ./data/IXI_2D_unigrad_io_smoke/ \\
      --splits Train --max-per-split 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unigradicon import get_unigradicon, make_sim

ATLAS_SLICE_INDEX = 111


def preprocess_for_unigrad(img_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize one 2D slice, pseudo-stack depth to 5, and resize to UniGradICON's 175^3 input."""
    im_min = torch.min(img_tensor)
    im_max = torch.quantile(img_tensor.view(-1), 0.99)
    denom = torch.clamp(im_max - im_min, min=1e-5)
    img = torch.clip(img_tensor, im_min, im_max)
    img = (img - im_min) / denom
    img = img.unsqueeze(2).repeat(1, 1, 5, 1, 1)
    return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)


def apply_displacement_2d(
    moving_np: np.ndarray, phi_px: np.ndarray, device: torch.device
) -> np.ndarray:
    """Warp a 2D image with a 2D pixel displacement field via ``grid_sample``.

    Args:
        moving_np: ``(H, W)`` float32 source image.
        phi_px: ``(2, H, W)`` displacement in pixels. Channel 0 = column (x)
            displacement, channel 1 = row (y) displacement. For each output
            pixel ``(y, x)`` the warped image samples ``moving`` at
            ``(y + phi_px[1, y, x], x + phi_px[0, y, x])``.

    Returns:
        ``(H, W)`` float32 warped image.
    """
    h, w = int(moving_np.shape[0]), int(moving_np.shape[1])
    img = torch.from_numpy(moving_np).to(device, dtype=torch.float32)[None, None]
    phi = torch.from_numpy(phi_px).to(device, dtype=torch.float32)

    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    src_x = grid_x + phi[0]
    src_y = grid_y + phi[1]
    src_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    src_y = 2.0 * src_y / max(h - 1, 1) - 1.0

    grid = torch.stack([src_x, src_y], dim=-1)[None]  # (1, H, W, 2)
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped.squeeze().detach().cpu().numpy().astype(np.float32)


def phi_vectorfield_to_slice_pixels(net: torch.nn.Module, orig_h: int, orig_w: int) -> np.ndarray:
    """Convert current UniGradICON vector field to a 2D pixel displacement map with shape (2, H, W)."""
    identity = net.identity_map
    phi_disp_175 = net.phi_AB_vectorfield - identity
    phi_rescaled = F.interpolate(
        phi_disp_175,
        [5, orig_h, orig_w],
        mode="trilinear",
        align_corners=True,
    )
    phi_plane = phi_rescaled[0, 1:3, 2, :, :].cpu().numpy()
    out = np.zeros((2, orig_h, orig_w), dtype=np.float32)
    out[0] = phi_plane[1] * (orig_w - 1)
    out[1] = phi_plane[0] * (orig_h - 1)
    return out.astype(np.float32)


def run_io_then_extract_phi_px(
    net: torch.nn.Module,
    source_175: torch.Tensor,
    target_175: torch.Tensor,
    *,
    steps: int,
    lr: float,
    orig_h: int,
    orig_w: int,
    optimizer_name: str = "adam",
) -> np.ndarray:
    """Run per-pair IO for ``steps`` and return the resulting 2D pixel displacement map.

    Defaults match the upstream UniGradICON / icon_registration IO protocol:
    Adam with ``lr=2e-5`` (``DEFAULT_FINETUNE_LEARNING_RATE`` in
    ``icon_registration.itk_wrapper``). Memory hygiene helpers for tight GPUs:
      - back up weights on CPU (saves ~ model size of VRAM vs deepcopy on GPU)
      - free optimizer / graph eagerly and call ``torch.cuda.empty_cache()`` after IO
      - ``--io-optimizer sgd`` is provided as a *documented deviation*, not the
        official setting.
    """
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
        phi_px = phi_vectorfield_to_slice_pixels(net, orig_h, orig_w)
    net.load_state_dict(state0_cpu)
    net.eval()
    del state0_cpu
    torch.cuda.empty_cache()
    return phi_px


def pick_default_atlas_index(atlas_dir: Path) -> tuple[int, Path]:
    """Return the fixed atlas slice index and file path (always atlas slice 111)."""
    path = atlas_dir / f"atlas_slice_{ATLAS_SLICE_INDEX}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing atlas file: {path}")
    return ATLAS_SLICE_INDEX, path


def load_atlas_slice(atlas_dir: Path) -> tuple[np.ndarray, int, Path]:
    """Load atlas slice 111 as float32 and return (image, index, path)."""
    i, p = pick_default_atlas_index(atlas_dir)
    return np.asarray(np.load(p), dtype=np.float32), i, p


def run_atlas_io_generation(
    ixi_root: Path,
    output_root: Path,
    *,
    splits: list[str],
    max_per_split: int | None,
    shard_id: int,
    num_shards: int,
    io_iterations: int,
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
    overwrite: bool = False,
) -> None:
    """Generate atlas-vs-subject UniGradICON registration data (zero-shot + IO) across splits."""
    device = torch.device("cuda")
    atlas_dir = ixi_root / "Atlas"
    target_img, atlas_i, atlas_path = load_atlas_slice(atlas_dir)
    th, tw = int(target_img.shape[0]), int(target_img.shape[1])
    print(f"Atlas slice index {atlas_i} from {atlas_path} shape=({th}, {tw})")

    print(f"Loading UniGradICON (IO similarity={io_sim}) on {device}...")
    net = get_unigradicon(loss_fn=make_sim(io_sim)).to(device)
    net.eval()

    I_target = torch.from_numpy(target_img).float().unsqueeze(0).unsqueeze(0)
    target_175 = preprocess_for_unigrad(I_target).to(device)

    for split in splits:
        in_dir = ixi_root / split
        out_dir = output_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.is_dir():
            print(f"Skip {split}: missing {in_dir}")
            continue
        files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
        if max_per_split is not None:
            files = files[:max_per_split]

        if num_shards > 1:
            files = [f for i, f in enumerate(files) if i % num_shards == shard_id]

        # Resumability: skip subjects whose NPZ already exists. A run that gets
        # killed mid-split (DataHub session timeout, OOM, power blip) can be
        # restarted with the same command and will pick up where it left off.
        # Use overwrite=True (or delete the NPZ) if you need to regenerate.
        if not overwrite:
            already_done = {f for f in os.listdir(out_dir) if f.endswith(".npz")}
            files_todo = [f for f in files if f"{Path(f).stem}.npz" not in already_done]
            skipped = len(files) - len(files_todo)
            files = files_todo
        else:
            skipped = 0

        cap_note = f" (cap {max_per_split})" if max_per_split else ""
        shard_note = f", shard {shard_id}/{num_shards}" if num_shards > 1 else ""
        skip_note = f", skipping {skipped} already-done" if skipped else ""
        print(f"{split}: {len(files)} slice(s) to process{cap_note}{shard_note}{skip_note}")
        if not files:
            continue

        for fname in tqdm(files, desc=split):
            subj_path = in_dir / fname
            source_img = np.load(subj_path).astype(np.float32)
            sh, sw = int(source_img.shape[0]), int(source_img.shape[1])
            if (sh, sw) != (th, tw):
                raise ValueError(
                    f"Shape mismatch {subj_path} ({sh},{sw}) vs atlas ({th},{tw})"
                )

            I_source = torch.from_numpy(source_img).float().unsqueeze(0).unsqueeze(0)
            source_175 = preprocess_for_unigrad(I_source).to(device)

            with torch.no_grad():
                net(source_175, target_175)
                phi_pred_px = phi_vectorfield_to_slice_pixels(net, sh, sw)

            phi_predio_px = run_io_then_extract_phi_px(
                net,
                source_175,
                target_175,
                steps=io_iterations,
                lr=io_lr,
                orig_h=sh,
                orig_w=sw,
                optimizer_name=io_optimizer,
            )
            del source_175
            torch.cuda.empty_cache()

            with torch.no_grad():
                warped_pred = apply_displacement_2d(source_img, phi_pred_px, device)
                warped_predio = apply_displacement_2d(source_img, phi_predio_px, device)

            error_map = np.sqrt(
                np.sum((phi_predio_px - phi_pred_px) ** 2, axis=0)
            ).astype(np.float32)

            stem = Path(fname).stem
            out_name = f"{stem}.npz"
            np.savez_compressed(
                out_dir / out_name,
                source=source_img,
                target=target_img,
                phi_pred=phi_pred_px,
                warped_pred=warped_pred,
                phi_predio=phi_predio_px,
                warped_predio=warped_predio,
                error_map=error_map,
            )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for atlas-vs-subject UniGradICON IO data generation."""
    ex = """
Examples:
  python create_unigrad_io_data.py --ixi-root ./data/IXI_2D/ --output-path ./data/IXI_2D_unigrad_io/
  python create_unigrad_io_data.py --ixi-root ./data/IXI_2D/ --max-per-split 2

Four disjoint GPUs on shared NFS/PVC (same output-path; script skips finished .npz)::

  python create_unigrad_io_data.py --ixi-root ./data/IXI_2D/ --output-path ./out/ \\
      --num-shards 4 --shard-id 0   # plus shards 1,2,3 on other pods
""".strip()
    p = argparse.ArgumentParser(
        description=(
            "Atlas-subject UniGradICON IO data: per slice stores "
            "source, target, phi_pred, warped_pred, phi_predio, warped_predio, error_map."
        ),
        epilog=ex,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ixi-root",
        type=Path,
        default=Path("./data/IXI_2D/"),
        help="Folder with Train/, Val/, Test/, Atlas/ (Atlas contains atlas_slice_*.npy).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("./data/IXI_2D_unigrad_io/"),
        help="Output root; mirrors Train/Val/Test subfolders. Files are named '<slice_stem>.npz'.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="Train,Val,Test",
        help="Comma-separated splits to process (default: Train,Val,Test).",
    )
    p.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N files per split (sorted by name).",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
        metavar="K",
        help="Parallel pods: keep slice i iff i modulo num-shards equals this id (0-based).",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        metavar="N",
        help="Parallel pods: split sorted slice list into N disjoint shards (same output-path on "
        "shared storage is OK; default 1 = single worker).",
    )
    p.add_argument(
        "--io-iterations",
        type=int,
        default=50,
        help="Instance optimization Adam steps per pair. 0 = no weight updates before phi_predio extract.",
    )
    p.add_argument(
        "--io-lr",
        type=float,
        default=2e-5,
        help="LR for IO. Default 2e-5 matches the upstream icon_registration "
        "itk_wrapper.DEFAULT_FINETUNE_LEARNING_RATE used by the official "
        "unigradicon-register CLI. Only adjust if you intentionally deviate.",
    )
    p.add_argument(
        "--io-sim",
        type=str,
        default="lncc",
        choices=["lncc", "lncc2", "mind"],
        help="Loss / similarity inside UniGradICON forward (matches unigradicon-register --io_sim).",
    )
    p.add_argument(
        "--io-optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer for IO. Default 'adam' matches upstream icon_registration / "
        "unigradicon-register. 'sgd' is provided as a memory-friendly fallback for "
        "small GPUs and is a documented deviation from the official protocol.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite NPZs even if they already exist. By default the "
        "script is resumable: subjects whose .npz is already present are skipped, "
        "so a killed run can be restarted with the same command.",
    )
    return p.parse_args()


def main() -> None:
    """Entrypoint for command-line execution."""
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"--shard-id must satisfy 0 <= shard-id < num-shards (got {args.shard_id}, {args.num_shards})")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    run_atlas_io_generation(
        args.ixi_root.resolve(),
        args.output_path.resolve(),
        splits=splits,
        max_per_split=args.max_per_split,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        io_iterations=args.io_iterations,
        io_lr=args.io_lr,
        io_sim=args.io_sim,
        io_optimizer=args.io_optimizer,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    main()
