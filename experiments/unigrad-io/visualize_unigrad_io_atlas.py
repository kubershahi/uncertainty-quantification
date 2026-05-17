#!/usr/bin/env python3
"""
QC figure for the shared atlas and foreground mask from ``create_unigrad_io_data.py``.

Reads ``<data-dir>/atlas_valid_mask.npz`` (``atlas`` ``(H,W,D)``, ``valid_mask`` ``(D,H,W)``).
Shows axial slices: atlas intensity and mask overlay at low / mid / high z.

Example:
python experiments/unigrad-io/visualize_unigrad_io_atlas.py --data-dir datasets/IXI_unigrad_io --save-path assets/images/unigrad-io/unigradio_atlas_mask.png --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATLAS_MASK_FILENAME = "atlas_valid_mask.npz"


def default_slice_indices(depth: int, n: int = 3) -> list[int]:
    if depth <= 0:
        return [0]
    if n == 1:
        return [depth // 2]
    zs = [0, depth // 2, depth - 1]
    return sorted(set(int(np.clip(z, 0, depth - 1)) for z in zs))


def load_atlas_bundle(data_dir: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    path = Path(data_dir) / ATLAS_MASK_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path} (run create_unigrad_io_data.py with --output-path {data_dir})"
        )
    with np.load(path) as z:
        atlas_key = "atlas" if "atlas" in z.files else "target"
        if atlas_key not in z.files:
            raise KeyError(f"{path} has no 'atlas' or 'target' array")
        atlas = np.asarray(z[atlas_key])
        if "valid_mask" not in z.files:
            raise KeyError(f"{path} missing 'valid_mask'")
        mask_dhw = np.asarray(z["valid_mask"])
        threshold = float(z["threshold"]) if "threshold" in z.files else float("nan")
        fg_pct = float(z["fg_percentile"]) if "fg_percentile" in z.files else float("nan")
    if atlas.ndim != 3:
        raise ValueError(f"atlas must be (H, W, D), got {atlas.shape}")
    if mask_dhw.ndim != 3:
        raise ValueError(f"valid_mask must be (D, H, W), got {mask_dhw.shape}")
    if mask_dhw.shape != (atlas.shape[2], atlas.shape[0], atlas.shape[1]):
        raise ValueError(
            f"mask shape {mask_dhw.shape} vs atlas (D,H,W)=({atlas.shape[2]}, {atlas.shape[0]}, {atlas.shape[1]})"
        )
    return atlas, mask_dhw.astype(bool), threshold, fg_pct


def axial_slice_atlas(atlas_hw_d: np.ndarray, z: int) -> np.ndarray:
    z = int(np.clip(z, 0, atlas_hw_d.shape[2] - 1))
    return atlas_hw_d[:, :, z]


def axial_slice_mask(mask_dhw: np.ndarray, z: int) -> np.ndarray:
    z = int(np.clip(z, 0, mask_dhw.shape[0] - 1))
    return mask_dhw[z]


def plot_atlas_mask_figure(
    atlas_hw_d: np.ndarray,
    mask_dhw: np.ndarray,
    *,
    threshold: float,
    fg_percentile: float,
    slice_indices: list[int] | None,
    save_path: Path | None,
    no_show: bool,
) -> None:
    depth = int(atlas_hw_d.shape[2])
    z_list = slice_indices if slice_indices is not None else default_slice_indices(depth)
    n = len(z_list)

    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.8))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    fg_frac = float(mask_dhw.mean())
    for col, z in enumerate(z_list):
        atlas_s = axial_slice_atlas(atlas_hw_d, z)
        mask_s = axial_slice_mask(mask_dhw, z)

        ax0 = axes[0, col]
        ax0.imshow(atlas_s, cmap="gray", aspect="equal")
        ax0.set_title(f"atlas  z = {z}", fontsize=9)
        ax0.axis("off")

        ax1 = axes[1, col]
        ax1.imshow(atlas_s, cmap="gray", aspect="equal")
        ax1.imshow(mask_s, cmap="Greens", alpha=0.45, vmin=0, vmax=1, aspect="equal")
        ax1.set_title("foreground mask", fontsize=9)
        ax1.axis("off")

    thr_s = f"{threshold:.4g}" if np.isfinite(threshold) else "n/a"
    pct_s = f"{fg_percentile:g}" if np.isfinite(fg_percentile) else "n/a"
    fig.suptitle(
        f"UniGrad IO atlas · threshold = {thr_s} (p{pct_s} of atlas>0) · "
        f"foreground = {fg_frac * 100:.1f}% of voxels",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize shared atlas + valid_mask from atlas_valid_mask.npz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/IXI_unigrad_io"),
        help="Dataset root containing atlas_valid_mask.npz.",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("assets/images/unigrad-io/atlas_mask_qc.png"),
    )
    p.add_argument(
        "--slice-index",
        type=int,
        default=None,
        metavar="Z",
        help="Single axial z; default shows z=0, mid, z=max.",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    atlas, mask, threshold, fg_pct = load_atlas_bundle(args.data_dir)
    print(
        f"Loaded {ATLAS_MASK_FILENAME}: atlas {atlas.shape} (H,W,D)  "
        f"mask {mask.shape} (D,H,W)  threshold={threshold}  fg_percentile={fg_pct}"
    )

    z_list = [args.slice_index] if args.slice_index is not None else None
    plot_atlas_mask_figure(
        atlas,
        mask,
        threshold=threshold,
        fg_percentile=fg_pct,
        slice_indices=z_list,
        save_path=args.save_path,
        no_show=args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
