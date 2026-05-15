#!/usr/bin/env python3
"""
Visualize atlas-based UniGradICON IO data (one ``.npz`` per subject slice).

Schema (created by create_unigrad_io_data.py):
  source, target, phi_pred (2,H,W), warped_pred,
  phi_predio (2,H,W), warped_predio, error_map (H,W)

Default panels per sample (paper-style):
  1) source (subject, moving)
  2) target (atlas, fixed)
  3) warped_pred    (source warped by phi_pred)    [+ deformation grid overlay]
  4) warped_predio  (source warped by phi_predio)  [+ deformation grid overlay]
  5) warped diff    (warped_predio - warped_pred, signed intensity)
  6) error_map      (per-pixel L2 norm of phi_predio - phi_pred)

The grid overlay is on by default (paper Fig. 2 style). Disable with ``--no-grid``.

Optional ``--phi`` swaps columns 3-4 for ``||phi_pred||`` / ``||phi_predio||``.

The display-only ``--io-iterations`` flag annotates the figure suptitle so the
IO step count used to generate the data is visible in saved images.

Examples:
  python visualize_unigrad_io_data.py --split Train --io-iterations 50 \\
      --save-path ./assets/images/unigrad-io/io_train_minmedmax.png --no-show
  python visualize_unigrad_io_data.py --split Val --selection random --num-samples 4 --phi --io-iterations 50
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_KEYS = frozenset(
    {
        "source",
        "target",
        "phi_pred",
        "warped_pred",
        "phi_predio",
        "warped_predio",
        "error_map",
    }
)
SPLITS = ("Train", "Val", "Test")
DATA_GLOB = "*.npz"


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    """Return per-pixel L2 magnitude of a 2-channel displacement field."""
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def overlay_deformation_grid(
    ax,
    phi_px: np.ndarray,
    *,
    stride: int = 12,
    color: str = "cyan",
    linewidth: float = 0.5,
    alpha: float = 0.7,
) -> None:
    """Overlay a deformed grid (paper Fig. 2 style) on the current axis.

    The grid is the level sets of the position map ``identity + phi_px`` taken
    at integer multiples of ``stride`` pixels. Channel 0 of ``phi_px`` is the
    column (x) displacement, channel 1 is the row (y) displacement.
    """
    _, h, w = phi_px.shape
    rows = np.arange(h)
    cols = np.arange(w)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    pos_col = grid_col + phi_px[0]
    pos_row = grid_row + phi_px[1]
    levels_col = np.arange(0, w + stride, stride)
    levels_row = np.arange(0, h + stride, stride)
    ax.contour(pos_col, levels=levels_col, colors=color, linewidths=linewidth, alpha=alpha)
    ax.contour(pos_row, levels=levels_row, colors=color, linewidths=linewidth, alpha=alpha)


def collect_files(data_dir: Path, split: str, pattern: str) -> list[Path]:
    """Collect generated ``.npz`` files for one split."""
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return sorted(split_dir.glob(pattern))


def load_record(path: Path) -> dict[str, np.ndarray]:
    """Load one ``.npz`` record and validate required keys."""
    with np.load(path) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{path.name} missing required keys: {sorted(missing)}")
        return {k: np.asarray(data[k]) for k in REQUIRED_KEYS}


def rank_scalar(blob: dict[str, np.ndarray], rank_by: str) -> float:
    """Compute scalar score for min/median/max selection."""
    pp = blob["phi_pred"].astype(np.float64)
    pio = blob["phi_predio"].astype(np.float64)
    err = blob["error_map"].astype(np.float64)
    if rank_by == "mean_error":
        return float(np.mean(err))
    if rank_by == "max_error":
        return float(np.max(err))
    if rank_by == "mean_phi_pred":
        return float(np.mean(phi_magnitude(pp)))
    if rank_by == "mean_phi_predio":
        return float(np.mean(phi_magnitude(pio)))
    raise ValueError(f"Unknown rank_by: {rank_by}")


def select_min_median_max(
    files: list[Path], rank_by: str
) -> list[tuple[Path, str, float]]:
    """Pick min/median/max files by ranking scalar."""
    if not files:
        return []
    scored: list[tuple[Path, float]] = [
        (fp, rank_scalar(load_record(fp), rank_by)) for fp in files
    ]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 1:
        return [(scored[0][0], "min", scored[0][1])]
    if n == 2:
        return [(scored[0][0], "min", scored[0][1]), (scored[1][0], "max", scored[1][1])]
    return [
        (scored[0][0], "min", scored[0][1]),
        (scored[n // 2][0], "median", scored[n // 2][1]),
        (scored[-1][0], "max", scored[-1][1]),
    ]


def visualize(
    data_dir: Path,
    split: str,
    *,
    selection: str,
    rank_by: str,
    num_samples: int,
    seed: int,
    show_phi: bool,
    show_grid: bool,
    grid_stride: int,
    io_iterations: int | None,
    err_vmax: float | None,
    err_percentile: float,
    phi_vmax: float | None,
    phi_percentile: float,
    save_path: Path | None,
    no_show: bool,
) -> None:
    """Render a 5-column figure per selected record."""
    files = collect_files(data_dir, split, DATA_GLOB)
    if not files:
        raise FileNotFoundError(f"No .npz under {data_dir / split}")

    if selection == "random":
        rng = random.Random(seed)
        chosen = rng.sample(files, min(num_samples, len(files)))
        picked = [(p, "", float("nan")) for p in chosen]
        title_suffix = f"random {len(picked)} of {len(files)} (seed={seed})"
    else:
        picked = select_min_median_max(files, rank_by)
        title_suffix = f"min/median/max by {rank_by} ({len(picked)} of {len(files)})"

    err_stack: list[np.ndarray] = []
    phi_pred_stack: list[np.ndarray] = []
    phi_predio_stack: list[np.ndarray] = []
    intensity_diff_stack: list[np.ndarray] = []
    for fp, _, _ in picked:
        d = load_record(fp)
        err_stack.append(d["error_map"].astype(np.float64).ravel())
        intensity_diff_stack.append(
            (d["warped_predio"].astype(np.float64) - d["warped_pred"].astype(np.float64)).ravel()
        )
        if show_phi:
            phi_pred_stack.append(phi_magnitude(d["phi_pred"]).ravel())
            phi_predio_stack.append(phi_magnitude(d["phi_predio"]).ravel())

    err_v = (
        float(err_vmax)
        if err_vmax is not None
        else max(float(np.percentile(np.concatenate(err_stack), err_percentile)), 1e-6)
    )
    diff_concat = np.concatenate(intensity_diff_stack)
    diff_v = max(
        float(np.percentile(np.abs(diff_concat), err_percentile)),
        1e-6,
    )
    if show_phi:
        phi_v = (
            float(phi_vmax)
            if phi_vmax is not None
            else max(
                float(np.percentile(np.concatenate(phi_pred_stack), phi_percentile)),
                float(np.percentile(np.concatenate(phi_predio_stack), phi_percentile)),
                1e-6,
            )
        )

    nrows = len(picked)
    fig, axes = plt.subplots(nrows, 6, figsize=(21, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for row, (fp, rank_label, score) in enumerate(picked):
        d = load_record(fp)
        source = d["source"]
        target = d["target"]
        warped_pred = d["warped_pred"]
        warped_predio = d["warped_predio"]
        err_map = d["error_map"].astype(np.float64)

        rank_note = (
            f" [{rank_label} {rank_by}={score:.4f}]"
            if (rank_label and np.isfinite(score))
            else ""
        )
        stem = fp.stem
        stem_title = stem if len(stem) <= 44 else f"{stem[:41]}..."

        axes[row, 0].imshow(source, cmap="gray")
        axes[row, 0].set_title(f"source — {stem_title}{rank_note}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(target, cmap="gray")
        axes[row, 1].set_title("target (atlas)", fontsize=9)
        axes[row, 1].axis("off")

        if show_phi:
            pred_plot = phi_magnitude(d["phi_pred"].astype(np.float64))
            predio_plot = phi_magnitude(d["phi_predio"].astype(np.float64))
            im2 = axes[row, 2].imshow(pred_plot, cmap="hot", vmin=0.0, vmax=phi_v)
            axes[row, 2].set_title("||phi_pred|| (px)", fontsize=9)
            im3 = axes[row, 3].imshow(predio_plot, cmap="hot", vmin=0.0, vmax=phi_v)
            axes[row, 3].set_title("||phi_predio|| (px)", fontsize=9)
            for ax, im in ((axes[row, 2], im2), (axes[row, 3], im3)):
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            axes[row, 2].imshow(warped_pred, cmap="gray")
            title_pred = "warped_pred + grid" if show_grid else "warped_pred"
            axes[row, 2].set_title(title_pred, fontsize=9)
            axes[row, 2].axis("off")
            if show_grid:
                overlay_deformation_grid(
                    axes[row, 2], d["phi_pred"].astype(np.float64), stride=grid_stride
                )
            axes[row, 3].imshow(warped_predio, cmap="gray")
            title_predio = "warped_predio + grid" if show_grid else "warped_predio"
            axes[row, 3].set_title(title_predio, fontsize=9)
            axes[row, 3].axis("off")
            if show_grid:
                overlay_deformation_grid(
                    axes[row, 3], d["phi_predio"].astype(np.float64), stride=grid_stride
                )

        intensity_diff = (
            warped_predio.astype(np.float64) - warped_pred.astype(np.float64)
        )
        im_id = axes[row, 4].imshow(
            intensity_diff, cmap="coolwarm", vmin=-diff_v, vmax=diff_v
        )
        axes[row, 4].set_title("warped_predio - warped_pred", fontsize=9)
        axes[row, 4].axis("off")
        fig.colorbar(im_id, ax=axes[row, 4], fraction=0.046, pad=0.02)

        im_e = axes[row, 5].imshow(err_map, cmap="hot", vmin=0.0, vmax=err_v)
        axes[row, 5].set_title("error_map (px)", fontsize=9)
        axes[row, 5].axis("off")
        fig.colorbar(im_e, ax=axes[row, 5], fraction=0.046, pad=0.02)

    mode_tag = "phi-magnitudes" if show_phi else "warped views"
    if not show_phi and show_grid:
        mode_tag += " + grid"
    io_note = f"  |  IO iters: {io_iterations}" if io_iterations is not None else ""
    fig.suptitle(
        f"Atlas UniGradICON IO — {split} ({title_suffix}, {mode_tag}){io_note}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Visualize atlas UniGradICON IO data: "
            "source, target, warped_pred, warped_predio, error_map."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-dir", type=Path, default=Path("./data/IXI_2D_unigrad_io"))
    p.add_argument("--split", type=str, default="Train", choices=list(SPLITS))
    p.add_argument(
        "--selection",
        type=str,
        default="min_median_max",
        choices=["min_median_max", "random"],
    )
    p.add_argument(
        "--rank-by",
        type=str,
        default="mean_error",
        choices=["mean_error", "max_error", "mean_phi_pred", "mean_phi_predio"],
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--phi",
        action="store_true",
        help="Show ||phi_pred|| and ||phi_predio|| in columns 3-4 instead of warped images.",
    )
    p.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay deformation grid contours on warped_pred and warped_predio "
        "(paper Fig. 2 style). Use --no-grid to disable.",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=12,
        help="Grid line spacing in pixels for the deformation overlay (default: 12).",
    )
    p.add_argument(
        "--io-iterations",
        type=int,
        default=None,
        help="Display-only: IO iterations used during data generation. Shown in suptitle.",
    )
    p.add_argument("--err-vmax", type=float, default=None)
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--phi-vmax", type=float, default=None)
    p.add_argument("--phi-percentile", type=float, default=99.0)
    p.add_argument("--save-path", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    visualize(
        data_dir=args.data_dir,
        split=args.split,
        selection=args.selection,
        rank_by=args.rank_by,
        num_samples=args.num_samples,
        seed=args.seed,
        show_phi=args.phi,
        show_grid=args.grid,
        grid_stride=args.grid_stride,
        io_iterations=args.io_iterations,
        err_vmax=args.err_vmax,
        err_percentile=args.err_percentile,
        phi_vmax=args.phi_vmax,
        phi_percentile=args.phi_percentile,
        save_path=args.save_path,
        no_show=args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
