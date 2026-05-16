#!/usr/bin/env python3
"""
Plot synthetic *_triplet.npz samples (fixed, warped, phi view).

Default selection: three slices with **minimum**, **median**, and **maximum** of a scalar
phi statistic (default: mean ‖φ‖ on the slice). Use ``--selection random`` for random slices.

Examples:
  python visualize_synth_data.py --split Train --phi-view magnitude --save-path ./assets/images/ixi_synth_minmedmax.png --no-show 
  python visualize_synth_data.py --selection random --num-samples 5 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Keep in sync with data_checks/check_synth_data.py
REQUIRED_KEYS = frozenset({"image", "warped", "phi"})
SPLITS = ("Train", "Val", "Test", "Atlas")
TRIPLET_GLOB = "*_triplet.npz"


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def _unpack_qc_passed(raw) -> tuple[bool | None, str | None]:
    a = np.asarray(raw)
    if a.size != 1:
        return None, f"qc_passed must be a single value, got shape {a.shape}"
    v = a.reshape(-1)[0]
    if isinstance(v, (np.floating, float)) and not np.isfinite(float(v)):
        return None, "qc_passed is non-finite"
    try:
        return bool(v), None
    except (ValueError, TypeError) as e:
        return None, f"qc_passed not bool-convertible: {e}"


def collect_triplets(input_dir: Path, split: str, pattern: str) -> list[Path]:
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return sorted(split_dir.glob(pattern))


def load_triplet(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    extra: dict = {}
    with np.load(npz_path) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        image = np.asarray(data["image"])
        warped = np.asarray(data["warped"])
        phi = np.asarray(data["phi"])
        if "qc_passed" in data.files:
            qc_val, qc_err = _unpack_qc_passed(data["qc_passed"])
            if qc_err:
                raise ValueError(f"{npz_path.name}: {qc_err}")
            extra["qc_passed"] = qc_val
    return image, warped, phi, extra


def scalar_phi_score(phi: np.ndarray, metric: str) -> float:
    """Single number per slice for ranking (min / median / max)."""
    mag = phi_magnitude(phi.astype(np.float64))
    if metric == "mean":
        return float(np.mean(mag))
    if metric == "max":
        return float(np.max(mag))
    raise ValueError(f"metric must be 'mean' or 'max', got {metric!r}")


def select_min_median_max(
    files: list[Path],
    phi_metric: str,
) -> list[tuple[Path, str, float]]:
    """
    Return up to three (path, label, score) with labels 'min', 'median', 'max'.
    """
    if not files:
        return []
    scored: list[tuple[Path, float]] = []
    for fp in files:
        _, _, phi, _ = load_triplet(fp)
        scored.append((fp, scalar_phi_score(phi, phi_metric)))
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 1:
        return [(scored[0][0], "min", scored[0][1])]
    if n == 2:
        return [
            (scored[0][0], "min", scored[0][1]),
            (scored[1][0], "max", scored[1][1]),
        ]
    i_min, i_med, i_max = 0, n // 2, n - 1
    return [
        (scored[i_min][0], "min", scored[i_min][1]),
        (scored[i_med][0], "median", scored[i_med][1]),
        (scored[i_max][0], "max", scored[i_max][1]),
    ]


def render_phi(phi: np.ndarray, phi_view: str) -> tuple[np.ndarray, str]:
    if phi_view == "x":
        return phi[0], "phi_x"
    if phi_view == "y":
        return phi[1], "phi_y"
    magnitude = np.sqrt(phi[0] ** 2 + phi[1] ** 2)
    return magnitude, "‖φ‖"


def plot_quiver_on_axis(ax: plt.Axes, phi: np.ndarray, step: int = 8) -> None:
    dx = phi[0, ::step, ::step]
    dy = phi[1, ::step, ::step]
    x, y = np.meshgrid(
        np.arange(0, phi.shape[2], step),
        np.arange(0, phi.shape[1], step),
    )
    ax.quiver(
        x,
        y,
        dx,
        -dy,
        color="teal",
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.003,
        headwidth=3.5,
        headlength=5.0,
        headaxislength=4.5,
    )
    ax.set_xlim(0, phi.shape[2])
    ax.set_ylim(phi.shape[1], 0)
    ax.set_aspect("equal")
    ax.set_title("Phi (quiver)", fontsize=10)
    ax.axis("off")


def visualize_triplets(
    input_dir: Path,
    split: str,
    pattern: str,
    phi_view: str,
    quiver_step: int,
    mag_vmin: float,
    mag_vmax: float | None,
    mag_percentile: float,
    save_path: Path | None,
    no_show: bool,
    *,
    selection: str = "min_median_max",
    phi_metric: str = "mean",
    num_samples: int = 3,
    seed: int = 42,
) -> None:
    files = collect_triplets(input_dir, split, pattern)
    if not files:
        raise FileNotFoundError(
            f"No files in '{input_dir / split}' matching '{pattern}'."
        )

    if selection == "random":
        rng = random.Random(seed)
        sample_count = min(num_samples, len(files))
        picked: list[tuple[Path, str, float]] = [
            (p, "", float("nan")) for p in rng.sample(files, sample_count)
        ]
        title_suffix = f"random {sample_count} of {len(files)}"
    elif selection == "min_median_max":
        picked = select_min_median_max(files, phi_metric)
        title_suffix = (
            f"min / median / max of {phi_metric} ‖φ‖ ({len(picked)} of {len(files)} files)"
        )
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    sample_count = len(picked)
    resolved_mag_vmax = mag_vmax
    if phi_view == "magnitude" and resolved_mag_vmax is None:
        mags = []
        for item in picked:
            fp = item[0]
            _, _, phi, _ = load_triplet(fp)
            mags.append(np.sqrt(phi[0] ** 2 + phi[1] ** 2).ravel())
        stacked = np.concatenate(mags)
        resolved_mag_vmax = float(np.percentile(stacked, mag_percentile))
        if resolved_mag_vmax <= mag_vmin:
            resolved_mag_vmax = mag_vmin + 1.0

    fig, axes = plt.subplots(sample_count, 3, figsize=(12, 3.4 * sample_count))
    axes = np.atleast_2d(axes)

    for row, (file_path, rank_label, score) in enumerate(picked):
        image, warped, phi, extra = load_triplet(file_path)
        phi_img, phi_title = render_phi(phi, phi_view=phi_view)
        qc_note = ""
        if "qc_passed" in extra:
            qc_note = f" qc_passed={extra['qc_passed']}"
        rank_note = ""
        if rank_label:
            rank_note = f" [{rank_label}"
            if np.isfinite(score):
                rank_note += f" {phi_metric}={score:.3f}px"
            rank_note += "]"

        ax_img = axes[row, 0]
        ax_img.imshow(image, cmap="gray")
        ax_img.set_title(f"Fixed: {file_path.stem}{qc_note}{rank_note}", fontsize=9)
        ax_img.axis("off")

        ax_warped = axes[row, 1]
        ax_warped.imshow(warped, cmap="gray")
        ax_warped.set_title("Warped", fontsize=10)
        ax_warped.axis("off")

        ax_phi = axes[row, 2]
        if phi_view == "quiver":
            plot_quiver_on_axis(ax_phi, phi, step=quiver_step)
        elif phi_view == "magnitude":
            phi_plot = ax_phi.imshow(
                phi_img,
                cmap="hot",
                vmin=mag_vmin,
                vmax=resolved_mag_vmax,
            )
            ax_phi.set_title("‖φ‖ (px)", fontsize=10)
            ax_phi.axis("off")
            cbar = fig.colorbar(phi_plot, ax=ax_phi, fraction=0.046, pad=0.04)
            cbar.set_label("‖φ‖ (pixels)")
        else:
            phi_plot = ax_phi.imshow(phi_img, cmap="coolwarm")
            ax_phi.set_title(phi_title, fontsize=10)
            ax_phi.axis("off")
            cbar = fig.colorbar(phi_plot, ax=ax_phi, fraction=0.046, pad=0.04)
            cbar.set_label(f"{phi_title} (pixels)")

        axes[row, 0].set_ylabel(file_path.stem[:32], fontsize=7, rotation=90)

    fig.suptitle(f"Synthetic triplets — {split} ({title_suffix})", fontsize=12)
    fig.tight_layout()

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
        description="Visualize synthetic *_triplet.npz (min/median/max ‖φ‖ or random).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        "--input-dir",
        type=Path,
        default=Path("./data/IXI_2D_synth_trip"),
        dest="data_dir",
        help="Root with Train/Val/Test/Atlas subfolders.",
    )
    p.add_argument("--split", type=str, default="Train", choices=list(SPLITS))
    p.add_argument(
        "--selection",
        type=str,
        default="min_median_max",
        choices=["min_median_max", "random"],
        help="How to pick slices: min/median/max of phi metric, or random.",
    )
    p.add_argument(
        "--phi-metric",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Scalar per slice for min/median/max: mean ‖φ‖ or max ‖φ‖ on slice.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="With --selection random: number of slices to draw.",
    )
    p.add_argument("--seed", type=int, default=42, help="Used when --selection random.")
    p.add_argument(
        "--phi-view",
        type=str,
        required=True,
        choices=["quiver", "magnitude", "x", "y"],
        help="Phi visualization mode.",
    )
    p.add_argument("--quiver-step", type=int, default=8)
    p.add_argument("--mag-vmin", type=float, default=0.0)
    p.add_argument("--mag-vmax", type=float, default=None)
    p.add_argument("--mag-percentile", type=float, default=99.0)
    p.add_argument("--save-path", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2
    visualize_triplets(
        input_dir=data_dir,
        split=args.split,
        pattern=TRIPLET_GLOB,
        phi_view=args.phi_view,
        quiver_step=args.quiver_step,
        mag_vmin=args.mag_vmin,
        mag_vmax=args.mag_vmax,
        mag_percentile=args.mag_percentile,
        save_path=args.save_path,
        no_show=args.no_show,
        selection=args.selection,
        phi_metric=args.phi_metric,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
