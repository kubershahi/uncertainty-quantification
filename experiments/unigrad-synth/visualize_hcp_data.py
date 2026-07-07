#!/usr/bin/env python3
"""
Visualization script for downloaded HCP T1w volumes.

Layout (columns = subjects):
  row 1 — T1w_acpc_dc_restore_brain (axial)
  row 2 — brainmask_fs
  row 3 — aparc+aseg (only with ``--show-segmentation``)

Example:
python experiments/unigrad-synth/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --save-path assets/images/unigrad-synth/hcp/hcp_random3.png --no-show
python experiments/unigrad-synth/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --show-segmentation --save-path assets/images/unigrad-synth/hcp/hcp_random3_seg.png --no-show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

T1_NAME = "T1w_acpc_dc_restore_brain.nii.gz"
SEG_NAME = "aparc+aseg.nii.gz"
MASK_NAME = "brainmask_fs.nii.gz"

# Research-figure defaults
_DPI = 200
_FONT_FAMILY = "DejaVu Sans"
_TITLE_SIZE = 12
_SUBTITLE_SIZE = 10
_LABEL_SIZE = 9
_TICK_SIZE = 8


def collect_subjects(data_dir: Path, *, require_seg: bool) -> list[Path]:
    """Subject dirs with T1 + mask; optionally require aparc+aseg."""
    required = (T1_NAME, MASK_NAME)
    if require_seg:
        required = (T1_NAME, SEG_NAME, MASK_NAME)
    out: list[Path] = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        t1w = subj_dir / "T1w"
        if all((t1w / name).is_file() for name in required):
            out.append(subj_dir)
    return out


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def default_slice_index(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def axial_slice(vol: np.ndarray, slice_index: int | None) -> tuple[np.ndarray, int]:
    z = default_slice_index(vol) if slice_index is None else int(slice_index)
    z = max(0, min(z, vol.shape[2] - 1))
    return vol[:, :, z], z


def orient_axial(sl: np.ndarray) -> np.ndarray:
    """Radiological-style display: superior at top, right on left of figure."""
    return np.rot90(sl)


def select_subjects(
    subjects: list[Path],
    *,
    num_samples: int,
    seed: int,
) -> list[Path]:
    n = min(num_samples, len(subjects))
    rng = random.Random(seed)
    return rng.sample(subjects, n)


def _style_axes(ax: plt.Axes, *, show_xlabel: bool, show_ylabel: bool, ylabel: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")
    if show_xlabel:
        ax.set_xlabel("R ←  axial (L–R)  → L", fontsize=_TICK_SIZE, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE, rotation=90, labelpad=8)


def _add_colorbar(fig: plt.Figure, ax: plt.Axes, im, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=_TICK_SIZE)
    cbar.ax.tick_params(labelsize=_TICK_SIZE)


def plot_hcp_samples(
    data_dir: Path,
    save_path: Path | None,
    no_show: bool,
    *,
    num_samples: int = 3,
    seed: int = 42,
    slice_index: int | None = None,
    show_segmentation: bool = False,
) -> None:
    subjects = collect_subjects(data_dir, require_seg=show_segmentation)
    if not subjects:
        need = f"{{{T1_NAME}, {MASK_NAME}"
        if show_segmentation:
            need += f", {SEG_NAME}"
        need += "}"
        raise FileNotFoundError(
            f"No complete HCP subjects under {data_dir} (expected <id>/T1w/{need})."
        )

    picked = select_subjects(subjects, num_samples=num_samples, seed=seed)
    ncols = len(picked)
    nrows = 3 if show_segmentation else 2

    # Preload slices for shared T1 intensity scale and consistent z labels
    rows_data: list[dict] = []
    t1_vals: list[np.ndarray] = []
    seg_vals: list[np.ndarray] = []
    for subj_dir in picked:
        t1w = subj_dir / "T1w"
        t1 = load_nifti(t1w / T1_NAME)
        mask = load_nifti(t1w / MASK_NAME)
        t1_sl, z = axial_slice(t1, slice_index)
        mask_sl, _ = axial_slice(mask, slice_index)
        entry: dict = {
            "id": subj_dir.name,
            "t1": orient_axial(t1_sl),
            "mask": orient_axial(mask_sl),
            "z": z,
            "nz": t1.shape[2],
            "shape": t1.shape,
        }
        t1_vals.append(t1_sl[np.isfinite(t1_sl) & (t1_sl > 0)])
        if show_segmentation:
            seg = load_nifti(t1w / SEG_NAME)
            seg_sl, _ = axial_slice(seg, slice_index)
            entry["seg"] = orient_axial(seg_sl)
            seg_vals.append(seg_sl[seg_sl > 0])
        rows_data.append(entry)

    t1_vmax = float(np.percentile(np.concatenate(t1_vals), 99.0)) if t1_vals else 1.0
    if t1_vmax <= 0:
        t1_vmax = 1.0
    seg_vmax = float(np.max(np.concatenate(seg_vals))) if seg_vals else 1.0

    plt.rcParams.update(
        {
            "font.family": _FONT_FAMILY,
            "axes.titlesize": _SUBTITLE_SIZE,
            "axes.labelsize": _LABEL_SIZE,
            "figure.dpi": _DPI,
            "savefig.dpi": _DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )

    fig_h = 2.8 * nrows + 1.4
    fig_w = 3.2 * ncols + 1.2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=False,
    )

    row_ylabels = [
        "T1w (brain-extracted)\nintensity",
        "Brain mask\n(brainmask_fs)",
    ]
    if show_segmentation:
        row_ylabels.append("Segmentation\n(aparc+aseg)")

    im_t1_last = None
    im_seg_last = None

    for col, entry in enumerate(rows_data):
        # Column header: subject id + slice
        axes[0, col].set_title(
            f"Subject {entry['id']}\naxial $z$ = {entry['z']} / {entry['nz'] - 1}",
            fontsize=_SUBTITLE_SIZE,
            fontweight="medium",
            pad=8,
        )

        # Row 0: T1
        im_t1_last = axes[0, col].imshow(
            entry["t1"],
            cmap="gray",
            vmin=0.0,
            vmax=t1_vmax,
            interpolation="nearest",
            origin="upper",
        )
        _style_axes(
            axes[0, col],
            show_xlabel=False,
            show_ylabel=(col == 0),
            ylabel=row_ylabels[0],
        )

        # Row 1: mask (binary display)
        mask_bin = (entry["mask"] > 0.5).astype(np.float32)
        axes[1, col].imshow(
            mask_bin,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="upper",
        )
        _style_axes(
            axes[1, col],
            show_xlabel=(not show_segmentation),
            show_ylabel=(col == 0),
            ylabel=row_ylabels[1],
        )

        # Row 2: segmentation (optional)
        if show_segmentation:
            seg_masked = np.ma.masked_where(entry["seg"] <= 0, entry["seg"])
            im_seg_last = axes[2, col].imshow(
                seg_masked,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=seg_vmax,
                interpolation="nearest",
                origin="upper",
            )
            _style_axes(
                axes[2, col],
                show_xlabel=True,
                show_ylabel=(col == 0),
                ylabel=row_ylabels[2],
            )

    # Colorbar on last column of T1 row
    if im_t1_last is not None:
        _add_colorbar(fig, axes[0, -1], im_t1_last, "T1 intensity (a.u.)")

    # Colorbar on last column of segmentation row
    if show_segmentation and im_seg_last is not None:
        _add_colorbar(fig, axes[2, -1], im_seg_last, "FreeSurfer label ID")

    # Mask legend (binary)
    mask_handles = [
        Patch(facecolor="black", edgecolor="0.35", label="Background (0)"),
        Patch(facecolor="white", edgecolor="0.35", label="Brain (1)"),
    ]
    axes[1, -1].legend(
        handles=mask_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=_TICK_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="0.5",
        title="Mask",
        title_fontsize=_TICK_SIZE,
    )

    z_note = "mid-volume" if slice_index is None else f"z = {slice_index}"
    fig.suptitle(
        "HCP Young Adult (S1200) — structural T1w QC",
        fontsize=_TITLE_SIZE,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        (
            f"Random sample of {ncols} / {len(subjects)} subjects "
            f"(seed = {seed})  ·  axial slice ({z_note})\n"
            f"Radiological display (top = superior; left of panel = subject right).  "
            f"Files: {T1_NAME}, {MASK_NAME}"
            + (f", {SEG_NAME}" if show_segmentation else "")
        ),
        ha="center",
        va="top",
        fontsize=_TICK_SIZE,
        color="0.25",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.88 if show_segmentation else 0.86,
        top=0.86,
        bottom=0.08,
        wspace=0.20,
        hspace=0.30,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize random HCP T1w downloads: T1 (top), mask (middle), "
            "optional aparc+aseg (bottom)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/hcp"),
        help="Root with per-subject T1w/ folders.",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Axial slice index (default: mid slice along z).",
    )
    p.add_argument(
        "--show-segmentation",
        action="store_true",
        help="Add a third row with aparc+aseg FreeSurfer labels.",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("assets/images/unigrad-synth/hcp/hcp_random3.png"),
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    plot_hcp_samples(
        args.data_dir,
        args.save_path,
        args.no_show,
        num_samples=args.num_samples,
        seed=args.seed,
        slice_index=args.slice_index,
        show_segmentation=args.show_segmentation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
