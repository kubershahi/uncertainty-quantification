#!/usr/bin/env python3
"""
Visualization script for downloaded HCP T1w volumes.

Layout (columns = subjects):
  row 1 — T1w_acpc_dc_restore_brain (axial)
  row 2 — brainmask_fs
  row 3 — aparc+aseg (only with ``--show-segmentation``)

Example:
python experiments/synth-data-gen/torchio/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --save-path assets/images/synth-data/torchio/hcp/hcp_random3.png --no-show
python experiments/synth-data-gen/torchio/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --show-segmentation --save-path assets/images/synth-data/torchio/hcp/hcp_random3_seg.png --no-show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
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


def load_nifti_with_axcodes(path: Path) -> tuple[np.ndarray, tuple[str, str, str]]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    axcodes = tuple(str(c) for c in aff2axcodes(img.affine))
    return data, axcodes  # e.g., ('R', 'A', 'S')


def default_slice_index(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def axial_slice(vol: np.ndarray, slice_index: int | None) -> tuple[np.ndarray, int]:
    z = default_slice_index(vol) if slice_index is None else int(slice_index)
    z = max(0, min(z, vol.shape[2] - 1))
    return vol[:, :, z], z


def orient_axial(sl: np.ndarray) -> np.ndarray:
    """Radiological-style display: superior at top, patient right on image-left."""
    return np.rot90(sl)


def _opposite_axis_label(code: str) -> str:
    pairs = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
    return pairs.get(code, "?")


def _display_lr_labels_from_axcodes(axcodes: tuple[str, str, str]) -> tuple[str, str]:
    """
    Patient L/R at image edges after ``orient_axial`` (np.rot90 on vol[:, :, z]).

    ``aff2axcodes`` gives the anatomical direction of *increasing* voxel index per
    axis — for the **stored 3D volume**, before any display rotation.  For axial
    ``sl = vol[:, :, z]``, axis 0 is slice rows and axis 1 is slice columns.
    ``np.rot90`` (counter-clockwise) turns those rows into the horizontal display
    axis: low row index (opposite of ``axcodes[0]``) on image-left, high row
    index (``axcodes[0]``) on image-right.

    Example: LAS storage (axcodes[0]='L') -> image-left=R, image-right=L
    (radiological convention).
    """
    axis0_positive = axcodes[0]
    image_right = axis0_positive
    image_left = _opposite_axis_label(axis0_positive)
    return image_left, image_right


def select_subjects(
    subjects: list[Path],
    *,
    num_samples: int,
    seed: int,
) -> list[Path]:
    n = min(num_samples, len(subjects))
    rng = random.Random(seed)
    return rng.sample(subjects, n)


def _style_axes(ax: plt.Axes, *, show_ylabel: bool, ylabel: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE, rotation=90, labelpad=8)


def _add_colorbar(fig: plt.Figure, ax: plt.Axes, im, label: str) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=_TICK_SIZE)
    cbar.ax.tick_params(labelsize=_TICK_SIZE)


def zscore_with_mask(vol: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score normalize using only in-mask voxels; outside-mask set to 0."""
    m = mask > 0.5
    in_vals = vol[m]
    in_vals = in_vals[np.isfinite(in_vals)]
    if in_vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mu = float(np.mean(in_vals))
    sigma = float(np.std(in_vals))
    sigma = max(sigma, eps)
    z = (vol.astype(np.float32) - mu) / sigma
    z[~m] = 0.0
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def plot_hcp_samples(
    data_dir: Path,
    save_path: Path | None,
    no_show: bool,
    *,
    num_samples: int = 3,
    seed: int = 42,
    slice_index: int | None = None,
    show_segmentation: bool = False,
    show_orientation_note: bool = True,
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

    # Preload slices and apply masked z-score normalization for T1 display
    rows_data: list[dict] = []
    seg_vals: list[np.ndarray] = []
    for subj_dir in picked:
        t1w = subj_dir / "T1w"
        t1, axcodes = load_nifti_with_axcodes(t1w / T1_NAME)
        mask = load_nifti(t1w / MASK_NAME)
        t1_z = zscore_with_mask(t1, mask)
        t1_sl, z = axial_slice(t1_z, slice_index)
        mask_sl, _ = axial_slice(mask, slice_index)
        left_label, right_label = _display_lr_labels_from_axcodes(axcodes)
        entry: dict = {
            "id": subj_dir.name,
            "t1": orient_axial(t1_sl),
            "mask": orient_axial(mask_sl),
            "z": z,
            "nz": t1.shape[2],
            "shape": t1.shape,
            "left_label": left_label,
            "right_label": right_label,
            "axcodes": axcodes,
        }
        if show_segmentation:
            seg = load_nifti(t1w / SEG_NAME)
            seg_sl, _ = axial_slice(seg, slice_index)
            entry["seg"] = orient_axial(seg_sl)
            seg_vals.append(seg_sl[seg_sl > 0])
        rows_data.append(entry)

    t1_vmin, t1_vmax = -3.0, 3.0
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

    row_ylabels = ["Brain T1w Slice", "Brain Mask"]
    if show_segmentation:
        row_ylabels.append("Segmentation labels")

    im_t1_last = None
    im_seg_last = None

    for col, entry in enumerate(rows_data):
        axes[0, col].set_title(
            f"Subject {entry['id']}",
            fontsize=_SUBTITLE_SIZE,
            fontweight="medium",
            pad=8,
        )

        # Row 0: T1
        im_t1_last = axes[0, col].imshow(
            entry["t1"],
            cmap="gray",
            vmin=t1_vmin,
            vmax=t1_vmax,
            interpolation="nearest",
            origin="upper",
        )
        _style_axes(
            axes[0, col],
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
                show_ylabel=(col == 0),
                ylabel=row_ylabels[2],
            )

    # Colorbar on last column of segmentation row
    if show_segmentation and im_seg_last is not None:
        _add_colorbar(fig, axes[2, -1], im_seg_last, "FreeSurfer label ID")

    z_values = sorted({int(e["z"]) for e in rows_data})
    nz_values = sorted({int(e["nz"] - 1) for e in rows_data})
    if len(z_values) == 1 and len(nz_values) == 1:
        z_note = f"axial z = {z_values[0]} / {nz_values[0]}"
    else:
        z_note = "axial z shown per panel"

    fig.suptitle(
        "HCP Young Adult (S1200) Dataset Plot (T1w)",
        fontsize=_TITLE_SIZE,
        fontweight="bold",
        y=0.98,
    )
    axcode_sets = sorted({e["axcodes"] for e in rows_data})
    left_right_pairs = sorted({(e["left_label"], e["right_label"]) for e in rows_data})
    if len(axcode_sets) == 1 and len(left_right_pairs) == 1:
        storage = "/".join(axcode_sets[0])
        img_l, img_r = left_right_pairs[0]
        orient_note = (
            f"Display convention = Radiological-style "
            f"(image-left = {img_l}, image-right = {img_r}, {storage})"
        )
    else:
        orient_note = "Display convention varies across selected subjects"

    subtitle = (
        f"Random sample of {ncols} / {len(subjects)} subjects "
        f"(seed = {seed})  ·  {z_note}"
    )
    if show_orientation_note:
        subtitle = subtitle + f"\n{orient_note}"

    fig.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="top",
        fontsize=_TICK_SIZE,
        color="black",
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.90 if show_segmentation else 0.88,
        top=0.84,
        bottom=0.08,
        wspace=0.24,
        hspace=0.34,
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
        "--no-orientation-note",
        action="store_true",
        help="Hide orientation convention line from figure subtitle.",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("assets/images/synth-data/torchio/hcp/hcp_random3.png"),
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
        show_orientation_note=not args.no_orientation_note,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
