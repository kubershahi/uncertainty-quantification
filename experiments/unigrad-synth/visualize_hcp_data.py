#!/usr/bin/env python3
"""
QC plot for downloaded HCP T1w volumes (random subjects by default).

Each row: axial slice of T1, aparc+aseg labels, brainmask_fs.

Example:
python experiments/unigrad-synth/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --save-path assets/images/hcp/hcp_random3.png --no-show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

T1_NAME = "T1w_acpc_dc_restore_brain.nii.gz"
SEG_NAME = "aparc+aseg.nii.gz"
MASK_NAME = "brainmask_fs.nii.gz"


def collect_subjects(data_dir: Path) -> list[Path]:
    """Subject dirs with a complete T1w triplet."""
    out: list[Path] = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        t1w = subj_dir / "T1w"
        if all((t1w / name).is_file() for name in (T1_NAME, SEG_NAME, MASK_NAME)):
            out.append(subj_dir)
    return out


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def default_slice_index(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def axial_slice(vol: np.ndarray, slice_index: int | None) -> np.ndarray:
    z = default_slice_index(vol) if slice_index is None else int(slice_index)
    z = max(0, min(z, vol.shape[2] - 1))
    return vol[:, :, z]


def select_subjects(
    subjects: list[Path],
    *,
    num_samples: int,
    seed: int,
) -> list[Path]:
    n = min(num_samples, len(subjects))
    rng = random.Random(seed)
    return rng.sample(subjects, n)


def plot_hcp_samples(
    data_dir: Path,
    save_path: Path | None,
    no_show: bool,
    *,
    num_samples: int = 3,
    seed: int = 42,
    slice_index: int | None = None,
) -> None:
    subjects = collect_subjects(data_dir)
    if not subjects:
        raise FileNotFoundError(
            f"No complete HCP subjects under {data_dir} "
            f"(expected <id>/T1w/{{{T1_NAME}, {SEG_NAME}, {MASK_NAME}}})."
        )

    picked = select_subjects(subjects, num_samples=num_samples, seed=seed)
    z_note = "mid" if slice_index is None else str(slice_index)

    fig, axes = plt.subplots(len(picked), 3, figsize=(11, 3.4 * len(picked)))
    axes = np.atleast_2d(axes)

    for row, subj_dir in enumerate(picked):
        t1w = subj_dir / "T1w"
        t1 = load_nifti(t1w / T1_NAME)
        seg = load_nifti(t1w / SEG_NAME)
        mask = load_nifti(t1w / MASK_NAME)

        t1_sl = axial_slice(t1, slice_index)
        seg_sl = axial_slice(seg, slice_index)
        mask_sl = axial_slice(mask, slice_index)
        z = default_slice_index(t1) if slice_index is None else int(slice_index)

        subj_id = subj_dir.name
        row_title = f"{subj_id}  (z={z}/{t1.shape[2] - 1})"

        axes[row, 0].imshow(np.rot90(t1_sl), cmap="gray")
        axes[row, 0].set_title("T1 (brain)", fontsize=9)
        axes[row, 0].axis("off")

        seg_masked = np.ma.masked_where(seg_sl <= 0, seg_sl)
        im_seg = axes[row, 1].imshow(
            np.rot90(seg_masked), cmap="nipy_spectral", interpolation="nearest"
        )
        axes[row, 1].set_title("aparc+aseg", fontsize=9)
        axes[row, 1].axis("off")
        fig.colorbar(im_seg, ax=axes[row, 1], fraction=0.046, pad=0.02)

        im_m = axes[row, 2].imshow(np.rot90(mask_sl), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("brainmask_fs", fontsize=9)
        axes[row, 2].axis("off")
        fig.colorbar(im_m, ax=axes[row, 2], fraction=0.046, pad=0.02)

        axes[row, 0].set_ylabel(row_title, fontsize=8, rotation=90, labelpad=4)

    fig.suptitle(
        f"HCP T1w QC | random {len(picked)} of {len(subjects)} (seed={seed}, axial {z_note})",
        fontsize=11,
    )
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
        description="Visualize random HCP T1w downloads (T1, segmentation, mask).",
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
        "--save-path",
        type=Path,
        default=Path("assets/images/hcp/hcp_random3.png"),
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
