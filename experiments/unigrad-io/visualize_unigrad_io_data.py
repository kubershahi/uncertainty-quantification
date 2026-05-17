#!/usr/bin/env python3
"""
Visualize 3D UniGrad ICON IO data from ``create_unigrad_io_data.py``.

Per subject (``Train|Val|Test/*.npz``): ``source``, ``phi_pred``, ``phi_predio``, ``error_map``.
Shared atlas: ``<data-dir>/atlas_valid_mask.npz`` (``atlas``, ``valid_mask``).

Axial panels per row: subject | atlas | ||phi_pred|| | ||phi_predio|| | error_map.

``--selection`` chooses which subjects to show from a split:
  - ``easy_normal_hard`` — lowest / median / highest by ``--rank-by`` (default: ``mean_error_map``)
  - ``random``           — ``--num-samples`` random subjects (``--seed``)

Use ``--split Train`` or ``--split Train,Test``. With multiple subjects, writes one PNG each
under ``--save-dir`` (``{split}_{label}.png``). Use ``--combined`` for a single multi-row figure.

Examples:
python experiments/unigrad-io/visualize_unigrad_io_data.py --data-dir datasets/IXI_unigrad_io --split Train,Test --selection easy_normal_hard --save-dir assets/images/unigrad-io/ --no-show
python experiments/unigrad-io/visualize_unigrad_io_data.py --data-dir datasets/IXI_unigrad_io --split Val --selection random --num-samples 4 --save-dir assets/images/unigrad-io/ --no-show
python experiments/unigrad-io/visualize_unigrad_io_data.py --data-dir datasets/IXI_unigrad_io --split Train --selection easy_normal_hard --combined --save-path assets/images/unigrad-io/unigradio_train_easy_normal_hard.png --no-show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATLAS_MASK_FILENAME = "atlas_valid_mask.npz"
SUBJECT_NPZ_KEYS = frozenset({"source", "phi_pred", "phi_predio", "error_map"})
SPLITS = ("Train", "Val", "Test")
DATA_GLOB = "*.npz"
SELECTION_MODES = ("easy_normal_hard", "random")

# Displacement fields and error_map are in voxel units (see create_unigrad_io_data.py).
DISPLACEMENT_UNIT = "voxels"

COLUMN_TITLES: list[str] = [
    "subject (source)",
    "atlas (target)",
    r"zero-shot ($\|\phi_{\mathrm{pred}}\|$)",
    r"post-IO ($\|\phi_{\mathrm{pred}}^{\mathrm{IO}}\|$)",
    r"error ($\|\phi^{\mathrm{IO}} - \phi_{\mathrm{pred}}\|$)",
]

RANK_BY_LABELS: dict[str, str] = {
    "mean_error_map": "mean(error_map)",
    "max_error_map": "max(error_map)",
    "mean_phi_pred": r"mean($\|\phi_{\mathrm{pred}}\|$)",
    "mean_phi_predio": r"mean($\|\phi_{\mathrm{pred}}^{\mathrm{IO}}\|$)",
}


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    """``(3, D, H, W)`` → ``(D, H, W)`` magnitude."""
    if phi.ndim != 4 or phi.shape[0] != 3:
        raise ValueError(f"Expected phi (3, D, H, W), got {phi.shape}")
    return np.sqrt(np.sum(phi.astype(np.float64) ** 2, axis=0))


def default_slice_index(depth: int) -> int:
    return depth // 2


def axial_slice_volume(vol: np.ndarray, slice_idx: int | None) -> np.ndarray:
    """``(H, W, D)`` → 2D axial slice."""
    if vol.ndim != 3:
        raise ValueError(f"Expected volume (H, W, D), got {vol.shape}")
    d = int(vol.shape[2])
    z = default_slice_index(d) if slice_idx is None else int(slice_idx)
    z = int(np.clip(z, 0, d - 1))
    return vol[:, :, z]


def axial_slice_error_map(err: np.ndarray, slice_idx: int | None) -> np.ndarray:
    """``(D, H, W)`` → 2D slice."""
    if err.ndim != 3:
        raise ValueError(f"Expected error_map (D, H, W), got {err.shape}")
    d = int(err.shape[0])
    z = default_slice_index(d) if slice_idx is None else int(slice_idx)
    z = int(np.clip(z, 0, d - 1))
    return err[z]


def axial_slice_phi_mag(phi: np.ndarray, slice_idx: int | None) -> np.ndarray:
    mag = phi_magnitude(phi)
    d = int(mag.shape[0])
    z = default_slice_index(d) if slice_idx is None else int(slice_idx)
    z = int(np.clip(z, 0, d - 1))
    return mag[z]


def collect_files(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return sorted(split_dir.glob(DATA_GLOB))


def load_shared_atlas_hw_d(data_dir: Path) -> np.ndarray:
    """Atlas ``(H, W, D)`` from ``atlas_valid_mask.npz`` at dataset root."""
    atlas_path = Path(data_dir) / ATLAS_MASK_FILENAME
    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"Missing {ATLAS_MASK_FILENAME} under {data_dir} "
            "(run create_unigrad_io_data.py first)"
        )
    with np.load(atlas_path) as z:
        key = "atlas" if "atlas" in z.files else "target"
        if key not in z.files:
            raise KeyError(f"{atlas_path} has no 'atlas' or 'target' array")
        atlas = np.asarray(z[key])
    if atlas.ndim != 3:
        raise ValueError(f"{atlas_path}: atlas must be (H, W, D), got {atlas.shape}")
    return atlas


def load_record(path: Path, atlas_hw_d: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = SUBJECT_NPZ_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{path.name} missing required keys: {sorted(missing)}")
        rec = {k: np.asarray(data[k]) for k in SUBJECT_NPZ_KEYS}
        rec["atlas"] = atlas_hw_d
        return rec


def score_file(path: Path, rank_by: str, atlas_hw_d: np.ndarray) -> float:
    if rank_by == "mean_error_map":
        with np.load(path) as data:
            return float(np.mean(data["error_map"]))
    return rank_scalar(load_record(path, atlas_hw_d), rank_by)


def rank_scalar(blob: dict[str, np.ndarray], rank_by: str) -> float:
    err = blob["error_map"].astype(np.float64)
    if rank_by == "mean_error_map":
        return float(np.mean(err))
    if rank_by == "max_error_map":
        return float(np.max(err))
    if rank_by == "mean_phi_pred":
        return float(np.mean(phi_magnitude(blob["phi_pred"])))
    if rank_by == "mean_phi_predio":
        return float(np.mean(phi_magnitude(blob["phi_predio"])))
    raise ValueError(f"Unknown rank_by: {rank_by}")


def select_ranked(
    files: list[Path],
    rank_by: str,
    labels: tuple[str, str, str],
    atlas_hw_d: np.ndarray,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    scored = [(fp, score_file(fp, rank_by, atlas_hw_d)) for fp in files]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 1:
        return [(scored[0][0], labels[0], scored[0][1])]
    if n == 2:
        return [
            (scored[0][0], labels[0], scored[0][1]),
            (scored[1][0], labels[2], scored[1][1]),
        ]
    return [
        (scored[0][0], labels[0], scored[0][1]),
        (scored[n // 2][0], labels[1], scored[n // 2][1]),
        (scored[-1][0], labels[2], scored[-1][1]),
    ]


def pick_samples(
    files: list[Path],
    *,
    selection: str,
    rank_by: str,
    num_samples: int,
    seed: int,
    atlas_hw_d: np.ndarray,
) -> list[tuple[Path, str, float]]:
    if selection == "random":
        rng = random.Random(seed)
        chosen = rng.sample(files, min(num_samples, len(files)))
        return [(fp, fp.stem, float("nan")) for fp in chosen]
    if selection == "easy_normal_hard":
        return select_ranked(files, rank_by, ("easy", "normal", "hard"), atlas_hw_d)
    raise ValueError(f"Unknown selection: {selection}")


def resolve_slice_index(record: dict[str, np.ndarray], slice_idx: int | None) -> int:
    d = int(record["source"].shape[2])
    z = default_slice_index(d) if slice_idx is None else int(slice_idx)
    return int(np.clip(z, 0, d - 1))


def panel_limits(
    record: dict[str, np.ndarray],
    slice_idx: int | None,
    *,
    err_vmax: float | None,
    err_percentile: float,
    phi_vmax: float | None,
    phi_percentile: float,
) -> tuple[float, float]:
    err_s = axial_slice_error_map(record["error_map"], slice_idx)
    phi_p = axial_slice_phi_mag(record["phi_pred"], slice_idx)
    phi_io = axial_slice_phi_mag(record["phi_predio"], slice_idx)
    err_v = (
        float(err_vmax)
        if err_vmax is not None
        else max(float(np.percentile(err_s, err_percentile)), 1e-6)
    )
    phi_v = (
        float(phi_vmax)
        if phi_vmax is not None
        else max(
            float(np.percentile(phi_p, phi_percentile)),
            float(np.percentile(phi_io, phi_percentile)),
            1e-6,
        )
    )
    return err_v, phi_v


def format_rank_metric(rank_by: str) -> str:
    return RANK_BY_LABELS.get(rank_by, rank_by)


def format_figure_main_title(split: str, selection: str, rank_by: str) -> str:
    """First suptitle line: ``Train samples | ranked by …``."""
    if selection == "easy_normal_hard":
        return f"{split} samples | ranked by {format_rank_metric(rank_by)}"
    if selection == "random":
        return f"{split} samples | random selection"
    return f"{split} samples | {selection.replace('_', ' ')}"


def format_figure_subtitle(z: int, depth: int, nrows: int) -> str:
    line = f"axial slice z = {z} / {depth - 1}"
    if nrows > 1:
        line += f"  ·  {nrows} subjects"
    return line


def format_subject_overlay(fp: Path, label: str) -> str:
    if not label or label == fp.stem:
        return fp.stem
    return f"{fp.stem} ({label})"


def format_mean_overlay(score: float, rank_by: str | None) -> str | None:
    if rank_by is None or not np.isfinite(score):
        return None
    return f"{format_rank_metric(rank_by)} = {score:.3f}"


def set_column_headers(axes_row) -> None:
    for ax, title in zip(axes_row, COLUMN_TITLES):
        ax.set_title(title, fontsize=9, pad=4)


def format_row_side_label(
    fp: Path,
    label: str,
    score: float,
    *,
    rank_by: str | None,
) -> str:
    lines = [format_subject_overlay(fp, label)]
    mean_line = format_mean_overlay(score, rank_by)
    if mean_line is not None:
        lines.append(mean_line)
    return "\n".join(lines)


def add_row_side_label(ax, text: str) -> None:
    """Rotated row label on the left of the first panel (bottom → top)."""
    ax.text(
        -0.04,
        0.5,
        text,
        rotation=90,
        va="center",
        ha="right",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        linespacing=1.25,
    )


def render_subject_row(
    axes,
    record: dict[str, np.ndarray],
    slice_z: int,
    *,
    err_v: float,
    phi_v: float,
    show_column_headers: bool,
    fp: Path | None = None,
    label: str = "",
    score: float = float("nan"),
    rank_by: str | None = None,
) -> None:
    source = axial_slice_volume(record["source"], slice_z)
    target = axial_slice_volume(record["atlas"], slice_z)
    phi_pred_s = axial_slice_phi_mag(record["phi_pred"], slice_z)
    phi_predio_s = axial_slice_phi_mag(record["phi_predio"], slice_z)
    err_s = axial_slice_error_map(record["error_map"], slice_z)

    if show_column_headers:
        set_column_headers(axes)

    panels: list[tuple[np.ndarray, str, float | None, float | None, str | None]] = [
        (source, "gray", None, None, None),
        (target, "gray", None, None, None),
        (phi_pred_s, "hot", 0.0, phi_v, f"displacement ({DISPLACEMENT_UNIT})"),
        (phi_predio_s, "hot", 0.0, phi_v, f"displacement ({DISPLACEMENT_UNIT})"),
        (err_s, "hot", 0.0, err_v, f"‖Δφ‖ ({DISPLACEMENT_UNIT})"),
    ]
    for ax, (img, cmap, vmin, vmax, cbar_label) in zip(axes, panels):
        if vmin is None:
            ax.imshow(img, cmap=cmap, aspect="equal")
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        ax.axis("off")

    if fp is not None:
        add_row_side_label(
            axes[0],
            format_row_side_label(fp, label, score, rank_by=rank_by),
        )


def render_figure(
    picked: list[tuple[Path, str, float]],
    *,
    data_dir: Path,
    atlas_hw_d: np.ndarray,
    split: str,
    selection: str,
    rank_by: str,
    slice_idx: int | None,
    err_vmax: float | None,
    err_percentile: float,
    phi_vmax: float | None,
    phi_percentile: float,
) -> plt.Figure:
    records = [load_record(fp, atlas_hw_d) for fp, _, _ in picked]
    z = resolve_slice_index(records[0], slice_idx)

    err_v, phi_v = panel_limits(
        records[0],
        z,
        err_vmax=err_vmax,
        err_percentile=err_percentile,
        phi_vmax=phi_vmax,
        phi_percentile=phi_percentile,
    )
    for rec in records[1:]:
        ev, pv = panel_limits(
            rec,
            z,
            err_vmax=err_vmax,
            err_percentile=err_percentile,
            phi_vmax=phi_vmax,
            phi_percentile=phi_percentile,
        )
        err_v = max(err_v, ev)
        phi_v = max(phi_v, pv)

    depth = int(records[0]["source"].shape[2])
    nrows = len(picked)
    fig, axes = plt.subplots(nrows, 5, figsize=(20, 3.8 * nrows))
    axes = np.atleast_2d(axes)
    rank_for_caption = rank_by if selection == "easy_normal_hard" else None

    for row, ((fp, label, score), rec) in enumerate(zip(picked, records)):
        render_subject_row(
            axes[row],
            rec,
            z,
            err_v=err_v,
            phi_v=phi_v,
            show_column_headers=(row == 0),
            fp=fp,
            label=label,
            score=score,
            rank_by=rank_for_caption,
        )

    left = 0.11 if nrows > 1 else 0.08
    fig.tight_layout(rect=(left, 0.02, 1, 0.97), pad=0.35, h_pad=0.5, w_pad=0.15)
    fig.subplots_adjust(top=0.93)
    fig.suptitle(
        f"{format_figure_main_title(split, selection, rank_by)}\n"
        f"{format_figure_subtitle(z, depth, nrows)}",
        fontsize=11,
        y=0.985,
    )
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {path}")


def visualize_split(
    data_dir: Path,
    split: str,
    *,
    selection: str,
    rank_by: str,
    num_samples: int,
    seed: int,
    slice_idx: int | None,
    save_dir: Path | None,
    save_path: Path | None,
    combined: bool,
    err_vmax: float | None,
    err_percentile: float,
    phi_vmax: float | None,
    phi_percentile: float,
    no_show: bool,
) -> None:
    files = collect_files(data_dir, split)
    if not files:
        raise FileNotFoundError(f"No .npz under {data_dir / split}")

    atlas_hw_d = load_shared_atlas_hw_d(data_dir)
    picked = pick_samples(
        files,
        selection=selection,
        rank_by=rank_by,
        num_samples=num_samples,
        seed=seed,
        atlas_hw_d=atlas_hw_d,
    )
    print(f"{split}: {len(files)} volumes — {selection} → {len(picked)} figure(s)")
    for fp, label, score in picked:
        if np.isfinite(score):
            print(f"  {label:8s}  score={score:.6f}  {fp.name}")
        else:
            print(f"  {label:8s}  {fp.name}")

    if combined:
        fig = render_figure(
            picked,
            data_dir=data_dir,
            atlas_hw_d=atlas_hw_d,
            split=split,
            selection=selection,
            rank_by=rank_by,
            slice_idx=slice_idx,
            err_vmax=err_vmax,
            err_percentile=err_percentile,
            phi_vmax=phi_vmax,
            phi_percentile=phi_percentile,
        )
        out = save_path or (save_dir / f"{split.lower()}_{selection}.png" if save_dir else None)
        if out is not None:
            save_figure(fig, out)
        if no_show:
            plt.close(fig)
        else:
            plt.show()
        return

    for fp, label, score in picked:
        fig = render_figure(
            [(fp, label, score)],
            data_dir=data_dir,
            atlas_hw_d=atlas_hw_d,
            split=split,
            selection=selection,
            rank_by=rank_by,
            slice_idx=slice_idx,
            err_vmax=err_vmax,
            err_percentile=err_percentile,
            phi_vmax=phi_vmax,
            phi_percentile=phi_percentile,
        )
        if save_dir is not None:
            safe = label.replace("/", "_") if label else fp.stem
            save_figure(fig, save_dir / f"{split.lower()}_{safe}.png")
        elif save_path is not None and len(picked) == 1:
            save_figure(fig, save_path)
        if no_show:
            plt.close(fig)
        else:
            plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize 3D UniGrad ICON IO .npz volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/IXI_unigrad_io"),
    )
    p.add_argument(
        "--split",
        type=str,
        default="Train",
        help="One split or comma-separated (e.g. Train,Test).",
    )
    p.add_argument(
        "--selection",
        type=str,
        default="easy_normal_hard",
        choices=list(SELECTION_MODES),
        help="How to pick subjects from each split.",
    )
    p.add_argument(
        "--rank-by",
        type=str,
        default="mean_error_map",
        choices=["mean_error_map", "max_error_map", "mean_phi_pred", "mean_phi_predio"],
        help="Metric to rank easy / normal / hard (default: mean over error_map volume).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random subjects when --selection random.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slice-index", type=int, default=None, metavar="Z")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for per-subject PNGs (default: assets/images/unigrad-io/3d/viz).",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Single output path (use with --combined or one subject).",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Stack all selected subjects in one multi-row figure.",
    )
    p.add_argument("--err-vmax", type=float, default=None)
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--phi-vmax", type=float, default=None)
    p.add_argument("--phi-percentile", type=float, default=99.0)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    atlas_path = args.data_dir / ATLAS_MASK_FILENAME
    if not atlas_path.is_file():
        print(f"ERROR: missing shared atlas file: {atlas_path}", file=sys.stderr)
        return 2

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    for s in splits:
        if s not in SPLITS:
            print(f"ERROR: unknown split {s!r} (choose {SPLITS})", file=sys.stderr)
            return 2

    save_dir = args.save_dir
    if save_dir is None and args.save_path is None and args.no_show:
        save_dir = Path("assets/images/unigrad-io/3d/viz")

    for split in splits:
        try:
            visualize_split(
                args.data_dir,
                split,
                selection=args.selection,
                rank_by=args.rank_by,
                num_samples=args.num_samples,
                seed=args.seed,
                slice_idx=args.slice_index,
                save_dir=save_dir,
                save_path=args.save_path if len(splits) == 1 else None,
                combined=args.combined,
                err_vmax=args.err_vmax,
                err_percentile=args.err_percentile,
                phi_vmax=args.phi_vmax,
                phi_percentile=args.phi_percentile,
                no_show=args.no_show,
            )
        except FileNotFoundError as e:
            print(f"Skip {split}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
