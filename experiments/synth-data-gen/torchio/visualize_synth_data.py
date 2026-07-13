#!/usr/bin/env python3
"""
Visualize HCP synthetic registration NPZ samples (``create_synth_data.py`` output).

Columns: source, moving, u vectors, ``‖u‖``; optional checkerboard (``--checkerboard``).
Axial display is radiological (``rot90``, posterior up).

Modes:
  - Dry-run folder (flat ``*.npz``): ``--selection per_class``
  - Full cohort (``Train``/``Val``/``Test``):
      ``--selection random`` (default) → one random sample per class per split
      (up to 15 plots) with orthogonal/montage views
      ``--selection min_median_max`` → three separate plots per split (min / median / max
      by ``--u-metric``), not stacked in one figure

``--run-view orthogonal|montage`` controls row layout (``--montage-z-step`` for montage).

Examples:
# Dry-run figures
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_dryrun --selection per_class --save-dir assets/images/synth-data/torchio/hcp/dryrun_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_dryrun --selection per_class --save-dir assets/images/synth-data/torchio/hcp/dryrun_montage --no-show --run-view montage --montage-z-step 10 --u-contours --checkerboard

# Full cohort: random one-per-class × all splits (15 plots) + CSV stats
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_100 --selection random --save-dir assets/images/synth-data/torchio/hcp/full100_random --no-show --run-view orthogonal --u-contours --checkerboard

# Full cohort: min/median/max by mean ‖u‖ (3 plots per split)
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_100 --selection min_median_max --u-metric mean --save-dir assets/images/synth-data/torchio/hcp/full100_mmm --no-show --run-view orthogonal --u-contours --checkerboard

# Single split only
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_100 --split Train --selection random --save-dir assets/images/synth-data/torchio/hcp/full100_train --no-show
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HCP_SYNTH_GLOB = "*.npz"
HCP_REQUIRED_KEYS = frozenset(
    {
        "source",
        "moving",
        "u",
        "source_mask",
        "moving_mask",
        "identity_grid_mask",
        "deformation_class",
        "subject_id",
    }
)

DEFORM_CLASSES = ("none", "rigid", "affine", "elastic", "affine_elastic")
FULL_SPLITS = ("Train", "Val", "Test")

DEFORM_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("rigid", "rig"),
    ("affine", "aff"),
    ("elastic", "ela"),
)

_CHECKER_TILE = 16
_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9
_QUIVER_COLOR = "lime"
_U_COLOR_PERCENTILE = 99.0  # per-row ‖u‖ color scale cap


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def _unpack_scalar_str(raw) -> str:
    a = np.asarray(raw)
    if a.size == 0:
        return ""
    return str(a.reshape(-1)[0])


def resolve_npz_dir(input_dir: Path, split: str | None) -> Path:
    """Use split subfolder when present; otherwise flat layout (e.g. dry-run output)."""
    if split:
        split_dir = input_dir / split
        if split_dir.is_dir():
            return split_dir
    return input_dir


def collect_npz_files(input_dir: Path, split: str | None) -> list[Path]:
    data_dir = resolve_npz_dir(input_dir, split)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob(HCP_SYNTH_GLOB))
    if not files and split:
        files = sorted(input_dir.glob(HCP_SYNTH_GLOB))
    return files


def default_slice_index(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def axial_slice(vol: np.ndarray, slice_index: int | None) -> tuple[np.ndarray, int]:
    z = default_slice_index(vol) if slice_index is None else int(slice_index)
    z = max(0, min(z, vol.shape[2] - 1))
    return vol[:, :, z], z


def orient_axial(sl: np.ndarray) -> np.ndarray:
    """Radiological axial display (CCW 90°): anterior up, posterior down."""
    return np.rot90(sl)


def orient_axial_u_inplane(u_inplane: np.ndarray) -> np.ndarray:
    """Rotate in-plane displacement to match ``orient_axial`` on the slice."""
    u0, u1 = u_inplane[0], u_inplane[1]
    u0_r = np.rot90(u1)
    u1_r = np.rot90(-u0)
    return np.stack([u0_r, u1_r], axis=0)


def plane_slice(vol: np.ndarray, plane: str, index: int | None) -> tuple[np.ndarray, int]:
    """Return 2D slice for plane in {axial, coronal, sagittal}."""
    if plane == "axial":
        z = default_slice_index(vol) if index is None else int(index)
        z = max(0, min(z, vol.shape[2] - 1))
        return vol[:, :, z], z
    if plane == "coronal":
        y = int(vol.shape[1] // 2) if index is None else int(index)
        y = max(0, min(y, vol.shape[1] - 1))
        return vol[:, y, :], y
    if plane == "sagittal":
        x = int(vol.shape[0] // 2) if index is None else int(index)
        x = max(0, min(x, vol.shape[0] - 1))
        return vol[x, :, :], x
    raise ValueError(f"Unknown plane: {plane}")


def plane_u_inplane_slice(u: np.ndarray, plane: str, index: int) -> np.ndarray:
    """In-plane displacement (2,H,W) for requested plane."""
    if plane == "axial":
        z = max(0, min(int(index), u.shape[3] - 1))
        return np.stack([u[0, :, :, z], u[1, :, :, z]], axis=0)
    if plane == "coronal":
        y = max(0, min(int(index), u.shape[2] - 1))
        return np.stack([u[0, :, y, :], u[2, :, y, :]], axis=0)
    if plane == "sagittal":
        x = max(0, min(int(index), u.shape[1] - 1))
        return np.stack([u[1, x, :, :], u[2, x, :, :]], axis=0)
    raise ValueError(f"Unknown plane: {plane}")


def checkerboard_mix(a: np.ndarray, b: np.ndarray, tile: int = _CHECKER_TILE) -> np.ndarray:
    """
    Interleave tiles from ``a`` and ``b`` like a chessboard.

    Odd tiles show source intensity; even tiles show moving. Where anatomy lines
    up across tile edges, deformation/mismatch is small; jagged edges highlight
    local differences.
    """
    h, w = a.shape
    yy, xx = np.indices((h, w))
    use_a = ((yy // tile) + (xx // tile)) % 2 == 0
    return np.where(use_a, a, b)


def load_sample(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = HCP_REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        return {
            "source": np.asarray(data["source"]),
            "moving": np.asarray(data["moving"]),
            "u": np.asarray(data["u"]),
            "source_mask": np.asarray(data["source_mask"]),
            "moving_mask": np.asarray(data["moving_mask"]),
            "identity_grid_mask": np.asarray(data["identity_grid_mask"]),
            "deformation_class": _unpack_scalar_str(data["deformation_class"]),
            "subject_id": _unpack_scalar_str(data["subject_id"]),
        }


def sample_u_stats(sample: dict) -> dict[str, float]:
    """‖u‖ min/Q1/mean/Q3/max over the full volume (all voxels)."""
    mag = displacement_magnitude(sample["u"].astype(np.float64))
    vals = mag.ravel()
    return {
        "min": float(np.min(vals)),
        "q1": float(np.percentile(vals, 25)),
        "mean": float(np.mean(vals)),
        "q3": float(np.percentile(vals, 75)),
        "max": float(np.max(vals)),
    }


def scalar_u_score(u: np.ndarray, metric: str) -> float:
    """Scalar ‖u‖ score over the full volume (for min/median/max selection)."""
    mag = displacement_magnitude(u.astype(np.float64)).ravel()
    if metric == "mean":
        return float(np.mean(mag))
    if metric == "max":
        return float(np.max(mag))
    raise ValueError(f"metric must be 'mean' or 'max', got {metric!r}")


def select_min_median_max_files(
    files: list[Path],
    u_metric: str,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    scored: list[tuple[Path, float]] = []
    for fp in files:
        sample = load_sample(fp)
        scored.append(
            (fp, scalar_u_score(sample["u"], u_metric))
        )
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


def select_deformation_class_examples(
    files: list[Path],
    seed: int,
) -> list[tuple[Path, str, float]]:
    pools: dict[str, list[Path]] = {cls: [] for cls, _ in DEFORM_ROW_ORDER}
    for fp in files:
        cls = load_sample(fp).get("deformation_class")
        if cls in pools:
            pools[str(cls)].append(fp)
    for cls, suf in DEFORM_ROW_ORDER:
        if not pools[cls]:
            pools[cls] = [fp for fp in files if fp.stem.endswith(f"_{suf}")]

    rng = random.Random(seed)
    picked: list[tuple[Path, str, float]] = []
    for cls, suf in DEFORM_ROW_ORDER:
        pool = pools[cls]
        if not pool:
            raise FileNotFoundError(
                f"No '{cls}' (*_{suf}.npz) sample found in split ({len(files)} files)."
            )
        picked.append((rng.choice(pool), cls, float("nan")))
    return picked


def group_class_examples(files: list[Path], seed: int) -> list[tuple[str, Path]]:
    """One random sample per deformation class."""
    pools: dict[str, list[Path]] = {cls: [] for cls in DEFORM_CLASSES}
    for fp in files:
        cls = load_sample(fp).get("deformation_class")
        if cls in pools:
            pools[str(cls)].append(fp)

    rng = random.Random(seed)
    groups: list[tuple[str, Path]] = []
    missing: list[str] = []
    for cls in DEFORM_CLASSES:
        pool = pools[cls]
        if not pool:
            missing.append(cls)
            continue
        groups.append((cls, rng.choice(pool)))
    if missing:
        raise FileNotFoundError(
            f"per_class missing class(es): {', '.join(missing)}. "
            f"Run: create_synth_data.py --dry-run 5"
        )
    return groups


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")


def _row_label(subject_id: str, extra: dict) -> str:
    lines = [subject_id]
    if extra.get("deformation_class"):
        lines.append(str(extra["deformation_class"]))
    return "\n".join(lines)


def _render_figure(
    picked: list[tuple[Path, str, float]],
    *,
    save_path: Path | None,
    no_show: bool,
    title: str,
    subtitle: str,
    z_slice_index: int | None,
    quiver_stride: int,
    row_planes: list[str] | None = None,
    row_slice_indices: list[int] | None = None,
    use_u_contours: bool = False,
    use_checkerboard: bool = False,
    sample_stats_note: str | None = None,
    row_h: float = 3.2,
    announce_save: bool = True,
) -> None:
    nrows = len(picked)
    ncols = 5 if use_checkerboard else 4
    # source, moving, u vectors, ‖u‖, optional checkerboard
    col_titles = ["Source (fixed)", "Warped (moving)", "u vectors", r"$\|u\|$"]
    if use_checkerboard:
        col_titles.append("checkerboard")

    plane_idx_notes: list[str] = []
    row_u_vmax: list[float] = []
    for row, (fp, _, _) in enumerate(picked):
        sample = load_sample(fp)
        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(sample["source"], plane, z_slice_index)[1]
        plane_idx_notes.append(f"{plane[0]}={idx}")
        if plane == "axial":
            u_sl = sample["u"][:, :, :, idx]
        elif plane == "coronal":
            u_sl = sample["u"][:, :, idx, :]
        else:
            u_sl = sample["u"][:, idx, :, :]
        mag = displacement_magnitude(u_sl.astype(np.float64)).ravel()
        row_u_vmax.append(max(float(np.percentile(mag, _U_COLOR_PERCENTILE)), 1e-6))

    full_subtitle = f"{subtitle} · {', '.join(plane_idx_notes)}"

    plt.rcParams.update({"font.family": _FONT, "figure.dpi": _DPI, "savefig.dpi": _DPI})
    fig_w = 3.0 * ncols + 1.6
    fig_h = row_h * nrows + 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    im_u_last = None
    for row, (file_path, rank_label, _) in enumerate(picked):
        sample = load_sample(file_path)
        source = sample["source"]
        moving = sample["moving"]
        u = sample["u"]
        extra = {
            k: sample[k]
            for k in ("deformation_class", "subject_id")
            if k in sample
        }

        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(source, plane, z_slice_index)[1]
        src_sl, _ = plane_slice(source, plane, idx)
        mov_sl, _ = plane_slice(moving, plane, idx)
        if plane == "axial":
            u_mag_raw = displacement_magnitude(u[:, :, :, idx].astype(np.float64))
        elif plane == "coronal":
            u_mag_raw = displacement_magnitude(u[:, :, idx, :].astype(np.float64))
        else:
            u_mag_raw = displacement_magnitude(u[:, idx, :, :].astype(np.float64))
        u_inplane = orient_axial_u_inplane(plane_u_inplane_slice(u, plane, idx))
        u_mag_sl = orient_axial(u_mag_raw)
        src_disp = orient_axial(src_sl)
        mov_disp = orient_axial(mov_sl)
        u_vmax = row_u_vmax[row]

        subject_id = extra.get("subject_id") or file_path.stem.split("_")[0]
        plane_tag = rank_label if rank_label else plane
        row_title = f"{subject_id} · {plane_tag}"

        ax_src = axes[row, 0]
        ax_src.imshow(src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        _style_axis(ax_src)
        ax_src.set_ylabel(
            _row_label(row_title, extra),
            fontsize=_LABEL,
            rotation=90,
            ha="center",
            va="center",
            labelpad=18,
        )

        ax_mov = axes[row, 1]
        ax_mov.imshow(mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        _style_axis(ax_mov)

        ax_q = axes[row, 2]
        ax_q.imshow(mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        step = max(14, quiver_stride + 6)
        uu = u_inplane[0, ::step, ::step]
        vv = u_inplane[1, ::step, ::step]
        xs, ys = np.meshgrid(
            np.arange(0, u_inplane.shape[2], step),
            np.arange(0, u_inplane.shape[1], step),
        )
        ax_q.quiver(
            xs,
            ys,
            uu,
            vv,
            color=_QUIVER_COLOR,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            headwidth=4.2,
            headlength=5.2,
        )
        _style_axis(ax_q)

        ax_u = axes[row, 3]
        im_u_last = ax_u.imshow(
            u_mag_sl, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        if use_u_contours:
            levels = np.linspace(0.15 * u_vmax, 0.95 * u_vmax, 6)
            ax_u.contour(u_mag_sl, levels=levels, colors="white", linewidths=0.5, alpha=0.7)
        _style_axis(ax_u)

        if use_checkerboard:
            ax_c = axes[row, 4]
            cb = checkerboard_mix(src_disp, mov_disp)
            ax_c.imshow(cb, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
            _style_axis(ax_c)

        if row == 0:
            for col, t in enumerate(col_titles):
                axes[0, col].set_title(t, fontsize=_SUBTITLE, fontweight="medium", pad=10)

    if im_u_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.22, 0.018, 0.56])
        cbar = fig.colorbar(im_u_last, cax=cbar_ax)
        cbar.set_label(r"$\|u\|$ (voxels)", fontsize=_LABEL)
        cbar.ax.tick_params(labelsize=_LABEL - 1)

    fig.suptitle(title, fontsize=_TITLE, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, full_subtitle, ha="center", va="top", fontsize=_LABEL, color="black")
    bottom = 0.10 if sample_stats_note else 0.08
    if sample_stats_note:
        fig.text(
            0.5,
            0.02,
            sample_stats_note,
            ha="center",
            va="bottom",
            fontsize=_LABEL - 1,
            color="0.25",
            family="monospace",
        )
    fig.subplots_adjust(left=0.24, right=0.90, top=0.86, bottom=bottom, wspace=0.26, hspace=0.34)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        if announce_save:
            print(f"Saved figure: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def is_full_cohort_dir(input_dir: Path) -> bool:
    return any((input_dir / sp).is_dir() for sp in FULL_SPLITS)


def resolve_full_splits(input_dir: Path, split: str | None) -> list[str]:
    if split:
        if not (input_dir / split).is_dir():
            raise FileNotFoundError(f"Split directory not found: {input_dir / split}")
        return [split]
    found = [sp for sp in FULL_SPLITS if (input_dir / sp).is_dir()]
    if not found:
        raise FileNotFoundError(
            f"No Train/Val/Test folders under {input_dir}. "
            "Pass a full-cohort root or use --selection per_class for a dry-run folder."
        )
    return found


def group_class_examples_optional(
    files: list[Path], seed: int
) -> tuple[list[tuple[str, Path]], list[str]]:
    """One random sample per class present; return (groups, missing_classes)."""
    pools: dict[str, list[Path]] = {cls: [] for cls in DEFORM_CLASSES}
    for fp in files:
        cls = load_sample(fp).get("deformation_class")
        if cls in pools:
            pools[str(cls)].append(fp)
    rng = random.Random(seed)
    groups: list[tuple[str, Path]] = []
    missing: list[str] = []
    for cls in DEFORM_CLASSES:
        pool = pools[cls]
        if not pool:
            missing.append(cls)
            continue
        groups.append((cls, rng.choice(pool)))
    return groups, missing


def _views_for_sample(
    sample: dict,
    *,
    z_slice_index: int | None,
    montage_z_step: int,
    run_view: str,
) -> tuple[list[str], list[int], list[str], str]:
    if run_view == "orthogonal":
        x0 = sample["source"].shape[0] // 2
        y0 = sample["source"].shape[1] // 2
        z0 = axial_slice(sample["source"], z_slice_index)[1]
        return (
            ["axial", "coronal", "sagittal"],
            [z0, y0, x0],
            ["axial", "coronal", "sagittal"],
            "Orthogonal sanity check · Radiological-style display",
        )
    z0 = axial_slice(sample["source"], z_slice_index)[1]
    z_offsets = [-int(montage_z_step), 0, int(montage_z_step)]
    idx_rows = [
        max(0, min(sample["source"].shape[2] - 1, z0 + dz)) for dz in z_offsets
    ]
    return (
        ["axial", "axial", "axial"],
        idx_rows,
        [f"z{dz:+d}" for dz in z_offsets],
        "3-slice montage sanity check · Radiological-style display",
    )


def _render_single_sample_plot(
    fp: Path,
    *,
    save_path: Path,
    title: str,
    z_slice_index: int | None,
    quiver_stride: int,
    use_u_contours: bool,
    use_checkerboard: bool,
    montage_z_step: int,
    run_view: str,
) -> dict:
    sample = load_sample(fp)
    u_stats = sample_u_stats(sample)
    stats_note = (
        f"‖u‖ voxels: min={u_stats['min']:.2f}  Q1={u_stats['q1']:.2f}  "
        f"mean={u_stats['mean']:.2f}  Q3={u_stats['q3']:.2f}  max={u_stats['max']:.2f}"
    )
    plane_rows, idx_rows, rank_rows, subtitle = _views_for_sample(
        sample,
        z_slice_index=z_slice_index,
        montage_z_step=montage_z_step,
        run_view=run_view,
    )
    picked = [(fp, r, float("nan")) for r in rank_rows]
    _render_figure(
        picked,
        save_path=save_path,
        no_show=True,
        title=title,
        subtitle=subtitle,
        z_slice_index=z_slice_index,
        quiver_stride=quiver_stride,
        row_planes=plane_rows,
        row_slice_indices=idx_rows,
        use_u_contours=use_u_contours,
        use_checkerboard=use_checkerboard,
        sample_stats_note=stats_note,
        row_h=3.0,
        announce_save=False,
    )
    return {
        "file": fp.name,
        "subject_id": sample.get("subject_id"),
        "deformation_class": sample.get("deformation_class"),
        **u_stats,
    }


def compute_split_class_u_stats_rows(input_dir: Path, splits: list[str]) -> list[dict]:
    """Per (split, class): per-sample ‖u‖ stats, then mean over samples."""
    rows: list[dict] = []
    for sp in splits:
        files = collect_npz_files(input_dir, sp)
        pools: dict[str, list[Path]] = {cls: [] for cls in DEFORM_CLASSES}
        for fp in files:
            cls = load_sample(fp).get("deformation_class")
            if cls in pools:
                pools[str(cls)].append(fp)
        for cls in DEFORM_CLASSES:
            per_sample: list[dict[str, float]] = []
            for fp in pools[cls]:
                per_sample.append(sample_u_stats(load_sample(fp)))
            if not per_sample:
                rows.append(
                    {
                        "split": sp,
                        "deformation_class": cls,
                        "n_samples": 0,
                        "min": float("nan"),
                        "q1": float("nan"),
                        "mean": float("nan"),
                        "q3": float("nan"),
                        "max": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "split": sp,
                    "deformation_class": cls,
                    "n_samples": len(per_sample),
                    "min": float(np.mean([s["min"] for s in per_sample])),
                    "q1": float(np.mean([s["q1"] for s in per_sample])),
                    "mean": float(np.mean([s["mean"] for s in per_sample])),
                    "q3": float(np.mean([s["q3"] for s in per_sample])),
                    "max": float(np.mean([s["max"] for s in per_sample])),
                }
            )
    return rows


def save_split_class_u_stats_csv(rows: list[dict], csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "deformation_class",
        "n_samples",
        "min",
        "q1",
        "mean",
        "q3",
        "max",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote {csv_path} (‖u‖ stats over all samples in each split × class)")
    return csv_path


def visualize_full_cohort(
    input_dir: Path,
    split: str | None,
    save_dir: Path,
    no_show: bool,
    *,
    selection: str,
    u_metric: str,
    seed: int,
    z_slice_index: int | None,
    quiver_stride: int,
    use_u_contours: bool,
    use_checkerboard: bool,
    montage_z_step: int,
    run_view: str,
) -> None:
    splits = resolve_full_splits(input_dir, split)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_figs = 0
    print(
        f"Full cohort viz: selection={selection}  splits={', '.join(splits)}  "
        f"view={run_view}  → {save_dir}"
    )

    for sp in splits:
        files = collect_npz_files(input_dir, sp)
        if not files:
            print(f"Warning: no NPZ in {sp}; skipping.", file=sys.stderr)
            continue
        out_split = save_dir / sp
        out_split.mkdir(parents=True, exist_ok=True)

        if selection == "random":
            groups, missing = group_class_examples_optional(files, seed=seed + hash(sp) % 10007)
            if missing:
                print(
                    f"Warning: {sp} missing class(es) {', '.join(missing)}; "
                    "plotting available classes only.",
                    file=sys.stderr,
                )
            for label, fp in groups:
                _render_single_sample_plot(
                    fp,
                    save_path=out_split / f"{label}.png",
                    title=f"HCP Synthetic — {sp} / {label}",
                    z_slice_index=z_slice_index,
                    quiver_stride=quiver_stride,
                    use_u_contours=use_u_contours,
                    use_checkerboard=use_checkerboard,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                )
                n_figs += 1
            print(f"  {sp}: {len(groups)} class plot(s) from {len(files)} NPZs")
        elif selection == "min_median_max":
            picked = select_min_median_max_files(files, u_metric)
            for fp, rank, score in picked:
                _render_single_sample_plot(
                    fp,
                    save_path=out_split / f"{rank}.png",
                    title=(
                        f"HCP Synthetic — {sp} / {rank} "
                        f"({u_metric} ‖u‖={score:.3f})"
                    ),
                    z_slice_index=z_slice_index,
                    quiver_stride=quiver_stride,
                    use_u_contours=use_u_contours,
                    use_checkerboard=use_checkerboard,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                )
                n_figs += 1
            print(
                f"  {sp}: min/median/max by {u_metric} ‖u‖ "
                f"({len(files)} NPZs → {len(picked)} plots)"
            )
        else:
            raise ValueError(f"Full cohort does not support selection={selection!r}")

    print("Computing split × class ‖u‖ stats (all samples)…")
    rows = compute_split_class_u_stats_rows(input_dir, splits)
    save_split_class_u_stats_csv(rows, save_dir / "split_class_u_stats.csv")
    if no_show:
        plt.close("all")
    print(f"Done: {n_figs} figures under {save_dir}")


def visualize_per_class_combinations(
    input_dir: Path,
    split: str | None,
    save_dir: Path,
    no_show: bool,
    *,
    seed: int,
    z_slice_index: int | None,
    quiver_stride: int,
    use_u_contours: bool,
    use_checkerboard: bool,
    montage_z_step: int,
    run_view: str,
) -> None:
    files = collect_npz_files(input_dir, split)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{resolve_npz_dir(input_dir, split)}'.")
    groups = group_class_examples(files, seed=seed)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Dry-run viz: {len(groups)} classes from {len(files)} NPZs  "
        f"view={run_view}  → {save_dir}"
    )
    chosen_stats: list[dict] = []
    for label, fp in groups:
        sample = load_sample(fp)
        u_stats = sample_u_stats(sample)
        chosen_stats.append(
            {
                "class": label,
                "file": fp.name,
                "subject_id": sample.get("subject_id"),
                **u_stats,
            }
        )
        stats_note = (
            f"‖u‖ voxels: min={u_stats['min']:.2f}  Q1={u_stats['q1']:.2f}  "
            f"mean={u_stats['mean']:.2f}  Q3={u_stats['q3']:.2f}  max={u_stats['max']:.2f}"
        )
        if run_view == "orthogonal":
            x0 = sample["source"].shape[0] // 2
            y0 = sample["source"].shape[1] // 2
            z0 = axial_slice(sample["source"], z_slice_index)[1]
            plane_rows = ["axial", "coronal", "sagittal"]
            idx_rows = [z0, y0, x0]
            rank_rows = ["axial", "coronal", "sagittal"]
            subtitle = "Orthogonal sanity check · Radiological-style display"
        else:
            z0 = axial_slice(sample["source"], z_slice_index)[1]
            z_offsets = [-int(montage_z_step), 0, int(montage_z_step)]
            plane_rows = ["axial", "axial", "axial"]
            idx_rows = [max(0, min(sample["source"].shape[2] - 1, z0 + dz)) for dz in z_offsets]
            rank_rows = [f"z{dz:+d}" for dz in z_offsets]
            subtitle = "3-slice montage sanity check · Radiological-style display"
        picked = [(fp, r, float("nan")) for r in rank_rows]
        _render_figure(
            picked,
            save_path=save_dir / f"{label}.png",
            no_show=True,
            title=f"HCP Synthetic Data Plot — {label}",
            subtitle=subtitle,
            z_slice_index=z_slice_index,
            quiver_stride=quiver_stride,
            row_planes=plane_rows,
            row_slice_indices=idx_rows,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            sample_stats_note=stats_note,
            row_h=3.0,
            announce_save=False,
        )
    csv_path = save_dir / "chosen_sample_u_stats.csv"
    fieldnames = ["class", "file", "subject_id", "min", "q1", "mean", "q3", "max"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in chosen_stats:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {csv_path} (‖u‖ stats for plotted samples only)")
    if no_show:
        plt.close("all")
    print(f"Done: {len(groups)} figures under {save_dir}")


def visualize_samples(
    input_dir: Path,
    split: str | None,
    save_path: Path | None,
    no_show: bool,
    *,
    selection: str,
    u_metric: str,
    seed: int,
    z_slice_index: int | None,
    quiver_stride: int,
    save_dir: Path | None = None,
    use_u_contours: bool = False,
    use_checkerboard: bool = False,
    montage_z_step: int = 10,
    run_view: str = "orthogonal",
) -> None:
    # Dry-run path (unchanged): flat folder + per_class.
    if selection == "per_class":
        if save_dir is None:
            if save_path is not None:
                save_dir = Path(save_path).parent / Path(save_path).stem
            else:
                save_dir = Path("assets/images/synth-data/torchio/hcp/per_class")
        visualize_per_class_combinations(
            input_dir,
            None,
            save_dir,
            no_show,
            seed=seed,
            z_slice_index=z_slice_index,
            quiver_stride=quiver_stride,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            montage_z_step=montage_z_step,
            run_view=run_view,
        )
        return

    # Full cohort: Train/Val/Test → one plot per sample (random per class, or min/median/max).
    if is_full_cohort_dir(input_dir) and selection in ("random", "min_median_max"):
        if save_dir is None:
            if save_path is not None:
                save_dir = Path(save_path).parent / Path(save_path).stem
            else:
                save_dir = Path("assets/images/synth-data/torchio/hcp/full_cohort")
        visualize_full_cohort(
            input_dir,
            split,
            save_dir,
            no_show,
            selection=selection,
            u_metric=u_metric,
            seed=seed,
            z_slice_index=z_slice_index,
            quiver_stride=quiver_stride,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            montage_z_step=montage_z_step,
            run_view=run_view,
        )
        return

    files = collect_npz_files(input_dir, split)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{resolve_npz_dir(input_dir, split)}'.")

    if selection == "random":
        picked = select_deformation_class_examples(files, seed)
        examples_note = f"Rigid / affine / elastic examples (seed = {seed})"
    elif selection == "min_median_max":
        picked = select_min_median_max_files(files, u_metric)
        examples_note = f"Min / median / max of {u_metric} " + r"$\|u\|$"
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    split_label = split or "all"
    _render_figure(
        picked,
        save_path=save_path,
        no_show=no_show,
        title="HCP Synthetic Data Plot",
        subtitle=f"{examples_note} · {split_label} split · Radiological-style display",
        z_slice_index=z_slice_index,
        quiver_stride=quiver_stride,
        use_u_contours=use_u_contours,
        use_checkerboard=use_checkerboard,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize HCP synthetic NPZ (source, moving, u from create_synth_data.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/synth-data/torchio/hcp"),
        help="Dry-run flat folder, or full-cohort root with Train/Val/Test.",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Full cohort: Train/Val/Test. Omit to plot all present splits. "
            "Ignored for dry-run --selection per_class."
        ),
    )
    p.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["min_median_max", "random", "per_class"],
        help=(
            "Dry-run: per_class. Full cohort: random (one sample/class/split) or "
            "min_median_max (three separate plots per split by --u-metric)."
        ),
    )
    p.add_argument(
        "--u-metric",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Scalar per volume for min/median/max selection (‖u‖).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--z-slice-index",
        type=int,
        default=None,
        help="Axial z slice index (default: mid z).",
    )
    p.add_argument(
        "--quiver-stride",
        type=int,
        default=20,
        help="Base spacing for sparse quiver arrows (voxels).",
    )
    p.add_argument(
        "--u-contours",
        action="store_true",
        help="Overlay contour lines on ‖u‖ magnitude map.",
    )
    p.add_argument(
        "--checkerboard",
        action="store_true",
        help=(
            "Add checkerboard column: alternating tiles of source and moving "
            "(highlights local mismatch at tile edges)."
        ),
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for per_class (dry-run) or full-cohort plots "
            "(split subfolders + split_class_u_stats.csv)."
        ),
    )
    p.add_argument(
        "--montage-z-step",
        type=int,
        default=10,
        metavar="VOX",
        help="Axial offset for --run-view montage (default 10).",
    )
    p.add_argument(
        "--run-view",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "montage"],
        help="Row layout: orthogonal planes (default) or 3-slice axial montage.",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("assets/images/synth-data/torchio/hcp/hcp_synth_preview.png"),
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    # Dry-run: leave split unset. Full cohort: None means all splits; else use given split.
    if args.selection == "per_class":
        split = args.split or None
    elif is_full_cohort_dir(data_dir):
        split = args.split  # None → all Train/Val/Test present
    else:
        split = args.split or "Train"

    visualize_samples(
        data_dir,
        split,
        args.save_path,
        args.no_show,
        selection=args.selection,
        u_metric=args.u_metric,
        seed=args.seed,
        z_slice_index=args.z_slice_index,
        quiver_stride=args.quiver_stride,
        save_dir=args.save_dir,
        use_u_contours=args.u_contours,
        use_checkerboard=args.checkerboard,
        montage_z_step=args.montage_z_step,
        run_view=args.run_view,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
