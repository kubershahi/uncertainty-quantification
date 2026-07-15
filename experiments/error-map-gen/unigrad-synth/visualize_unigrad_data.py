#!/usr/bin/env python3
"""
Visualize HCP UniGrad synth error-map NPZ (``create_unigrad_synth_data.py`` output).

Columns: Source (fixed), Warped (moving), ``‖u_gt‖``, ``‖u_pred‖``, error map;
optional ``--cosine-similarity`` adds cosine(u_gt, u_pred) after the error map.
Axial display is radiological (``rot90``).

Modes (match ``visualize_synth_data.py`` output names):
  - ``--selection random`` → one random sample per class per split
    (``{save-dir}/{split}/{class}.png``)
  - ``--selection min_median_max`` → min / median / max per split from the
    Phase-I selection CSV (mean ‖u‖; no re-scoring). Default CSV:
    ``assets/images/synth-data/torchio/hcp/fullrun_mmm_orthogonal/min_median_max_selection.csv``
    Writes ``{save-dir}/{split}/{min|median|max}.png``

``--run-view orthogonal|montage`` controls row layout (``--montage-z-step`` for montage).

Examples:
python experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py --data-dir datasets/error-map/unigrad-synth/hcp --selection random --save-dir assets/images/error-map/unigrad-synth/hcp/fullrun_random_orthogonal --no-show --run-view orthogonal
python experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py --data-dir datasets/error-map/unigrad-synth/hcp --selection min_median_max --save-dir assets/images/error-map/unigrad-synth/hcp/fullrun_mmm_orthogonal --no-show --run-view orthogonal --cosine-similarity
python experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py --data-dir datasets/error-map/unigrad-synth/hcp --split Train --selection random --save-dir assets/images/error-map/unigrad-synth/hcp/fullrun_train --no-show
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HCP_UNIGRAD_GLOB = "*.npz"
HCP_UNIGRAD_REQUIRED_KEYS = frozenset(
    {
        "source",
        "moving",
        "u_gt",
        "u_pred",
        "u_error_map",
        "deformation_class",
        "subject_id",
    }
)

DEFORM_CLASSES = ("none", "rigid", "affine", "elastic", "affine_elastic")
FULL_SPLITS = ("Train", "Val", "Test")

DEFORM_TITLE_LABELS: dict[str, str] = {
    "none": "No",
    "rigid": "Rigid",
    "affine": "Affine",
    "elastic": "Elastic",
    "affine_elastic": "Affine+Elastic",
}

DEFORM_SUFFIX = {
    "none": "none",
    "rigid": "rig",
    "affine": "aff",
    "elastic": "ela",
    "affine_elastic": "aela",
}
SUFFIX_TO_CLASS = {suffix: cls for cls, suffix in DEFORM_SUFFIX.items()}

MMM_SELECTION_CSV = "min_median_max_selection.csv"
MMM_RANK_ORDER = ("min", "median", "max")
DEFAULT_MMM_CSV = Path(
    "assets/images/synth-data/torchio/hcp/fullrun_mmm_orthogonal/min_median_max_selection.csv"
)

_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9
_U_COLOR_PERCENTILE = 99.0
_ERR_COLOR_PERCENTILE = 99.0


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def _unpack_scalar_str(raw) -> str:
    a = np.asarray(raw)
    if a.size == 0:
        return ""
    return str(a.reshape(-1)[0])


def resolve_npz_dir(input_dir: Path, split: str | None) -> Path:
    if split:
        split_dir = input_dir / split
        if split_dir.is_dir():
            return split_dir
    return input_dir


def collect_npz_files(input_dir: Path, split: str | None) -> list[Path]:
    data_dir = resolve_npz_dir(input_dir, split)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob(HCP_UNIGRAD_GLOB))
    if not files and split:
        files = sorted(input_dir.glob(HCP_UNIGRAD_GLOB))
    return files


def default_slice_index(vol: np.ndarray) -> int:
    return int(vol.shape[2] // 2)


def axial_slice(vol: np.ndarray, slice_index: int | None) -> tuple[np.ndarray, int]:
    z = default_slice_index(vol) if slice_index is None else int(slice_index)
    z = max(0, min(z, vol.shape[2] - 1))
    return vol[:, :, z], z


def orient_axial(sl: np.ndarray) -> np.ndarray:
    """Radiological axial display (CCW 90°)."""
    return np.rot90(sl)


def plane_slice(vol: np.ndarray, plane: str, index: int | None) -> tuple[np.ndarray, int]:
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


def magnitude_plane_slice(u: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Magnitude of a 3-channel displacement on one plane."""
    if plane == "axial":
        z = max(0, min(int(index), u.shape[3] - 1))
        return displacement_magnitude(u[:, :, :, z].astype(np.float64))
    if plane == "coronal":
        y = max(0, min(int(index), u.shape[2] - 1))
        return displacement_magnitude(u[:, :, y, :].astype(np.float64))
    if plane == "sagittal":
        x = max(0, min(int(index), u.shape[1] - 1))
        return displacement_magnitude(u[:, x, :, :].astype(np.float64))
    raise ValueError(f"Unknown plane: {plane}")


def u_plane_vectors(u: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Return displacement vectors ``(3, H, W)`` for one plane."""
    if plane == "axial":
        z = max(0, min(int(index), u.shape[3] - 1))
        return u[:, :, :, z].astype(np.float64)
    if plane == "coronal":
        y = max(0, min(int(index), u.shape[2] - 1))
        return u[:, :, y, :].astype(np.float64)
    if plane == "sagittal":
        x = max(0, min(int(index), u.shape[1] - 1))
        return u[:, x, :, :].astype(np.float64)
    raise ValueError(f"Unknown plane: {plane}")


def cosine_similarity_map(
    u_a: np.ndarray,
    u_b: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Per-voxel cosine similarity in ``[-1, 1]``.

    Voxels where either vector has ‖u‖ ≤ ``eps`` are set to NaN (direction undefined).
    Accepts ``(3, ...)`` arrays (full volume or a plane).
    """
    a = u_a.astype(np.float64, copy=False)
    b = u_b.astype(np.float64, copy=False)
    if a.shape != b.shape or a.shape[0] != 3:
        raise ValueError(f"expected matching (3, ...) arrays, got {a.shape} vs {b.shape}")
    dot = np.sum(a * b, axis=0)
    na = np.sqrt(np.sum(a * a, axis=0))
    nb = np.sqrt(np.sum(b * b, axis=0))
    valid = (na > eps) & (nb > eps)
    cos = np.full(dot.shape, np.nan, dtype=np.float64)
    cos[valid] = np.clip(dot[valid] / (na[valid] * nb[valid]), -1.0, 1.0)
    return cos


def cosine_plane_slice(u_gt: np.ndarray, u_pred: np.ndarray, plane: str, index: int) -> np.ndarray:
    return cosine_similarity_map(
        u_plane_vectors(u_gt, plane, index),
        u_plane_vectors(u_pred, plane, index),
    )

def load_sample(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = HCP_UNIGRAD_REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        return {
            "source": np.asarray(data["source"]),
            "moving": np.asarray(data["moving"]),
            "u_gt": np.asarray(data["u_gt"], dtype=np.float32),
            "u_pred": np.asarray(data["u_pred"], dtype=np.float32),
            "u_error_map": np.asarray(data["u_error_map"], dtype=np.float32),
            "deformation_class": _unpack_scalar_str(data["deformation_class"]),
            "subject_id": _unpack_scalar_str(data["subject_id"]),
        }


def deformation_class_from_filename(npz_path: Path) -> str:
    stem = npz_path.stem
    if "_" not in stem:
        raise ValueError(f"Cannot parse deformation class from filename: {npz_path.name}")
    suffix = stem.rsplit("_", 1)[-1]
    if suffix in SUFFIX_TO_CLASS:
        return SUFFIX_TO_CLASS[suffix]
    if suffix.isdigit():
        stem = stem.rsplit("_", 1)[0]
    for cls in sorted(DEFORM_CLASSES, key=len, reverse=True):
        if stem.endswith(f"_{cls}"):
            return cls
    raise ValueError(f"Cannot parse deformation class from filename: {npz_path.name}")


def _cached_sample(path: Path, cache: dict[Path, dict]) -> dict:
    if path not in cache:
        cache[path] = load_sample(path)
    return cache[path]


def sample_u_mag_stats(u: np.ndarray) -> dict[str, float]:
    mag = displacement_magnitude(u.astype(np.float64)).ravel()
    return {
        "min": float(np.min(mag)),
        "q1": float(np.percentile(mag, 25)),
        "mean": float(np.mean(mag)),
        "q3": float(np.percentile(mag, 75)),
        "max": float(np.max(mag)),
    }


def _format_u_stats_line(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name} voxels: min={stats['min']:.2f}  Q1={stats['q1']:.2f}  "
        f"mean={stats['mean']:.2f}  Q3={stats['q3']:.2f}  max={stats['max']:.2f}"
    )


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
            f"No Train/Val/Test folders under {input_dir}."
        )
    return found


def group_class_examples_optional(
    files: list[Path], seed: int
) -> tuple[list[tuple[str, Path]], list[str]]:
    pools: dict[str, list[Path]] = {cls: [] for cls in DEFORM_CLASSES}
    for fp in files:
        cls = deformation_class_from_filename(fp)
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


def _resolve_split_npz(input_dir: Path, split: str, file_name: str) -> Path | None:
    fp = input_dir / split / file_name
    if fp.is_file():
        return fp
    fp = resolve_npz_dir(input_dir, split) / file_name
    return fp if fp.is_file() else None


def load_mmm_picks_from_csv(
    csv_path: Path,
    input_dir: Path,
    split: str,
    *,
    u_metric: str = "mean",
) -> list[tuple[Path, str, float]]:
    """Reuse Phase-I min/median/max picks (mean ‖u‖) for plotting error-map NPZ."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"min/median/max selection CSV not found: {csv_path}\n"
            "Pass --mmm-selection-csv pointing at Phase-I "
            "fullrun_mmm_orthogonal/min_median_max_selection.csv"
        )
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("u_metric") == u_metric and row.get("split") == split:
                rows.append(row)
    if not rows:
        raise ValueError(
            f"No rows for split={split!r} u_metric={u_metric!r} in {csv_path}"
        )
    rank_index = {rank: i for i, rank in enumerate(MMM_RANK_ORDER)}
    rows.sort(key=lambda r: rank_index.get(r["rank"], 99))
    picked: list[tuple[Path, str, float]] = []
    for row in rows:
        fp = _resolve_split_npz(input_dir, split, row["file"])
        if fp is None:
            raise FileNotFoundError(
                f"CSV pick missing under error-map data: {split}/{row['file']} "
                f"(looked in {input_dir}). Run create_unigrad_synth_data.py first."
            )
        picked.append((fp, row["rank"], float(row["u_score"])))
    return picked


def _mmm_rank_subtitle(rank: str, u_metric: str = "mean") -> str:
    rank_label = {"min": "Minimum", "median": "Median", "max": "Maximum"}[rank]
    return f"{rank_label} of u {u_metric} sample"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")


def _class_plot_title(deformation_class: str) -> str:
    label = DEFORM_TITLE_LABELS.get(
        deformation_class, deformation_class.replace("_", "+").title()
    )
    return f"Unigrad Synthetic Error Map Plot ({label} Transformation)"


def _view_label(run_view: str) -> str:
    return "Orthogonal View" if run_view == "orthogonal" else "Montage View"


def _plot_subtitle(subject_id: str, run_view: str, extra: str | None = None) -> str:
    base = (
        f"Subject {subject_id} - Radiological-style display - {_view_label(run_view)}"
    )
    if extra:
        return f"{base} · {extra}"
    return base


def _plane_row_label(plane: str, idx: int) -> str:
    return f"{plane} ({plane[0]}={idx})"


def _add_midheight_colorbar(
    fig: plt.Figure,
    mappable,
    *,
    gs,
    cbar_col: int,
    ref_ax: plt.Axes,
    label: str,
    ticks: list[float] | None = None,
    ticklabels: list[str] | None = None,
) -> None:
    """
    Short colorbar (~1.5× mid-row panel height), centered on ``ref_ax``.

    The colorbar sits on the **left** of its reserved column so the vertical
    label stays inside the gap and does not overlap the next image panel.
    """
    probe = fig.add_subplot(gs[0, cbar_col])
    slot = probe.get_position()
    probe.remove()

    ref = ref_ax.get_position()
    h = min(ref.height * 1.5, 0.40)
    y0 = ref.y0 + 0.5 * ref.height - 0.5 * h
    bar_w = min(max(slot.width * 0.22, 0.008), 0.011)
    # Left-align bar within the wide slot; leave the rest for the label.
    x0 = slot.x0 + 0.05 * slot.width
    cax = fig.add_axes([x0, y0, bar_w, h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=_LABEL - 1, labelpad=8, rotation=90)
    cbar.ax.tick_params(labelsize=_LABEL - 2, pad=2)
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    # Keep label close to the bar (still inside the reserved column).
    cbar.ax.yaxis.set_label_coords(3.6, 0.5)


def _views_for_sample(
    sample: dict,
    *,
    z_slice_index: int | None,
    montage_z_step: int,
    run_view: str,
) -> tuple[list[str], list[int]]:
    if run_view == "orthogonal":
        x0 = sample["source"].shape[0] // 2
        y0 = sample["source"].shape[1] // 2
        z0 = axial_slice(sample["source"], z_slice_index)[1]
        return (["axial", "coronal", "sagittal"], [z0, y0, x0])
    z0 = axial_slice(sample["source"], z_slice_index)[1]
    z_offsets = [-int(montage_z_step), 0, int(montage_z_step)]
    idx_rows = [
        max(0, min(sample["source"].shape[2] - 1, z0 + dz)) for dz in z_offsets
    ]
    return (["axial", "axial", "axial"], idx_rows)


def _render_figure(
    picked: list[tuple[Path, str, float]],
    *,
    save_path: Path | None,
    no_show: bool,
    title: str,
    subtitle: str,
    z_slice_index: int | None,
    row_planes: list[str] | None = None,
    row_slice_indices: list[int] | None = None,
    sample_stats_note: str | None = None,
    row_h: float = 3.0,
    announce_save: bool = True,
    sample_cache: dict[Path, dict] | None = None,
    show_cosine_similarity: bool = False,
) -> None:
    """
    Column layout (left → right):

      Source | Warped | ‖u_gt‖ | ‖u_pred‖ | [‖u‖ cbar] | Error Map | [error cbar]
      [| Cosine | cosine cbar]   ← only if ``show_cosine_similarity``
    """
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    nrows = len(picked)
    if show_cosine_similarity:
        # 0..3 images, 4 u-cbar, 5 err, 6 err-cbar, 7 cos, 8 cos-cbar
        n_grid_cols = 9
        img_cols = (0, 1, 2, 3, 5, 7)
        # Extra-wide cbar columns so vertical labels stay out of the next image.
        width_ratios = [1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 0.32, 1.0, 0.36]
        n_image_panels = 6
    else:
        n_grid_cols = 7
        img_cols = (0, 1, 2, 3, 5)
        width_ratios = [1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 0.32]
        n_image_panels = 5

    col_titles = {
        0: "Source (fixed)",
        1: "Warped (moving)",
        2: r"$\|u_{\mathrm{gt}}\|$",
        3: r"$\|u_{\mathrm{pred}}\|$",
        5: r"Error Map ($\|u_{\mathrm{gt}} - u_{\mathrm{pred}}\|$)",
    }
    if show_cosine_similarity:
        col_titles[7] = r"Cosine Similarity ($\cos\theta$)"

    cache: dict[Path, dict] = {} if sample_cache is None else sample_cache
    row_u_vmax: list[float] = []
    row_err_vmax: list[float] = []
    for row, (fp, _, _) in enumerate(picked):
        sample = _cached_sample(fp, cache)
        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(sample["source"], plane, z_slice_index)[1]
        mag_gt = magnitude_plane_slice(sample["u_gt"], plane, idx).ravel()
        mag_pr = magnitude_plane_slice(sample["u_pred"], plane, idx).ravel()
        err_sl, _ = plane_slice(sample["u_error_map"], plane, idx)
        u_cap = max(
            float(np.percentile(mag_gt, _U_COLOR_PERCENTILE)),
            float(np.percentile(mag_pr, _U_COLOR_PERCENTILE)),
            1e-6,
        )
        err_cap = max(float(np.percentile(err_sl.ravel(), _ERR_COLOR_PERCENTILE)), 1e-6)
        row_u_vmax.append(u_cap)
        row_err_vmax.append(err_cap)

    plt.rcParams.update({"font.family": _FONT, "figure.dpi": _DPI, "savefig.dpi": _DPI})
    fig_w = 3.0 * n_image_panels + 3.2
    fig_h = row_h * nrows + 1.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    bottom = 0.12 if sample_stats_note else 0.08
    left, right = 0.16, 0.99
    gs = GridSpec(
        nrows,
        n_grid_cols,
        figure=fig,
        width_ratios=width_ratios,
        left=left,
        right=right,
        top=0.86,
        bottom=bottom,
        wspace=0.35,
        hspace=0.34,
    )

    axes: dict[tuple[int, int], plt.Axes] = {}
    for row in range(nrows):
        for col in img_cols:
            axes[(row, col)] = fig.add_subplot(gs[row, col])

    # coolwarm: blue (−1) → white (0) → red (+1). NaN must not use light grey
    # (looks like orthogonal); use near-black for undefined ‖u‖≈0 voxels.
    cos_cmap = plt.get_cmap("coolwarm").copy()
    cos_cmap.set_bad(color="#1a1a1a")
    cos_norm = Normalize(vmin=-1.0, vmax=1.0)

    im_u_last = None
    im_err_last = None
    im_cos_last = None
    for row, (file_path, _rank_label, _) in enumerate(picked):
        sample = _cached_sample(file_path, cache)
        source = sample["source"]
        moving = sample["moving"]
        u_gt = sample["u_gt"]
        u_pred = sample["u_pred"]
        err = sample["u_error_map"]

        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(source, plane, z_slice_index)[1]

        src_sl, _ = plane_slice(source, plane, idx)
        mov_sl, _ = plane_slice(moving, plane, idx)
        err_sl, _ = plane_slice(err, plane, idx)
        mag_gt = orient_axial(magnitude_plane_slice(u_gt, plane, idx))
        mag_pr = orient_axial(magnitude_plane_slice(u_pred, plane, idx))
        src_disp = orient_axial(src_sl)
        mov_disp = orient_axial(mov_sl)
        err_disp = orient_axial(err_sl)
        u_vmax = row_u_vmax[row]
        err_vmax = row_err_vmax[row]

        ax_src = axes[(row, 0)]
        ax_src.imshow(
            src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3
        )
        _style_axis(ax_src)
        ax_src.set_ylabel(
            _plane_row_label(plane, idx),
            fontsize=_LABEL,
            rotation=90,
            ha="center",
            va="center",
            labelpad=18,
        )

        ax_mov = axes[(row, 1)]
        ax_mov.imshow(
            mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3
        )
        _style_axis(ax_mov)

        ax_gt = axes[(row, 2)]
        im_u_last = ax_gt.imshow(
            mag_gt, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        _style_axis(ax_gt)

        ax_pr = axes[(row, 3)]
        ax_pr.imshow(
            mag_pr, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        _style_axis(ax_pr)

        ax_e = axes[(row, 5)]
        im_err_last = ax_e.imshow(
            err_disp,
            cmap="hot",
            vmin=0.0,
            vmax=err_vmax,
            origin="upper",
            interpolation="nearest",
        )
        _style_axis(ax_e)

        if show_cosine_similarity:
            cos_sl = orient_axial(cosine_plane_slice(u_gt, u_pred, plane, idx))
            cos_masked = np.ma.masked_invalid(cos_sl)
            ax_c = axes[(row, 7)]
            im_cos_last = ax_c.imshow(
                cos_masked,
                cmap=cos_cmap,
                norm=cos_norm,
                origin="upper",
                interpolation="nearest",
            )
            _style_axis(ax_c)

        if row == 0:
            for col, t in col_titles.items():
                axes[(0, col)].set_title(t, fontsize=_SUBTITLE, fontweight="medium", pad=10)

    mid_row = nrows // 2
    ref_ax = axes[(mid_row, 0)]
    fig.canvas.draw()
    if im_u_last is not None:
        _add_midheight_colorbar(
            fig,
            im_u_last,
            gs=gs,
            cbar_col=4,
            ref_ax=ref_ax,
            label=r"$\|u\|$ (voxels)",
        )
    if im_err_last is not None:
        _add_midheight_colorbar(
            fig,
            im_err_last,
            gs=gs,
            cbar_col=6,
            ref_ax=ref_ax,
            label=r"$\|u_{\mathrm{gt}}-u_{\mathrm{pred}}\|$ (voxels)",
        )
    if show_cosine_similarity and im_cos_last is not None:
        _add_midheight_colorbar(
            fig,
            im_cos_last,
            gs=gs,
            cbar_col=8,
            ref_ax=ref_ax,
            label=r"$\cos\theta$ ($-1$ opposite, $0$ orthogonal, $+1$ aligned)",
            ticks=[-1.0, 0.0, 1.0],
        )

    title_x = 0.5 * (left + right)
    fig.suptitle(title, fontsize=_TITLE, fontweight="bold", x=title_x, y=0.98, ha="center")
    fig.text(title_x, 0.935, subtitle, ha="center", va="top", fontsize=_LABEL, color="black")
    if sample_stats_note:
        fig.text(
            title_x,
            0.02,
            sample_stats_note,
            ha="center",
            va="bottom",
            fontsize=_LABEL - 1,
            color="0.25",
            family="monospace",
        )

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


def _render_single_sample_plot(
    fp: Path,
    *,
    save_path: Path,
    z_slice_index: int | None,
    montage_z_step: int,
    run_view: str,
    subtitle_extra: str | None = None,
    show_cosine_similarity: bool = False,
) -> None:
    sample = load_sample(fp)
    gt_stats = sample_u_mag_stats(sample["u_gt"])
    pr_stats = sample_u_mag_stats(sample["u_pred"])
    stats_note = (
        _format_u_stats_line(r"‖u_gt‖", gt_stats)
        + "\n"
        + _format_u_stats_line(r"‖u_pred‖", pr_stats)
    )
    plane_rows, idx_rows = _views_for_sample(
        sample,
        z_slice_index=z_slice_index,
        montage_z_step=montage_z_step,
        run_view=run_view,
    )
    subject_id = sample.get("subject_id") or fp.stem.split("_")[0]
    deform_cls = str(sample.get("deformation_class") or deformation_class_from_filename(fp))
    picked = [(fp, r, float("nan")) for r in plane_rows]
    print(f"    Plotting → {save_path}")
    _render_figure(
        picked,
        save_path=save_path,
        no_show=True,
        title=_class_plot_title(deform_cls),
        subtitle=_plot_subtitle(str(subject_id), run_view, subtitle_extra),
        z_slice_index=z_slice_index,
        row_planes=plane_rows,
        row_slice_indices=idx_rows,
        sample_stats_note=stats_note,
        row_h=3.0,
        announce_save=False,
        sample_cache={fp: sample},
        show_cosine_similarity=show_cosine_similarity,
    )


def visualize_full_cohort(
    input_dir: Path,
    split: str | None,
    save_dir: Path,
    no_show: bool,
    *,
    selection: str,
    seed: int,
    z_slice_index: int | None,
    montage_z_step: int,
    run_view: str,
    mmm_selection_csv: Path,
    show_cosine_similarity: bool = False,
) -> None:
    splits = resolve_full_splits(input_dir, split)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_figs = 0
    print(
        f"UniGrad error-map viz: selection={selection}  splits={', '.join(splits)}  "
        f"view={run_view}  cosine={show_cosine_similarity}  → {save_dir}"
    )

    for sp in splits:
        files = collect_npz_files(input_dir, sp)
        if not files:
            print(f"Warning: no NPZ in {sp}; skipping.", file=sys.stderr)
            continue
        out_split = save_dir / sp
        out_split.mkdir(parents=True, exist_ok=True)

        if selection == "random":
            print(
                f"  {sp}: grouping {len(files)} NPZs by class "
                f"(filename suffix only)…"
            )
            groups, missing = group_class_examples_optional(
                files, seed=seed + hash(sp) % 10007
            )
            if missing:
                print(
                    f"Warning: {sp} missing class(es) {', '.join(missing)}; "
                    "plotting available classes only.",
                    file=sys.stderr,
                )
            for label, fp in groups:
                print(f"  {sp} / {label}: plotting {fp.name}")
                _render_single_sample_plot(
                    fp,
                    save_path=out_split / f"{label}.png",
                    z_slice_index=z_slice_index,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                    show_cosine_similarity=show_cosine_similarity,
                )
                n_figs += 1
            print(f"  {sp}: wrote {len(groups)} class plot(s)")
        elif selection == "min_median_max":
            print(f"  {sp}: loading Phase-I mean ‖u‖ picks from {mmm_selection_csv}")
            picked = load_mmm_picks_from_csv(
                mmm_selection_csv, input_dir, sp, u_metric="mean"
            )
            for fp, rank, score in picked:
                print(
                    f"  {sp} / {rank}: mean ‖u‖={score:.3f} from {fp.name} (cached)"
                )
                _render_single_sample_plot(
                    fp,
                    save_path=out_split / f"{rank}.png",
                    z_slice_index=z_slice_index,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                    subtitle_extra=_mmm_rank_subtitle(rank, "mean"),
                    show_cosine_similarity=show_cosine_similarity,
                )
                n_figs += 1
            print(f"  {sp}: wrote {len(picked)} min/median/max plot(s)")
        else:
            raise ValueError(f"Unknown selection: {selection!r}")

    if no_show:
        plt.close("all")
    print(f"Done: {n_figs} figures under {save_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize UniGrad HCP error-map NPZ (random or min/median/max from Phase-I CSV)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/error-map/unigrad-synth/hcp"),
        help="Error-map root with Train/Val/Test/*.npz.",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        choices=list(FULL_SPLITS),
        help="Optional single split; default = all present splits.",
    )
    p.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["random", "min_median_max"],
        help=(
            "random (one-per-class per split) or min_median_max "
            "(reuse Phase-I mean ‖u‖ selection CSV)."
        ),
    )
    p.add_argument(
        "--mmm-selection-csv",
        type=Path,
        default=DEFAULT_MMM_CSV,
        help=(
            "Phase-I min_median_max_selection.csv (mean ‖u‖ picks). "
            f"Default: {DEFAULT_MMM_CSV}"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--z-slice",
        type=int,
        default=None,
        dest="z_slice_index",
        help="Axial slice index (default mid-volume).",
    )
    p.add_argument(
        "--montage-z-step",
        type=int,
        default=10,
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
        "--save-dir",
        type=Path,
        default=Path("assets/images/error-map/unigrad-synth/hcp/fullrun_random_orthogonal"),
        help="Output directory (creates {split}/*.png).",
    )
    p.add_argument(
        "--cosine-similarity",
        action="store_true",
        help=(
            "Append cosine similarity map cosθ(u_gt, u_pred) after the error map "
            "(+1 aligned, −1 opposite; NaN where either ‖u‖≈0)."
        ),
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    if not is_full_cohort_dir(args.data_dir):
        print(
            f"ERROR: expected Train/Val/Test under {args.data_dir}",
            file=sys.stderr,
        )
        return 2
    visualize_full_cohort(
        input_dir=args.data_dir,
        split=args.split,
        save_dir=args.save_dir,
        no_show=args.no_show,
        selection=args.selection,
        seed=args.seed,
        z_slice_index=args.z_slice_index,
        montage_z_step=args.montage_z_step,
        run_view=args.run_view,
        mmm_selection_csv=args.mmm_selection_csv,
        show_cosine_similarity=args.cosine_similarity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
