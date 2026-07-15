#!/usr/bin/env python3
"""
Visualize HCP synthetic registration NPZ samples (``create_synth_data.py`` output).

Columns: source, source + ``u_gt`` vectors, moving (warped), ``‖u_gt‖`` (+ colorbar);
optional checkerboard (``--checkerboard``). Axial display is radiological (``rot90``).

``u_gt`` is the registration displacement on the **source/fixed** lattice
(``moving(x + u_gt(x)) ≈ source(x)``).

Modes:
  - Dry-run folder (flat ``*.npz``): ``--selection per_class``
  - Full cohort (``Train``/``Val``/``Test``):
      ``--selection random`` (default) → one random sample per class per split
      (up to 15 plots) with orthogonal/montage views
      ``--selection min_median_max`` → min / median / max per split over non-``none``
      classes (4 transforms × 3 splits → 9 plots) by ``--u-metric``; writes
      ``min_median_max_selection.csv`` in ``--save-dir`` to skip re-scoring on reruns

``--run-view orthogonal|montage`` controls row layout (``--montage-z-step`` for montage).

Examples:
# Dry-run figures
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_dryrun --selection per_class --save-dir assets/images/synth-data/torchio/hcp/dryrun_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_dryrun --selection per_class --save-dir assets/images/synth-data/torchio/hcp/dryrun_montage --no-show --run-view montage --montage-z-step 10 --u-contours --checkerboard

# Full cohort: random one-per-class × all splits (15 plots)
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp --selection random --save-dir assets/images/synth-data/torchio/hcp/fullrun_random_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard

# Full cohort: min/median/max per split by mean ‖u‖, excluding none (9 plots)
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp --selection min_median_max --u-metric mean --save-dir assets/images/synth-data/torchio/hcp/fullrun_mmm_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard

# Single split only
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp --split Train --selection random --save-dir assets/images/synth-data/torchio/hcp/fullrun_train_orthogonal --no-show
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
        "u_gt",
        "source_mask",
        "moving_mask",
        "identity_grid_mask",
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
    """Radiological display (CCW 90°): same ``np.rot90`` for every orthogonal plane."""
    return np.rot90(sl)


def orient_axial_u_inplane(u_inplane: np.ndarray) -> np.ndarray:
    """
    Map in-plane ``u`` to Matplotlib quiver ``(U, V)`` after ``orient_axial``.

    ``u_inplane[0]`` / ``u_inplane[1]`` are displacements along the raw
    ``plane_slice`` axes (axis 0 / axis 1) before ``np.rot90``.

    ``np.rot90`` sends voxel ``(i, j)`` → display ``(row, col) = (n1 - 1 - j, i)``.
    A displacement ``(di, dj)`` therefore becomes
    ``(d_row, d_col) = (-dj, di)`` in display data coordinates.

    Quiver: ``U`` = +column (right), ``V`` = +row in data coords. With
    ``imshow(..., origin="upper")`` the y-axis is inverted, so +``V`` points
    down the screen — the same direction as +``d_row``. Hence:

        U = rot90(u0)          # horizontal / rightward
        V = rot90(-u1)         # = -u_vertical after the same rot90 placement
    """
    u0, u1 = u_inplane[0], u_inplane[1]
    uu = np.rot90(u0)
    vv = np.rot90(-u1)
    return np.stack([uu, vv], axis=0)


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
    """
    In-plane displacement ``(2, …)`` matching ``plane_slice`` axis order.

    Channel 0 = offset along the slice's axis 0; channel 1 = along axis 1.
    Pass through ``orient_axial_u_inplane`` before quiver (same ``rot90`` as the image).
    """
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
            "u_gt": np.asarray(data["u_gt"]),
            "source_mask": np.asarray(data["source_mask"]),
            "moving_mask": np.asarray(data["moving_mask"]),
            "identity_grid_mask": np.asarray(data["identity_grid_mask"]),
            "deformation_class": _unpack_scalar_str(data["deformation_class"]),
            "subject_id": _unpack_scalar_str(data["subject_id"]),
        }


def deformation_class_from_filename(npz_path: Path) -> str:
    """Parse class from NPZ stem (full run: ``{subject}_{suf}``; dry-run: ``{subject}_{class}[_NN]``)."""
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


def sample_u_stats(sample: dict) -> dict[str, float]:
    """‖u_gt‖ min/Q1/mean/Q3/max over the full volume (all voxels)."""
    mag = displacement_magnitude(sample["u_gt"].astype(np.float64))
    vals = mag.ravel()
    return {
        "min": float(np.min(vals)),
        "q1": float(np.percentile(vals, 25)),
        "mean": float(np.mean(vals)),
        "q3": float(np.percentile(vals, 75)),
        "max": float(np.max(vals)),
    }


def _format_u_gt_stats_note(stats: dict[str, float]) -> str:
    """Footer line with mathtext subscript (not literal ``u_gt``)."""
    return (
        rf"$\|u_{{\mathrm{{gt}}}}\|$ voxels: min={stats['min']:.2f}  "
        rf"Q1={stats['q1']:.2f}  mean={stats['mean']:.2f}  "
        rf"Q3={stats['q3']:.2f}  max={stats['max']:.2f}"
    )


def scalar_u_score(u: np.ndarray, metric: str) -> float:
    """Scalar ‖u_gt‖ score over the full volume (for min/median/max selection)."""
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
            (fp, scalar_u_score(sample["u_gt"], u_metric))
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
        cls = deformation_class_from_filename(fp)
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
    if missing:
        raise FileNotFoundError(
            f"per_class missing class(es): {', '.join(missing)}. "
            f"Run: create_synth_data.py --dry-run 5"
        )
    return groups


MMM_SELECTION_CSV = "min_median_max_selection.csv"
MMM_RANK_ORDER = ("min", "median", "max")
MMM_SELECTION_FIELDS = (
    "split",
    "rank",
    "u_metric",
    "subject_id",
    "file",
    "deformation_class",
    "u_score",
)


def _resolve_split_npz(input_dir: Path, split: str, file_name: str) -> Path | None:
    fp = input_dir / split / file_name
    if fp.is_file():
        return fp
    fp = resolve_npz_dir(input_dir, split) / file_name
    return fp if fp.is_file() else None


def _subject_id_from_npz_path(fp: Path) -> str:
    stem = fp.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def _mmm_pick_row(
    *,
    split: str,
    rank: str,
    u_metric: str,
    fp: Path,
    score: float,
) -> dict[str, str | float]:
    return {
        "split": split,
        "rank": rank,
        "u_metric": u_metric,
        "subject_id": _subject_id_from_npz_path(fp),
        "file": fp.name,
        "deformation_class": deformation_class_from_filename(fp),
        "u_score": score,
    }


def try_load_mmm_split_cache(
    save_dir: Path,
    input_dir: Path,
    split: str,
    u_metric: str,
) -> list[tuple[Path, str, float]] | None:
    csv_path = save_dir / MMM_SELECTION_CSV
    if not csv_path.is_file():
        return None
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("u_metric") == u_metric and row.get("split") == split:
                rows.append(row)
    if not rows:
        return None
    rank_index = {rank: i for i, rank in enumerate(MMM_RANK_ORDER)}
    rows.sort(key=lambda r: rank_index.get(r["rank"], 99))
    picked: list[tuple[Path, str, float]] = []
    for row in rows:
        fp = _resolve_split_npz(input_dir, split, row["file"])
        if fp is None:
            return None
        picked.append((fp, row["rank"], float(row["u_score"])))
    return picked


def save_mmm_selection_csv(
    save_dir: Path,
    new_rows: list[dict[str, str | float]],
    u_metric: str,
    splits_updated: set[str],
) -> Path:
    csv_path = save_dir / MMM_SELECTION_CSV
    kept: list[dict[str, str]] = []
    if csv_path.is_file():
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("u_metric") == u_metric and row.get("split") in splits_updated:
                    continue
                kept.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MMM_SELECTION_FIELDS)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in MMM_SELECTION_FIELDS})
        for row in new_rows:
            writer.writerow({k: row[k] for k in MMM_SELECTION_FIELDS})
    print(f"Wrote {csv_path} (min/median/max picks for u {u_metric})")
    return csv_path


def select_min_median_max_full_cohort_split(
    files: list[Path],
    u_metric: str,
) -> list[tuple[Path, str, float]]:
    """Min / median / max by ``u_metric`` over a split, excluding ``none`` class."""
    eligible = [
        fp for fp in files if deformation_class_from_filename(fp) != "none"
    ]
    return select_min_median_max_files(eligible, u_metric)


def _mmm_rank_subtitle(rank: str, u_metric: str) -> str:
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
    return f"HCP Synthetic Data Plot ({label} Transformation)"


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
    x0 = slot.x0 + 0.05 * slot.width
    cax = fig.add_axes([x0, y0, bar_w, h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=_LABEL - 1, labelpad=8, rotation=90)
    cbar.ax.tick_params(labelsize=_LABEL - 2, pad=2)
    cbar.ax.yaxis.set_label_coords(3.6, 0.5)


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
    sample_cache: dict[Path, dict] | None = None,
) -> None:
    """
    Column layout (left → right):

      Source | Source + u_gt vectors | Warped (moving) | ‖u_gt‖ | [‖u‖ cbar]
      [| Checkerboard]
    """
    from matplotlib.gridspec import GridSpec

    nrows = len(picked)
    # Image cols: 0 source, 1 vectors-on-source, 2 moving, 3 ‖u‖; 4 reserved cbar.
    if use_checkerboard:
        n_grid_cols = 6
        img_cols = (0, 1, 2, 3, 5)
        width_ratios = [1.0, 1.0, 1.0, 1.0, 0.32, 1.0]
        n_image_panels = 5
    else:
        n_grid_cols = 5
        img_cols = (0, 1, 2, 3)
        width_ratios = [1.0, 1.0, 1.0, 1.0, 0.32]
        n_image_panels = 4

    col_titles = {
        0: "Source (fixed)",
        1: r"Source + $u_{\mathrm{gt}}$ vectors",
        2: "Warped (moving)",
        3: r"$\|u_{\mathrm{gt}}\|$",
    }
    if use_checkerboard:
        col_titles[5] = "Checkerboard"

    cache: dict[Path, dict] = {} if sample_cache is None else sample_cache
    row_u_vmax: list[float] = []
    for row, (fp, _, _) in enumerate(picked):
        sample = _cached_sample(fp, cache)
        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(sample["source"], plane, z_slice_index)[1]
        if plane == "axial":
            u_sl = sample["u_gt"][:, :, :, idx]
        elif plane == "coronal":
            u_sl = sample["u_gt"][:, :, idx, :]
        else:
            u_sl = sample["u_gt"][:, idx, :, :]
        mag = displacement_magnitude(u_sl.astype(np.float64)).ravel()
        row_u_vmax.append(max(float(np.percentile(mag, _U_COLOR_PERCENTILE)), 1e-6))

    plt.rcParams.update({"font.family": _FONT, "figure.dpi": _DPI, "savefig.dpi": _DPI})
    fig_w = 3.0 * n_image_panels + 2.4
    fig_h = row_h * nrows + 1.8
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
        wspace=0.30,
        hspace=0.34,
    )

    axes: dict[tuple[int, int], plt.Axes] = {}
    for row in range(nrows):
        for col in img_cols:
            axes[(row, col)] = fig.add_subplot(gs[row, col])

    im_u_last = None
    for row, (file_path, _rank_label, _) in enumerate(picked):
        sample = _cached_sample(file_path, cache)
        source = sample["source"]
        moving = sample["moving"]
        u = sample["u_gt"]

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

        ax_src = axes[(row, 0)]
        ax_src.imshow(src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        _style_axis(ax_src)
        ax_src.set_ylabel(
            _plane_row_label(plane, idx),
            fontsize=_LABEL,
            rotation=90,
            ha="center",
            va="center",
            labelpad=18,
        )

        # u_gt lives on the source lattice → quiver over source.
        # uu/vv are already in display (col, row) coords from orient_axial_u_inplane.
        ax_q = axes[(row, 1)]
        ax_q.imshow(src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
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

        ax_mov = axes[(row, 2)]
        ax_mov.imshow(mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        _style_axis(ax_mov)

        ax_u = axes[(row, 3)]
        im_u_last = ax_u.imshow(
            u_mag_sl, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        if use_u_contours:
            levels = np.linspace(0.15 * u_vmax, 0.95 * u_vmax, 6)
            ax_u.contour(u_mag_sl, levels=levels, colors="white", linewidths=0.5, alpha=0.7)
        _style_axis(ax_u)

        if use_checkerboard:
            ax_c = axes[(row, 5)]
            cb = checkerboard_mix(src_disp, mov_disp)
            ax_c.imshow(cb, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
            _style_axis(ax_c)

        if row == 0:
            for col, t in col_titles.items():
                axes[(0, col)].set_title(t, fontsize=_SUBTITLE, fontweight="medium", pad=10)

    # Center colorbar on the middle row; height = 1.5 × that panel.
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
            label=r"$\|u_{\mathrm{gt}}\|$ (voxels)",
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


def _views_for_sample(
    sample: dict,
    *,
    z_slice_index: int | None,
    montage_z_step: int,
    run_view: str,
) -> tuple[list[str], list[int], list[str]]:
    if run_view == "orthogonal":
        x0 = sample["source"].shape[0] // 2
        y0 = sample["source"].shape[1] // 2
        z0 = axial_slice(sample["source"], z_slice_index)[1]
        return (
            ["axial", "coronal", "sagittal"],
            [z0, y0, x0],
            ["axial", "coronal", "sagittal"],
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
    )


def _render_single_sample_plot(
    fp: Path,
    *,
    save_path: Path,
    z_slice_index: int | None,
    quiver_stride: int,
    use_u_contours: bool,
    use_checkerboard: bool,
    montage_z_step: int,
    run_view: str,
    subtitle_extra: str | None = None,
    include_plot_u_stats: bool = True,
) -> dict:
    sample = load_sample(fp)
    stats_note = None
    u_stats: dict[str, float] = {}
    if include_plot_u_stats:
        print(f"    Computing ‖u_gt‖ stats for plot: {fp.name}")
        u_stats = sample_u_stats(sample)
        stats_note = _format_u_gt_stats_note(u_stats)
    plane_rows, idx_rows, rank_rows = _views_for_sample(
        sample,
        z_slice_index=z_slice_index,
        montage_z_step=montage_z_step,
        run_view=run_view,
    )
    subject_id = sample.get("subject_id") or fp.stem.split("_")[0]
    deform_cls = str(sample.get("deformation_class") or "unknown")
    picked = [(fp, r, float("nan")) for r in rank_rows]
    print(f"    Plotting → {save_path}")
    _render_figure(
        picked,
        save_path=save_path,
        no_show=True,
        title=_class_plot_title(deform_cls),
        subtitle=_plot_subtitle(str(subject_id), run_view, subtitle_extra),
        z_slice_index=z_slice_index,
        quiver_stride=quiver_stride,
        row_planes=plane_rows,
        row_slice_indices=idx_rows,
        use_u_contours=use_u_contours,
        use_checkerboard=use_checkerboard,
        sample_stats_note=stats_note,
        row_h=3.0,
        announce_save=False,
        sample_cache={fp: sample},
    )
    return {
        "file": fp.name,
        "subject_id": sample.get("subject_id"),
        "deformation_class": sample.get("deformation_class"),
        **u_stats,
    }


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
    mmm_cache_rows: list[dict[str, str | float]] = []
    mmm_splits_updated: set[str] = set()
    print(
        f"Full cohort viz: selection={selection}  splits={', '.join(splits)}  "
        f"view={run_view}  → {save_dir}"
    )

    if selection == "min_median_max" and (save_dir / MMM_SELECTION_CSV).is_file():
        print(f"Found {MMM_SELECTION_CSV}; will reuse picks for u {u_metric} when valid")

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
            groups, missing = group_class_examples_optional(files, seed=seed + hash(sp) % 10007)
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
                    quiver_stride=quiver_stride,
                    use_u_contours=use_u_contours,
                    use_checkerboard=use_checkerboard,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                )
                n_figs += 1
            print(f"  {sp}: wrote {len(groups)} class plot(s)")
        elif selection == "min_median_max":
            picked = try_load_mmm_split_cache(save_dir, input_dir, sp, u_metric)
            if picked is not None:
                print(f"  {sp}: using cached min/median/max selection ({len(picked)} picks)")
            else:
                eligible = [
                    fp for fp in files if deformation_class_from_filename(fp) != "none"
                ]
                print(
                    f"  {sp}: scoring {u_metric} ‖u_gt‖ on {len(eligible)} NPZs "
                    f"(excluding none; {len(files) - len(eligible)} skipped)…"
                )
                picked = select_min_median_max_full_cohort_split(files, u_metric)
                mmm_splits_updated.add(sp)
                for fp, rank, score in picked:
                    mmm_cache_rows.append(
                        _mmm_pick_row(
                            split=sp,
                            rank=rank,
                            u_metric=u_metric,
                            fp=fp,
                            score=score,
                        )
                    )
            for fp, rank, score in picked:
                print(
                    f"  {sp} / {rank}: {u_metric} ‖u_gt‖={score:.3f} from {fp.name}"
                )
                _render_single_sample_plot(
                    fp,
                    save_path=out_split / f"{rank}.png",
                    z_slice_index=z_slice_index,
                    quiver_stride=quiver_stride,
                    use_u_contours=use_u_contours,
                    use_checkerboard=use_checkerboard,
                    montage_z_step=montage_z_step,
                    run_view=run_view,
                    subtitle_extra=_mmm_rank_subtitle(rank, u_metric),
                )
                n_figs += 1
            print(
                f"  {sp}: wrote {len(picked)} min/median/max plot(s) "
                f"by {u_metric} ‖u_gt‖"
            )
        else:
            raise ValueError(f"Full cohort does not support selection={selection!r}")

    if selection == "min_median_max" and mmm_cache_rows:
        save_mmm_selection_csv(save_dir, mmm_cache_rows, u_metric, mmm_splits_updated)

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
        print(f"  {label}: loading {fp.name}")
        sample = load_sample(fp)
        print(f"  {label}: computing ‖u_gt‖ stats for plot")
        u_stats = sample_u_stats(sample)
        chosen_stats.append(
            {
                "class": label,
                "file": fp.name,
                "subject_id": sample.get("subject_id"),
                **u_stats,
            }
        )
        stats_note = _format_u_gt_stats_note(u_stats)
        if run_view == "orthogonal":
            x0 = sample["source"].shape[0] // 2
            y0 = sample["source"].shape[1] // 2
            z0 = axial_slice(sample["source"], z_slice_index)[1]
            plane_rows = ["axial", "coronal", "sagittal"]
            idx_rows = [z0, y0, x0]
            rank_rows = ["axial", "coronal", "sagittal"]
        else:
            z0 = axial_slice(sample["source"], z_slice_index)[1]
            z_offsets = [-int(montage_z_step), 0, int(montage_z_step)]
            plane_rows = ["axial", "axial", "axial"]
            idx_rows = [max(0, min(sample["source"].shape[2] - 1, z0 + dz)) for dz in z_offsets]
            rank_rows = [f"z{dz:+d}" for dz in z_offsets]
        subject_id = sample.get("subject_id") or fp.stem.split("_")[0]
        picked = [(fp, r, float("nan")) for r in rank_rows]
        print(f"  {label}: plotting → {save_dir / f'{label}.png'}")
        _render_figure(
            picked,
            save_path=save_dir / f"{label}.png",
            no_show=True,
            title=_class_plot_title(label),
            subtitle=_plot_subtitle(str(subject_id), run_view),
            z_slice_index=z_slice_index,
            quiver_stride=quiver_stride,
            row_planes=plane_rows,
            row_slice_indices=idx_rows,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            sample_stats_note=stats_note,
            row_h=3.0,
            announce_save=False,
            sample_cache={fp: sample},
        )
    csv_path = save_dir / "chosen_sample_u_stats.csv"
    fieldnames = ["class", "file", "subject_id", "min", "q1", "mean", "q3", "max"]
    print(f"Writing chosen-sample ‖u_gt‖ stats → {csv_path}")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in chosen_stats:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {csv_path} (‖u_gt‖ stats for plotted samples only)")
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
        examples_note = f"min / median / max ({u_metric} ‖u_gt‖)"
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    split_label = split or "all"
    _render_figure(
        picked,
        save_path=save_path,
        no_show=no_show,
        title="HCP Synthetic Data Plot",
        subtitle=(
            f"{examples_note} - {split_label} split - Radiological-style display"
        ),
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
            "min_median_max (min/median/max per split over non-none classes, 9 plots)."
        ),
    )
    p.add_argument(
        "--u-metric",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Scalar per volume for min/median/max selection (‖u_gt‖).",
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
        help="Overlay contour lines on ‖u_gt‖ magnitude map.",
    )
    p.add_argument(
        "--checkerboard",
        action="store_true",
        help=(
            "Add optional checkerboard column after ‖u_gt‖ / colorbar: "
            "alternating tiles of source and moving "
            "(highlights local mismatch at tile edges)."
        ),
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for per_class (dry-run) or full-cohort plots "
            "(split subfolders)."
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
