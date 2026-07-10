#!/usr/bin/env python3
"""
Visualize HCP synthetic registration NPZ samples (``create_synth_data.py`` output).

Columns: source (fixed) + grid, warped/moving + grid, ``‖u‖``, sparse quiver;
optional checkerboard column (``--checkerboard``). Axial display is radiological
(``rot90``, posterior up).

Examples:
python experiments/unigrad-synth/visualize_synth_data.py --data-dir datasets/hcp_synth_dryrun --selection range_grid --save-dir assets/images/unigrad-synth/hcp/range_grid_report --no-show --range-grid-view orthogonal --u-contours --checkerboard
python experiments/unigrad-synth/visualize_synth_data.py --data-dir datasets/hcp_synth --split Train --selection min_median_max --u-metric mean --save-path assets/images/unigrad-synth/hcp/hcp_synth_minmedmax.png --no-show
"""

from __future__ import annotations

import argparse
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
        "source_grid",
        "moving_grid",
        "identity_grid_mask",
        "deformation_class",
        "magnitude_range",
        "subject_id",
        "qc_passed",
    }
)

DEFORMATION_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("rigid", "rig"),
    ("affine", "aff"),
    ("elastic", "ela"),
)

RANGE_GRID_COMBINATIONS: tuple[tuple[str, str], ...] = (
    ("none", "none"),
    ("rigid", "low"),
    ("rigid", "mid"),
    ("rigid", "high"),
    ("affine", "low"),
    ("affine", "mid"),
    ("affine", "high"),
    ("elastic", "low"),
    ("elastic", "mid"),
    ("elastic", "high"),
    ("affine_elastic", "low"),
    ("affine_elastic", "mid"),
    ("affine_elastic", "high"),
)

_GRID_COLOR = "cyan"
_CHECKER_TILE = 16
_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9
_QUIVER_COLOR = "lime"


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def _unpack_scalar_str(raw) -> str:
    a = np.asarray(raw)
    if a.size == 0:
        return ""
    return str(a.reshape(-1)[0])


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


def resolve_npz_dir(input_dir: Path, split: str | None) -> Path:
    """Use split subfolder when present; otherwise flat layout (e.g. range-grid dry run)."""
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


def overlay_regular_grid(
    ax: plt.Axes,
    height: int,
    width: int,
    *,
    stride: int = 12,
    color: str = _GRID_COLOR,
    linewidth: float = 0.55,
    alpha: float = 0.9,
) -> None:
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    levels_col = np.arange(0, width + stride, stride)
    levels_row = np.arange(0, height + stride, stride)
    ax.contour(grid_col, levels=levels_col, colors=color, linewidths=linewidth, alpha=alpha)
    ax.contour(grid_row, levels=levels_row, colors=color, linewidths=linewidth, alpha=alpha)


def overlay_deformation_grid(
    ax: plt.Axes,
    u_inplane: np.ndarray,
    *,
    stride: int = 12,
    color: str = _GRID_COLOR,
    linewidth: float = 0.55,
    alpha: float = 0.85,
) -> None:
    if u_inplane.ndim != 3 or u_inplane.shape[0] != 2:
        raise ValueError(f"Expected u_inplane (2, H, W), got {u_inplane.shape}")
    _, h, w = u_inplane.shape
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    pos_col = grid_col + u_inplane[0]
    pos_row = grid_row + u_inplane[1]
    levels_col = np.arange(0, w + stride, stride)
    levels_row = np.arange(0, h + stride, stride)
    ax.contour(pos_col, levels=levels_col, colors=color, linewidths=linewidth, alpha=alpha)
    ax.contour(pos_row, levels=levels_row, colors=color, linewidths=linewidth, alpha=alpha)


def overlay_binary_grid(
    ax: plt.Axes,
    grid_mask_2d: np.ndarray,
    *,
    color: str = _GRID_COLOR,
    alpha: float = 0.75,
) -> None:
    mask = grid_mask_2d > 0.5
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.array(plt.matplotlib.colors.to_rgb(color), dtype=np.float32)
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba, origin="upper", interpolation="nearest")


def checkerboard_mix(a: np.ndarray, b: np.ndarray, tile: int = _CHECKER_TILE) -> np.ndarray:
    h, w = a.shape
    yy, xx = np.indices((h, w))
    use_a = ((yy // tile) + (xx // tile)) % 2 == 0
    return np.where(use_a, a, b)


def load_sample(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = HCP_REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        qc_val, qc_err = _unpack_qc_passed(data["qc_passed"])
        if qc_err:
            raise ValueError(f"{npz_path.name}: {qc_err}")
        return {
            "source": np.asarray(data["source"]),
            "moving": np.asarray(data["moving"]),
            "u": np.asarray(data["u"]),
            "source_mask": np.asarray(data["source_mask"]),
            "moving_mask": np.asarray(data["moving_mask"]),
            "source_grid": np.asarray(data["source_grid"]),
            "moving_grid": np.asarray(data["moving_grid"]),
            "identity_grid_mask": np.asarray(data["identity_grid_mask"]),
            "qc_passed": qc_val,
            "deformation_class": _unpack_scalar_str(data["deformation_class"]),
            "magnitude_range": _unpack_scalar_str(data["magnitude_range"]),
            "subject_id": _unpack_scalar_str(data["subject_id"]),
        }


def prefer_qc_passed_files(files: list[Path]) -> tuple[list[Path], bool]:
    passed = [fp for fp in files if load_sample(fp).get("qc_passed") is True]
    if passed:
        return passed, False
    return files, True


def scalar_u_score(u: np.ndarray, mask: np.ndarray | None, metric: str) -> float:
    mag = displacement_magnitude(u.astype(np.float64))
    if mask is not None and np.any(mask > 0.5):
        vals = mag[mask > 0.5]
    else:
        vals = mag.ravel()
    if metric == "mean":
        return float(np.mean(vals))
    if metric == "max":
        return float(np.max(vals))
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
            (fp, scalar_u_score(sample["u"], sample.get("source_mask"), u_metric))
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
    pool_files, used_fallback = prefer_qc_passed_files(files)
    if used_fallback:
        print(
            "Warning: no qc_passed=True samples in split; using all files for row selection.",
            file=sys.stderr,
        )
    pools: dict[str, list[Path]] = {cls: [] for cls, _ in DEFORMATION_ROW_ORDER}
    for fp in pool_files:
        cls = load_sample(fp).get("deformation_class")
        if cls in pools:
            pools[str(cls)].append(fp)
    for cls, suf in DEFORMATION_ROW_ORDER:
        if not pools[cls]:
            pools[cls] = [fp for fp in pool_files if fp.stem.endswith(f"_{suf}")]

    rng = random.Random(seed)
    picked: list[tuple[Path, str, float]] = []
    for cls, suf in DEFORMATION_ROW_ORDER:
        pool = pools[cls]
        if not pool:
            raise FileNotFoundError(
                f"No '{cls}' (*_{suf}.npz) sample found in split ({len(files)} files)."
            )
        picked.append((rng.choice(pool), cls, float("nan")))
    return picked


def _pool_range_grid_files(files: list[Path]) -> dict[tuple[str, str], list[Path]]:
    by_key: dict[tuple[str, str], list[Path]] = {}
    for fp in files:
        meta = load_sample(fp)
        cls = meta.get("deformation_class")
        mag_range = meta.get("magnitude_range")
        if cls and mag_range:
            by_key.setdefault((str(cls), str(mag_range)), []).append(fp)
    for cls, mag_range in RANGE_GRID_COMBINATIONS:
        key = (cls, mag_range)
        if key in by_key:
            continue
        needle = f"_{cls}_{mag_range}"
        matches = [fp for fp in files if needle in fp.stem]
        if matches:
            by_key[key] = matches
    return by_key


def group_range_grid_examples(files: list[Path]) -> list[tuple[str, Path]]:
    pool_files, used_fallback = prefer_qc_passed_files(files)
    if used_fallback:
        print(
            "Warning: no qc_passed=True samples in split; using all files for range_grid.",
            file=sys.stderr,
        )
    pools = _pool_range_grid_files(pool_files)
    groups: list[tuple[str, Path]] = []
    missing: list[str] = []
    for cls, mag_range in RANGE_GRID_COMBINATIONS:
        label = f"{cls}_{mag_range}"
        pool = sorted(pools.get((cls, mag_range), []), key=lambda p: p.name)
        if not pool:
            missing.append(label)
            continue
        groups.append((label, pool[0]))
    if missing:
        raise FileNotFoundError(
            f"range_grid missing combination(s): {', '.join(missing)}. "
            f"Run: create_synth_data.py --range-grid"
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
    meta: list[str] = []
    if extra.get("deformation_class"):
        meta.append(str(extra["deformation_class"]))
    if extra.get("magnitude_range"):
        meta.append(str(extra["magnitude_range"]))
    if "qc_passed" in extra:
        meta.append(f"qc={extra['qc_passed']}")
    if meta:
        lines.append(" · ".join(meta))
    return "\n".join(lines)


def _render_figure(
    picked: list[tuple[Path, str, float]],
    *,
    save_path: Path | None,
    no_show: bool,
    title: str,
    subtitle: str,
    slice_index: int | None,
    grid_stride: int,
    u_percentile: float,
    per_row_u_scale: bool,
    row_planes: list[str] | None = None,
    row_slice_indices: list[int] | None = None,
    use_u_contours: bool = False,
    use_checkerboard: bool = False,
    row_h: float = 3.2,
) -> None:
    nrows = len(picked)
    ncols = 5 if use_checkerboard else 4
    col_titles = ["Source (fixed)", "Warped (moving)", r"$\|u\|$", "u vectors"]
    if use_checkerboard:
        col_titles.append("checkerboard")

    plane_idx_notes: list[str] = []
    row_u_vmax: list[float] = []
    global_u_mags: list[np.ndarray] = []
    for row, (fp, _, _) in enumerate(picked):
        sample = load_sample(fp)
        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(sample["source"], plane, slice_index)[1]
        plane_idx_notes.append(f"{plane[0]}={idx}")
        if plane == "axial":
            u_sl = sample["u"][:, :, :, idx]
        elif plane == "coronal":
            u_sl = sample["u"][:, :, idx, :]
        else:
            u_sl = sample["u"][:, idx, :, :]
        mag = displacement_magnitude(u_sl.astype(np.float64)).ravel()
        global_u_mags.append(mag)
        row_u_vmax.append(max(float(np.percentile(mag, u_percentile)), 1e-6))
    u_vmax_global = float(np.percentile(np.concatenate(global_u_mags), u_percentile))
    u_vmax_global = max(u_vmax_global, 1e-6)

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
            for k in ("qc_passed", "deformation_class", "magnitude_range", "subject_id")
            if k in sample
        }

        plane = row_planes[row] if row_planes is not None else "axial"
        if row_slice_indices is not None:
            idx = int(row_slice_indices[row])
        else:
            idx = plane_slice(source, plane, slice_index)[1]
        src_sl, _ = plane_slice(source, plane, idx)
        mov_sl, _ = plane_slice(moving, plane, idx)
        if plane == "axial":
            src_grid_raw = sample["source_grid"][:, :, idx]
            mov_grid_raw = sample["moving_grid"][:, :, idx]
            u_mag_raw = displacement_magnitude(u[:, :, :, idx].astype(np.float64))
        elif plane == "coronal":
            src_grid_raw = sample["source_grid"][:, idx, :]
            mov_grid_raw = sample["moving_grid"][:, idx, :]
            u_mag_raw = displacement_magnitude(u[:, :, idx, :].astype(np.float64))
        else:
            src_grid_raw = sample["source_grid"][idx, :, :]
            mov_grid_raw = sample["moving_grid"][idx, :, :]
            u_mag_raw = displacement_magnitude(u[:, idx, :, :].astype(np.float64))
        src_grid_sl = orient_axial(src_grid_raw)
        mov_grid_sl = orient_axial(mov_grid_raw)
        u_inplane = orient_axial_u_inplane(plane_u_inplane_slice(u, plane, idx))
        u_mag_sl = orient_axial(u_mag_raw)
        src_disp = orient_axial(src_sl)
        mov_disp = orient_axial(mov_sl)
        h, w = src_disp.shape
        u_vmax = row_u_vmax[row] if per_row_u_scale else u_vmax_global

        subject_id = extra.get("subject_id") or file_path.stem.split("_")[0]
        plane_tag = rank_label if rank_label else plane
        row_title = f"{subject_id} · {plane_tag}"
        deformation_class = str(extra.get("deformation_class") or "")
        use_2d_grid_overlay = deformation_class in {"rigid", "affine", "affine_elastic"}

        ax_src = axes[row, 0]
        ax_src.imshow(src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        if use_2d_grid_overlay:
            overlay_regular_grid(ax_src, h, w, stride=grid_stride, color=_GRID_COLOR, alpha=0.75)
        else:
            overlay_binary_grid(ax_src, src_grid_sl, color=_GRID_COLOR)
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
        if use_2d_grid_overlay:
            overlay_deformation_grid(
                ax_mov, u_inplane, stride=grid_stride, color=_GRID_COLOR, linewidth=0.6, alpha=0.72
            )
        else:
            overlay_binary_grid(ax_mov, mov_grid_sl, color=_GRID_COLOR)
        _style_axis(ax_mov)

        ax_u = axes[row, 2]
        im_u_last = ax_u.imshow(
            u_mag_sl, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        if use_u_contours:
            levels = np.linspace(0.15 * u_vmax, 0.95 * u_vmax, 6)
            ax_u.contour(u_mag_sl, levels=levels, colors="white", linewidths=0.5, alpha=0.7)
        _style_axis(ax_u)

        ax_q = axes[row, 3]
        ax_q.imshow(mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        step = max(14, grid_stride + 6)
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
        cbar.set_label(r"Displacement norm $\|u\|$ (voxels)", fontsize=_LABEL)
        cbar.ax.tick_params(labelsize=_LABEL - 1)

    fig.suptitle(title, fontsize=_TITLE, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, full_subtitle, ha="center", va="top", fontsize=_LABEL, color="black")
    fig.subplots_adjust(left=0.24, right=0.90, top=0.86, bottom=0.08, wspace=0.26, hspace=0.34)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def visualize_range_grid_combinations(
    input_dir: Path,
    split: str | None,
    save_dir: Path,
    no_show: bool,
    *,
    slice_index: int | None,
    grid_stride: int,
    u_percentile: float,
    per_row_u_scale: bool,
    use_u_contours: bool,
    use_checkerboard: bool,
    montage_z_step: int,
    range_grid_view: str,
) -> None:
    files = collect_npz_files(input_dir, split)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{resolve_npz_dir(input_dir, split)}'.")
    groups = group_range_grid_examples(files)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for label, fp in groups:
        sample = load_sample(fp)
        if range_grid_view == "orthogonal":
            x0 = sample["source"].shape[0] // 2
            y0 = sample["source"].shape[1] // 2
            z0 = axial_slice(sample["source"], slice_index)[1]
            plane_rows = ["axial", "coronal", "sagittal"]
            idx_rows = [z0, y0, x0]
            rank_rows = ["axial", "coronal", "sagittal"]
            subtitle = "Orthogonal sanity check · Radiological-style display"
        else:
            z0 = axial_slice(sample["source"], slice_index)[1]
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
            slice_index=slice_index,
            grid_stride=grid_stride,
            u_percentile=u_percentile,
            per_row_u_scale=per_row_u_scale,
            row_planes=plane_rows,
            row_slice_indices=idx_rows,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            row_h=3.0,
        )
    if no_show:
        plt.close("all")
    print(f"Saved {len(groups)} figures under {save_dir}")


def visualize_samples(
    input_dir: Path,
    split: str | None,
    save_path: Path | None,
    no_show: bool,
    *,
    selection: str,
    u_metric: str,
    seed: int,
    slice_index: int | None,
    grid_stride: int,
    u_percentile: float,
    per_row_u_scale: bool,
    save_dir: Path | None = None,
    use_u_contours: bool = False,
    use_checkerboard: bool = False,
    montage_z_step: int = 10,
    range_grid_view: str = "orthogonal",
) -> None:
    if selection == "range_grid":
        if save_dir is None:
            if save_path is not None:
                save_dir = Path(save_path).parent / Path(save_path).stem
            else:
                save_dir = Path("assets/images/unigrad-synth/hcp/range_grid")
        visualize_range_grid_combinations(
            input_dir,
            None,
            save_dir,
            no_show,
            slice_index=slice_index,
            grid_stride=grid_stride,
            u_percentile=u_percentile,
            per_row_u_scale=per_row_u_scale,
            use_u_contours=use_u_contours,
            use_checkerboard=use_checkerboard,
            montage_z_step=montage_z_step,
            range_grid_view=range_grid_view,
        )
        return

    files = collect_npz_files(input_dir, split)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{resolve_npz_dir(input_dir, split)}'.")

    if selection == "random":
        picked = select_deformation_class_examples(files, seed)
        examples_note = f"Rigid / affine / elastic examples (seed = {seed})"
    elif selection == "min_median_max":
        pool_files, used_fallback = prefer_qc_passed_files(files)
        if used_fallback:
            print(
                "Warning: no qc_passed=True samples in split; using all files for min/median/max.",
                file=sys.stderr,
            )
        picked = select_min_median_max_files(pool_files, u_metric)
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
        slice_index=slice_index,
        grid_stride=grid_stride,
        u_percentile=u_percentile,
        per_row_u_scale=per_row_u_scale,
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
        default=Path("datasets/hcp_synth"),
        help="Root with Train/Val/Test subfolders, or flat dry-run output.",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Train/Val/Test subfolder (default Train; ignored for range_grid dry runs).",
    )
    p.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["min_median_max", "random", "range_grid"],
        help="Sample pick: random, min/median/max, or 13 PNGs (one per class×range combo).",
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
        "--slice-index",
        type=int,
        default=None,
        help="Axial slice index (default: mid z).",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=20,
        help="Grid line spacing in voxels.",
    )
    p.add_argument(
        "--u-percentile",
        type=float,
        default=99.0,
        help="Color scale cap for ‖u‖ (percentile; per-row unless --global-u-scale).",
    )
    p.add_argument(
        "--u-contours",
        action="store_true",
        help="Overlay contour lines on ‖u‖ magnitude map.",
    )
    p.add_argument(
        "--checkerboard",
        action="store_true",
        help="Add checkerboard source/moving comparison column (16 px tiles).",
    )
    p.add_argument(
        "--global-u-scale",
        action="store_true",
        help="Use one shared ‖u‖ color scale across rows (default: per-row scaling).",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output directory for --selection range_grid (one PNG per combination).",
    )
    p.add_argument(
        "--range-grid-z-step",
        type=int,
        default=10,
        metavar="VOX",
        help="Axial offset for montage view (range_grid only).",
    )
    p.add_argument(
        "--range-grid-view",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "montage"],
        help="range_grid rows: orthogonal planes or 3-slice axial montage.",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=Path("assets/images/unigrad-synth/hcp/hcp_synth_preview.png"),
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    split = args.split if args.selection != "range_grid" else (args.split or None)
    if split is None and args.selection != "range_grid":
        split = "Train"

    visualize_samples(
        data_dir,
        split,
        args.save_path,
        args.no_show,
        selection=args.selection,
        u_metric=args.u_metric,
        seed=args.seed,
        slice_index=args.slice_index,
        grid_stride=args.grid_stride,
        u_percentile=args.u_percentile,
        per_row_u_scale=not args.global_u_scale,
        save_dir=args.save_dir,
        use_u_contours=args.u_contours,
        use_checkerboard=args.checkerboard,
        montage_z_step=args.range_grid_z_step,
        range_grid_view=args.range_grid_view,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
