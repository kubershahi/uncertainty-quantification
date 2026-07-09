#!/usr/bin/env python3
"""
Visualize synthetic registration NPZ samples.

Supports two formats (auto-detected or ``--format``):

**HCP volume** (``create_synth_data.py`` output): rows = rigid / affine / elastic examples;
columns = source (fixed) + reference grid, warped/moving + grid bent by ``u``, ``‖u‖``.
Axial slices use radiological display (``rot90``, posterior up).

**IXI 2D legacy** (``*_triplet.npz``): ``image``, ``warped``, ``phi``.

Examples:
python experiments/unigrad-synth/visualize_synth_data.py --data-dir datasets/hcp_synth_dryrun --selection range_grid --save-dir assets/images/unigrad-synth/hcp/range_grid --no-show
python experiments/unigrad-synth/visualize_synth_data.py --data-dir datasets/hcp_synth --split Train --selection min_median_max --save-path assets/images/unigrad-synth/hcp/hcp_synth_minmedmax.png --no-show
python experiments/unigrad-synth/visualize_synth_data.py --data-dir data/IXI_2D_synth_trip --format triplet --split Train --phi-view magnitude --save-path assets/images/synth/ixi_minmedmax.png --no-show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SPLITS = ("Train", "Val", "Test", "Atlas")
TRIPLET_GLOB = "*_triplet.npz"
HCP_SYNTH_GLOB = "*.npz"
TRIPLET_KEYS = frozenset({"image", "warped", "phi"})
HCP3D_KEYS = frozenset({"source", "moving", "u"})
HCP3D_REQUIRED_KEYS = frozenset(
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

# Default rows for ``--selection random`` (HCP volume): one example per class, top to bottom.
DEFORMATION_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("rigid", "rig"),
    ("affine", "aff"),
    ("elastic", "ela"),
)

# All (class, range) pairs for ``--selection range_grid`` (matches create_synth_data.py).
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
_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9
_QUIVER_COLOR = "lime"


def displacement_magnitude(u: np.ndarray) -> np.ndarray:
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


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


def detect_format(npz_path: Path) -> str:
    with np.load(npz_path) as data:
        keys = set(data.files)
    if HCP3D_KEYS.issubset(keys):
        return "hcp3d"
    if TRIPLET_KEYS.issubset(keys):
        return "triplet"
    raise ValueError(
        f"{npz_path.name}: unrecognized NPZ keys {sorted(keys)} "
        f"(expected HCP volume {sorted(HCP3D_KEYS)} or triplet {sorted(TRIPLET_KEYS)})"
    )


def resolve_npz_dir(input_dir: Path, split: str | None) -> Path:
    """Use split subfolder when present; otherwise flat layout (e.g. range-grid dry run)."""
    if split:
        split_dir = input_dir / split
        if split_dir.is_dir():
            return split_dir
    return input_dir


def collect_npz_files(input_dir: Path, split: str | None, pattern: str) -> list[Path]:
    data_dir = resolve_npz_dir(input_dir, split)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob(pattern))
    if not files and split:
        files = sorted(input_dir.glob(pattern))
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


def axial_u_slice(u: np.ndarray, slice_index: int) -> np.ndarray:
    """In-plane displacement ``(2, H, W)`` at axial ``z`` (components 0 and 1)."""
    z = max(0, min(int(slice_index), u.shape[3] - 1))
    return np.stack([u[0, :, :, z], u[1, :, :, z]], axis=0)


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
    """Undeformed reference grid on the fixed slice."""
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
    """Grid lines displaced by in-plane ``u`` (shape ``(2, H, W)``)."""
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
    """Overlay transformed binary grid with transparent background."""
    mask = grid_mask_2d > 0.5
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = np.array(plt.matplotlib.colors.to_rgb(color), dtype=np.float32)
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba, origin="upper", interpolation="nearest")


def checkerboard_mix(a: np.ndarray, b: np.ndarray, tile: int = 16) -> np.ndarray:
    """Alternating-tile mix of two same-shape 2D arrays."""
    h, w = a.shape
    yy, xx = np.indices((h, w))
    use_a = ((yy // tile) + (xx // tile)) % 2 == 0
    return np.where(use_a, a, b)


def load_hcp3d_sample(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = HCP3D_REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        out = {
            "source": np.asarray(data["source"]),
            "moving": np.asarray(data["moving"]),
            "u": np.asarray(data["u"]),
        }
        out["source_mask"] = np.asarray(data["source_mask"])
        out["moving_mask"] = np.asarray(data["moving_mask"])
        out["source_grid"] = np.asarray(data["source_grid"])
        out["moving_grid"] = np.asarray(data["moving_grid"])
        out["identity_grid_mask"] = np.asarray(data["identity_grid_mask"])
        qc_val, qc_err = _unpack_qc_passed(data["qc_passed"])
        if qc_err:
            raise ValueError(f"{npz_path.name}: {qc_err}")
        out["qc_passed"] = qc_val
        out["deformation_class"] = _unpack_scalar_str(data["deformation_class"])
        out["magnitude_range"] = _unpack_scalar_str(data["magnitude_range"])
        out["subject_id"] = _unpack_scalar_str(data["subject_id"])
    return out


def prefer_qc_passed_files(files: list[Path]) -> tuple[list[Path], bool]:
    """Return qc_passed=True files when available; warn via bool if falling back."""
    passed: list[Path] = []
    for fp in files:
        sample = load_hcp3d_sample(fp)
        if sample.get("qc_passed") is True:
            passed.append(fp)
    if passed:
        return passed, False
    return files, True


def load_triplet(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    extra: dict = {}
    with np.load(npz_path) as data:
        missing = TRIPLET_KEYS - set(data.files)
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


def scalar_phi_score(phi: np.ndarray, metric: str) -> float:
    mag = phi_magnitude(phi.astype(np.float64))
    if metric == "mean":
        return float(np.mean(mag))
    if metric == "max":
        return float(np.max(mag))
    raise ValueError(f"metric must be 'mean' or 'max', got {metric!r}")


def select_min_median_max_files(
    files: list[Path],
    fmt: str,
    score_metric: str,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    scored: list[tuple[Path, float]] = []
    for fp in files:
        if fmt == "hcp3d":
            sample = load_hcp3d_sample(fp)
            scored.append(
                (fp, scalar_u_score(sample["u"], sample.get("mask"), score_metric))
            )
        else:
            _, _, phi, _ = load_triplet(fp)
            scored.append((fp, scalar_phi_score(phi, score_metric)))
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
    """One sample per class: rigid (row 1), affine (row 2), elastic (row 3)."""
    pool_files, used_fallback = prefer_qc_passed_files(files)
    if used_fallback:
        print(
            "Warning: no qc_passed=True samples in split; using all files for row selection.",
            file=sys.stderr,
        )
    pools: dict[str, list[Path]] = {cls: [] for cls, _ in DEFORMATION_ROW_ORDER}
    for fp in pool_files:
        cls = load_hcp3d_sample(fp).get("deformation_class")
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
        meta = load_hcp3d_sample(fp)
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
    """One subject/sample per (class, range) combination."""
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
            f"Run: create_synth_data.py --range-grid  # default 3 replicates per combo"
        )
    return groups


def _render_hcp3d_figure(
    picked: list[tuple[Path, str, float]],
    *,
    split: str,
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
    checker_tile: int = 16,
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
        sample = load_hcp3d_sample(fp)
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
        sample = load_hcp3d_sample(file_path)
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
        src_grid_sl = orient_axial(src_grid_raw) if "source_grid" in sample else None
        mov_grid_sl = orient_axial(mov_grid_raw) if "moving_grid" in sample else None
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
        elif src_grid_sl is not None:
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
        elif mov_grid_sl is not None:
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
            cb = checkerboard_mix(src_disp, mov_disp, tile=checker_tile)
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
    checker_tile: int,
    montage_z_step: int,
    range_grid_view: str,
) -> None:
    files = collect_npz_files(input_dir, split, HCP_SYNTH_GLOB)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{input_dir / split}'.")
    groups = group_range_grid_examples(files)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for label, fp in groups:
        sample = load_hcp3d_sample(fp)
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
        _render_hcp3d_figure(
            picked,
            split=split,
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
            checker_tile=checker_tile,
            row_h=3.0,
        )
    if no_show:
        plt.close("all")
    print(f"Saved {len(groups)} figures under {save_dir}")


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


def visualize_hcp3d_samples(
    input_dir: Path,
    split: str | None,
    save_path: Path | None,
    no_show: bool,
    *,
    selection: str,
    score_metric: str,
    num_samples: int,
    seed: int,
    slice_index: int | None,
    grid_stride: int,
    u_percentile: float,
    per_row_u_scale: bool,
    save_dir: Path | None = None,
    use_u_contours: bool = False,
    use_checkerboard: bool = False,
    checker_tile: int = 16,
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
            checker_tile=checker_tile,
            montage_z_step=montage_z_step,
            range_grid_view=range_grid_view,
        )
        return

    files = collect_npz_files(input_dir, split, HCP_SYNTH_GLOB)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{input_dir / split}'.")

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
        picked = select_min_median_max_files(pool_files, "hcp3d", score_metric)
        examples_note = f"Min / median / max of {score_metric} " + r"$\|u\|$"
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    _render_hcp3d_figure(
        picked,
        split=split,
        save_path=save_path,
        no_show=no_show,
        title="HCP Synthetic Data Plot",
        subtitle=f"{examples_note} · {split} split · Radiological-style display",
        slice_index=slice_index,
        grid_stride=grid_stride,
        u_percentile=u_percentile,
        per_row_u_scale=per_row_u_scale,
        use_u_contours=use_u_contours,
        use_checkerboard=use_checkerboard,
        checker_tile=checker_tile,
    )


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
    files = collect_npz_files(input_dir, split, TRIPLET_GLOB)
    if not files:
        raise FileNotFoundError(
            f"No files in '{input_dir / split}' matching '{TRIPLET_GLOB}'."
        )

    if selection == "random":
        rng = random.Random(seed)
        sample_count = min(num_samples, len(files))
        picked: list[tuple[Path, str, float]] = [
            (p, "", float("nan")) for p in rng.sample(files, sample_count)
        ]
        title_suffix = f"random {sample_count} of {len(files)}"
    elif selection == "min_median_max":
        picked = select_min_median_max_files(files, "triplet", phi_metric)
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


def resolve_format(data_dir: Path, split: str | None, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    for pattern in (HCP_SYNTH_GLOB, TRIPLET_GLOB):
        files = collect_npz_files(data_dir, split, pattern)
        if not files:
            continue
        try:
            return detect_format(files[0])
        except ValueError:
            continue
    loc = resolve_npz_dir(data_dir, split)
    raise FileNotFoundError(
        f"No recognizable synth NPZ in '{loc}' (expected HCP volume or triplet keys)."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize synthetic NPZ (HCP source/moving/u or IXI 2D triplets).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        "--input-dir",
        type=Path,
        default=Path("datasets/hcp_synth"),
        dest="data_dir",
        help="Root with Train/Val/Test subfolders.",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Train/Val/Test subfolder (optional; flat data-dir used for range_grid dry runs).",
    )
    p.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "hcp3d", "triplet"],
        help="NPZ layout (default: auto from first file in split).",
    )
    p.add_argument(
        "--selection",
        type=str,
        default="random",
        choices=["min_median_max", "random", "range_grid"],
        help="Sample pick: random, min/median/max, or 13 PNGs (one per class×range combo).",
    )
    p.add_argument(
        "--score-metric",
        "--phi-metric",
        type=str,
        default="mean",
        choices=["mean", "max"],
        dest="score_metric",
        help="Scalar per volume for min/median/max (‖u‖ or ‖φ‖).",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Axial slice for HCP volumes (default: mid z).",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=20,
        help="Grid line spacing in voxels (HCP overlay).",
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
        help="Overlay contour lines on top of ||u|| magnitude map.",
    )
    p.add_argument(
        "--checkerboard",
        action="store_true",
        help="Add checkerboard source/moving comparison column.",
    )
    p.add_argument(
        "--checker-tile",
        type=int,
        default=16,
        help="Checkerboard tile size in pixels (when --checkerboard).",
    )
    p.add_argument(
        "--global-u-scale",
        action="store_true",
        help="Use one shared ‖u‖ color scale across rows (default: per-row scaling).",
    )
    p.add_argument(
        "--phi-view",
        type=str,
        default=None,
        choices=["quiver", "magnitude", "x", "y"],
        help="Required for triplet format only.",
    )
    p.add_argument("--quiver-step", type=int, default=8)
    p.add_argument("--mag-vmin", type=float, default=0.0)
    p.add_argument("--mag-vmax", type=float, default=None)
    p.add_argument("--mag-percentile", type=float, default=99.0)
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
        help="Slice offset for 3-slice range_grid montage: z-step around center slice.",
    )
    p.add_argument(
        "--range-grid-view",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "montage"],
        help="Dry-run figure rows: orthogonal planes or 3-slice axial montage.",
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

    try:
        split = args.split if args.selection != "range_grid" else (args.split or None)
        if split is None and args.selection != "range_grid":
            split = "Train"
        fmt = resolve_format(data_dir, split, args.format)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if fmt == "hcp3d":
        visualize_hcp3d_samples(
            data_dir,
            split,
            args.save_path,
            args.no_show,
            selection=args.selection,
            score_metric=args.score_metric,
            num_samples=args.num_samples,
            seed=args.seed,
            slice_index=args.slice_index,
            grid_stride=args.grid_stride,
            u_percentile=args.u_percentile,
            per_row_u_scale=not args.global_u_scale,
            save_dir=args.save_dir,
            use_u_contours=args.u_contours,
            use_checkerboard=args.checkerboard,
            checker_tile=args.checker_tile,
            montage_z_step=args.range_grid_z_step,
            range_grid_view=args.range_grid_view,
        )
    else:
        if args.phi_view is None:
            print("ERROR: --phi-view is required for triplet format.", file=sys.stderr)
            return 2
        visualize_triplets(
            input_dir=data_dir,
            split=args.split,
            phi_view=args.phi_view,
            quiver_step=args.quiver_step,
            mag_vmin=args.mag_vmin,
            mag_vmax=args.mag_vmax,
            mag_percentile=args.mag_percentile,
            save_path=args.save_path,
            no_show=args.no_show,
            selection=args.selection,
            phi_metric=args.score_metric,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
