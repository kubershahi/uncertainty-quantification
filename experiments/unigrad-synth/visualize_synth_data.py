#!/usr/bin/env python3
"""
Visualize synthetic registration NPZ samples.

Supports two formats (auto-detected or ``--format``):

**HCP 3D** (``create_synth_data.py`` output): rows = rigid / affine / non-rigid examples;
columns = source (fixed) + reference grid, warped/moving + grid bent by ``u``, ``|u|``.
Axial slices use radiological display (``rot90``, posterior up).

**IXI 2D legacy** (``*_triplet.npz``): ``image``, ``warped``, ``phi``.

Examples:
python experiments/unigrad-synth/visualize_synth_data.py --data-dir datasets/hcp_synth --split Train --num-samples 3 --save-path assets/images/unigrad-synth/hcp/hcp_synth_random3.png --no-show
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

# Default rows for ``--selection random`` (HCP 3D): one example per class, top to bottom.
DEFORMATION_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("rigid_like", "rig"),
    ("affine", "aff"),
    ("non_rigid", "nr"),
)

_GRID_COLOR = "cyan"
_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9


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
        f"(expected HCP3D {sorted(HCP3D_KEYS)} or triplet {sorted(TRIPLET_KEYS)})"
    )


def collect_npz_files(input_dir: Path, split: str, pattern: str) -> list[Path]:
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return sorted(split_dir.glob(pattern))


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


def load_hcp3d_sample(npz_path: Path) -> dict:
    with np.load(npz_path) as data:
        missing = HCP3D_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        out = {
            "source": np.asarray(data["source"]),
            "moving": np.asarray(data["moving"]),
            "u": np.asarray(data["u"]),
        }
        if "mask" in data.files:
            out["mask"] = np.asarray(data["mask"])
        if "qc_passed" in data.files:
            qc_val, qc_err = _unpack_qc_passed(data["qc_passed"])
            if qc_err:
                raise ValueError(f"{npz_path.name}: {qc_err}")
            out["qc_passed"] = qc_val
        if "deformation_class" in data.files:
            out["deformation_class"] = _unpack_scalar_str(data["deformation_class"])
        if "subject_id" in data.files:
            out["subject_id"] = _unpack_scalar_str(data["subject_id"])
    return out


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
    """One sample per deformation class: rigid (row 1), affine (row 2), non-rigid (row 3)."""
    pools: dict[str, list[Path]] = {cls: [] for cls, _ in DEFORMATION_ROW_ORDER}
    for fp in files:
        cls = load_hcp3d_sample(fp).get("deformation_class")
        if cls in pools:
            pools[str(cls)].append(fp)
    for cls, suf in DEFORMATION_ROW_ORDER:
        if not pools[cls]:
            pools[cls] = [fp for fp in files if fp.stem.endswith(f"_{suf}")]

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
    if "qc_passed" in extra:
        meta.append(f"qc={extra['qc_passed']}")
    if meta:
        lines.append(" · ".join(meta))
    return "\n".join(lines)


def visualize_hcp3d_samples(
    input_dir: Path,
    split: str,
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
) -> None:
    from matplotlib.lines import Line2D

    files = collect_npz_files(input_dir, split, HCP_SYNTH_GLOB)
    if not files:
        raise FileNotFoundError(f"No NPZ files in '{input_dir / split}'.")

    if selection == "random":
        picked = select_deformation_class_examples(files, seed)
        subtitle = (
            f"Rigid / affine / non-rigid examples (seed = {seed}) · "
            f"{len(files)} subjects in {split}"
        )
    elif selection == "min_median_max":
        picked = select_min_median_max_files(files, "hcp3d", score_metric)
        subtitle = f"Min / median / max of {score_metric} ‖u‖ across split ({len(picked)} subjects)"
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    nrows = len(picked)
    ncols = 3
    col_titles = ["Source (fixed)", "Warped (moving)", r"$|u|$"]

    z_values: list[int] = []
    u_mags: list[np.ndarray] = []
    for fp, _, _ in picked:
        sample = load_hcp3d_sample(fp)
        z = axial_slice(sample["source"], slice_index)[1]
        z_values.append(z)
        u_sl = sample["u"][:, :, :, z]
        u_mags.append(displacement_magnitude(u_sl.astype(np.float64)).ravel())
    u_vmax = float(np.percentile(np.concatenate(u_mags), u_percentile))
    u_vmax = max(u_vmax, 1e-6)

    ref_shape = load_hcp3d_sample(picked[0][0])["source"].shape
    if len(set(z_values)) == 1:
        z_note = f"axial z = {z_values[0]} / {ref_shape[2] - 1}"
    else:
        z_note = "axial z varies by row"

    plt.rcParams.update({"font.family": _FONT, "figure.dpi": _DPI, "savefig.dpi": _DPI})
    fig_w = 3.5 * ncols + 1.4
    fig_h = 3.2 * nrows + 1.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    im_u_last = None
    for row, (file_path, rank_label, _) in enumerate(picked):
        sample = load_hcp3d_sample(file_path)
        source = sample["source"]
        moving = sample["moving"]
        u = sample["u"]
        extra = {k: sample[k] for k in ("qc_passed", "deformation_class", "subject_id") if k in sample}

        src_sl, z = axial_slice(source, slice_index)
        mov_sl, _ = axial_slice(moving, slice_index)
        u_inplane = orient_axial_u_inplane(axial_u_slice(u, z))
        u_mag_sl = orient_axial(displacement_magnitude(u[:, :, :, z].astype(np.float64)))
        src_disp = orient_axial(src_sl)
        mov_disp = orient_axial(mov_sl)
        h, w = src_disp.shape

        subject_id = extra.get("subject_id") or file_path.stem.split("_")[0]

        # Col 0: source + reference grid
        ax_src = axes[row, 0]
        ax_src.imshow(src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        overlay_regular_grid(ax_src, h, w, stride=grid_stride, color=_GRID_COLOR)
        _style_axis(ax_src)
        ax_src.set_ylabel(
            _row_label(subject_id, extra),
            fontsize=_LABEL,
            rotation=45,
            ha="right",
            va="center",
            labelpad=4,
        )

        # Col 1: moving + grid bent by u
        ax_mov = axes[row, 1]
        ax_mov.imshow(mov_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3)
        overlay_deformation_grid(ax_mov, u_inplane, stride=grid_stride, color=_GRID_COLOR)
        _style_axis(ax_mov)

        # Col 2: |u| magnitude
        ax_u = axes[row, 2]
        im_u_last = ax_u.imshow(
            u_mag_sl, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        _style_axis(ax_u)

        if row == 0:
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, fontsize=_SUBTITLE, fontweight="medium", pad=10)

    # Colorbar for |u|
    if im_u_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.22, 0.018, 0.56])
        cbar = fig.colorbar(im_u_last, cax=cbar_ax)
        cbar.set_label(r"Displacement magnitude $|u|$ (voxels)", fontsize=_LABEL)
        cbar.ax.tick_params(labelsize=_LABEL - 1)

    # Legend for grid types
    legend_handles = [
        Line2D([0], [0], color=_GRID_COLOR, linewidth=1.2, label="Reference grid (source, fixed)"),
        Line2D([0], [0], color=_GRID_COLOR, linewidth=1.2, linestyle="--", label=r"Grid displaced by $u$ (moving)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=_LABEL,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "HCP Synthetic Data Plot",
        fontsize=_TITLE,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"{subtitle} · {z_note} · intensities masked z-score · radiological axial (rot90)",
        ha="center",
        va="top",
        fontsize=_LABEL,
        color="black",
    )

    fig.subplots_adjust(left=0.12, right=0.90, top=0.86, bottom=0.10, wspace=0.22, hspace=0.32)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


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


def resolve_format(data_dir: Path, split: str, fmt: str) -> str:
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
    raise FileNotFoundError(
        f"No recognizable synth NPZ in '{data_dir / split}' "
        f"(expected HCP3D or triplet keys)."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize synthetic NPZ (HCP 3D source/moving/u or IXI 2D triplets).",
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
    p.add_argument("--split", type=str, default="Train", choices=list(SPLITS))
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
        choices=["min_median_max", "random"],
        help="Sample pick: random or min/median/max deformation score.",
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
        help="Axial slice for HCP 3D (default: mid z).",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=12,
        help="Grid line spacing in voxels (HCP 3D overlay).",
    )
    p.add_argument(
        "--u-percentile",
        type=float,
        default=99.0,
        help="Color scale cap for ‖u‖ (percentile across selected samples).",
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
        fmt = resolve_format(data_dir, args.split, args.format)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if fmt == "hcp3d":
        visualize_hcp3d_samples(
            data_dir,
            args.split,
            args.save_path,
            args.no_show,
            selection=args.selection,
            score_metric=args.score_metric,
            num_samples=args.num_samples,
            seed=args.seed,
            slice_index=args.slice_index,
            grid_stride=args.grid_stride,
            u_percentile=args.u_percentile,
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
