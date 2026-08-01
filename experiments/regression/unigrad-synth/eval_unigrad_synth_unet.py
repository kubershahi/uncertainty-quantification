#!/usr/bin/env python3
"""
Evaluate a trained HCP 3D error-map U-Net (Phase III).

Figures (orthogonal, radiological) match Phase II layout, with predicted error:

  Source | Warped | ‖u_gt‖ | ‖u_pred‖ | [‖u‖ cbar] | GT Error Map | Pred Error Map | [err cbar]

Error maps are shown after ``source_mask`` (same brain-only focus as training loss).

Writes under ``--run-path`` (when enabled by ``--mode``):

  - ``training_curves.png``
  - ``test_random_orthogonal/{none,rigid,affine,elastic,affine_elastic}.png`` (5)
  - ``test_mmm_orthogonal/{min,median,max}.png`` (3; picks from Phase-I CSV)
  - ``test_metrics.json`` (masked Test MAE / RMSE / Pearson r) — after figures

Example:
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/hcp/error_unet_run1 --eval-dir datasets/error-map/unigrad-synth/hcp --mode both --no-show

Example (figures only):
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/hcp/error_unet_run1 --mode figures --no-show

Example (metrics only):
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/hcp/error_unet_run1 --mode metrics --no-show
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_DH = Path(__file__).resolve().parent
_REPO = _DH.parents[2]
_VIZ_DIR = _REPO / "experiments" / "error-map-gen" / "unigrad-synth"
for _p in (_DH, _VIZ_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_unigrad_synth_unet as teu
import visualize_unigrad_data as viz

CHECKPOINT_FILENAME = "best_model.pt"
DISPLACEMENT_UNIT = "voxels"
DEFAULT_MMM_CSV = viz.DEFAULT_MMM_CSV
DEFAULT_RUN_PATH = Path("assets/runs/regression/unigrad-synth/hcp/error_unet_run1")
DEFAULT_EVAL_DIR = Path("datasets/error-map/unigrad-synth/hcp")

_DPI = 150
_FONT = "DejaVu Sans"
_TITLE = 12
_SUBTITLE = 10
_LABEL = 9
_U_COLOR_PERCENTILE = 99.0
_ERR_COLOR_PERCENTILE = 99.0


def load_train_config(ckpt: dict, base_channels_override: int | None) -> dict:
    c = ckpt.get("config") or {}
    base = int(c.get("base_channels", 32))
    if base_channels_override is not None:
        base = base_channels_override
    return {
        "model": str(c.get("model", "UNet3D")),
        "in_channels": int(c.get("in_channels", teu.IN_CHANNELS)),
        "base_channels": base,
        "image_norm": str(c.get("image_norm", "none")),
        "quantile_high": float(c.get("quantile_high", 0.99)),
        "u_scale": float(c.get("u_scale", c.get("phi_scale", 64.0))),
        "mask_u_pred": bool(c.get("mask_u_pred", c.get("mask_input_u_pred", False))),
    }


def build_model(cfg: dict) -> teu.UNet3D:
    if cfg["model"] != "UNet3D":
        raise ValueError(f"Expected UNet3D checkpoint, got {cfg['model']!r}")
    return teu.UNet3D(in_channels=cfg["in_channels"], base=cfg["base_channels"])


def load_eval_sample(npz_path: Path) -> dict:
    sample = viz.load_sample(npz_path)
    with np.load(npz_path) as data:
        if "source_mask" not in data.files:
            raise KeyError(f"{npz_path.name} missing source_mask")
        sample["source_mask"] = np.asarray(data["source_mask"], dtype=bool)
    return sample


@torch.no_grad()
def predict_error_map(
    fp: Path,
    model: torch.nn.Module,
    ds: teu.HCPErrorMapDataset,
    device: torch.device,
) -> np.ndarray:
    """Full-volume predicted error map ``(X, Y, Z)`` float32 (native NPZ spatial size).

    ``UNet3D.forward`` pads internally and crops back. A defensive crop remains for
    older checkpoints / code paths that returned padded outputs.
    """
    idx = ds.paths.index(fp)
    batch = ds[idx]
    x = batch["x"].unsqueeze(0).to(device)
    pred = model(x).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    with np.load(fp) as z:
        out_shape = tuple(int(s) for s in np.asarray(z["source"]).shape)
    return _crop_spatial_to(pred, out_shape)


def _crop_spatial_to(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Crop trailing pad if present (no-op when shapes already match)."""
    if arr.shape == shape:
        return arr
    if len(arr.shape) != len(shape):
        raise ValueError(f"rank mismatch: {arr.shape} vs {shape}")
    if any(a < s for a, s in zip(arr.shape, shape)):
        raise ValueError(f"cannot crop {arr.shape} down to {shape}")
    return np.asarray(arr[tuple(slice(0, s) for s in shape)])


def _masked_error(err: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = _crop_spatial_to(np.asarray(err, dtype=np.float32), mask.shape)
    out = err.copy()
    out[~np.asarray(mask, dtype=bool)] = 0.0
    return out


def _eval_plot_title(deformation_class: str) -> str:
    label = viz.DEFORM_TITLE_LABELS.get(
        deformation_class, deformation_class.replace("_", "+").title()
    )
    return f"Unigrad Synthetic Predicted Error Map Plot ({label} Transformation)"


def _eval_plot_subtitle(
    subject_id: str, run_view: str = "orthogonal", extra: str | None = None
) -> str:
    base = (
        f"Test Subject {subject_id} - Radiological-Style Display - "
        f"{viz._view_label(run_view)}"
    )
    if extra:
        return f"{base} · {extra}"
    return base


def _error_map_stats(err: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """min / Q1 / mean / Q3 / max of error voxels inside ``source_mask``."""
    vals = np.asarray(err, dtype=np.float64)[np.asarray(mask, dtype=bool)].ravel()
    if vals.size == 0:
        return {"min": float("nan"), "q1": float("nan"), "mean": float("nan"), "q3": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(vals)),
        "q1": float(np.percentile(vals, 25)),
        "mean": float(np.mean(vals)),
        "q3": float(np.percentile(vals, 75)),
        "max": float(np.max(vals)),
    }


def _render_eval_orthogonal(
    fp: Path,
    *,
    err_pred: np.ndarray,
    save_path: Path,
    no_show: bool,
    subtitle_extra: str | None = None,
) -> None:
    """
    Orthogonal 3-row figure:

      Source | Warped | ‖u_gt‖ | ‖u_pred‖ | [‖u‖ cbar] | GT Error | Pred Error | [err cbar]
    """
    from matplotlib.gridspec import GridSpec

    sample = load_eval_sample(fp)
    source = sample["source"]
    moving = sample["moving"]
    u_gt = sample["u_gt"]
    u_pred = sample["u_pred"]
    mask = sample["source_mask"]
    err_gt = _masked_error(sample["u_error_map"], mask)
    err_pr = _masked_error(err_pred, mask)
    if err_pr.shape != err_gt.shape:
        raise ValueError(
            f"pred error shape {err_pr.shape} != GT error shape {err_gt.shape} for {fp.name}"
        )

    plane_rows, idx_rows = viz._views_for_sample(
        sample,
        z_slice_index=None,
        montage_z_step=10,
        run_view="orthogonal",
    )

    n_grid_cols = 8
    img_cols = (0, 1, 2, 3, 5, 6)
    width_ratios = [1.0, 1.0, 1.0, 1.0, 0.28, 1.0, 1.0, 0.32]
    col_titles = {
        0: "Source (Fixed)",
        1: "Warped (Moving)",
        2: r"$\|u_{\mathrm{gt}}\|$",
        3: r"$\|u_{\mathrm{pred}}\|$",
        5: r"GT Error Map ($\|u_{\mathrm{gt}}-u_{\mathrm{pred}}\|$)",
        6: "Predicted Error Map",
    }

    nrows = len(plane_rows)
    row_u_vmax: list[float] = []
    row_err_vmax: list[float] = []
    for row in range(nrows):
        plane = plane_rows[row]
        idx = int(idx_rows[row])
        mag_gt = viz.magnitude_plane_slice(u_gt, plane, idx).ravel()
        mag_pr = viz.magnitude_plane_slice(u_pred, plane, idx).ravel()
        err_gt_sl, _ = viz.plane_slice(err_gt, plane, idx)
        err_pr_sl, _ = viz.plane_slice(err_pr, plane, idx)
        mask_sl, _ = viz.plane_slice(mask.astype(np.float32), plane, idx)
        m = mask_sl.astype(bool)
        u_cap = max(
            float(np.percentile(mag_gt, _U_COLOR_PERCENTILE)),
            float(np.percentile(mag_pr, _U_COLOR_PERCENTILE)),
            1e-6,
        )
        err_vals = np.concatenate([err_gt_sl[m].ravel(), err_pr_sl[m].ravel()])
        if err_vals.size == 0:
            err_cap = 1e-6
        else:
            err_cap = max(float(np.percentile(err_vals, _ERR_COLOR_PERCENTILE)), 1e-6)
        row_u_vmax.append(u_cap)
        row_err_vmax.append(err_cap)

    gt_err_stats = _error_map_stats(err_gt, mask)
    pred_err_stats = _error_map_stats(err_pr, mask)
    stats_note = (
        viz._format_u_stats_line(r"‖GT Error Map‖", gt_err_stats)
        + "\n"
        + viz._format_u_stats_line(r"‖Predicted Error Map‖", pred_err_stats)
    )

    subject_id = sample.get("subject_id") or fp.stem.split("_")[0]
    deform_cls = str(
        sample.get("deformation_class") or viz.deformation_class_from_filename(fp)
    )
    title = _eval_plot_title(deform_cls)
    subtitle = _eval_plot_subtitle(str(subject_id), "orthogonal", subtitle_extra)

    plt.rcParams.update({"font.family": _FONT, "figure.dpi": _DPI, "savefig.dpi": _DPI})
    fig_w = 3.0 * 6 + 3.2
    fig_h = 3.0 * nrows + 1.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, right = 0.16, 0.99
    gs = GridSpec(
        nrows,
        n_grid_cols,
        figure=fig,
        width_ratios=width_ratios,
        left=left,
        right=right,
        top=0.86,
        bottom=0.12,
        wspace=0.35,
        hspace=0.34,
    )

    axes: dict[tuple[int, int], plt.Axes] = {}
    for row in range(nrows):
        for col in img_cols:
            axes[(row, col)] = fig.add_subplot(gs[row, col])

    im_u_last = None
    im_err_last = None
    for row in range(nrows):
        plane = plane_rows[row]
        idx = int(idx_rows[row])
        src_sl, _ = viz.plane_slice(source, plane, idx)
        mov_sl, _ = viz.plane_slice(moving, plane, idx)
        err_gt_sl, _ = viz.plane_slice(err_gt, plane, idx)
        err_pr_sl, _ = viz.plane_slice(err_pr, plane, idx)
        mag_gt = viz.orient_axial(viz.magnitude_plane_slice(u_gt, plane, idx))
        mag_pr = viz.orient_axial(viz.magnitude_plane_slice(u_pred, plane, idx))
        src_disp = viz.orient_axial(src_sl)
        mov_disp = viz.orient_axial(mov_sl)
        err_gt_disp = viz.orient_axial(err_gt_sl)
        err_pr_disp = viz.orient_axial(err_pr_sl)
        u_vmax = row_u_vmax[row]
        err_vmax = row_err_vmax[row]

        ax_src = axes[(row, 0)]
        ax_src.imshow(
            src_disp, cmap="gray", origin="upper", interpolation="nearest", vmin=-3, vmax=3
        )
        viz._style_axis(ax_src)
        ax_src.set_ylabel(
            viz._plane_row_label(plane, idx),
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
        viz._style_axis(ax_mov)

        ax_ugt = axes[(row, 2)]
        im_u_last = ax_ugt.imshow(
            mag_gt, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        viz._style_axis(ax_ugt)

        ax_upr = axes[(row, 3)]
        ax_upr.imshow(
            mag_pr, cmap="hot", vmin=0.0, vmax=u_vmax, origin="upper", interpolation="nearest"
        )
        viz._style_axis(ax_upr)

        ax_egt = axes[(row, 5)]
        im_err_last = ax_egt.imshow(
            err_gt_disp,
            cmap="hot",
            vmin=0.0,
            vmax=err_vmax,
            origin="upper",
            interpolation="nearest",
        )
        viz._style_axis(ax_egt)

        ax_epr = axes[(row, 6)]
        ax_epr.imshow(
            err_pr_disp,
            cmap="hot",
            vmin=0.0,
            vmax=err_vmax,
            origin="upper",
            interpolation="nearest",
        )
        viz._style_axis(ax_epr)

        if row == 0:
            for col, t in col_titles.items():
                axes[(0, col)].set_title(t, fontsize=_SUBTITLE, fontweight="medium", pad=10)

    mid_row = nrows // 2
    ref_ax = axes[(mid_row, 0)]
    fig.canvas.draw()
    if im_u_last is not None:
        viz._add_midheight_colorbar(
            fig,
            im_u_last,
            gs=gs,
            cbar_col=4,
            ref_ax=ref_ax,
            label=r"$\|u\|$ (voxels)",
        )
    if im_err_last is not None:
        viz._add_midheight_colorbar(
            fig,
            im_err_last,
            gs=gs,
            cbar_col=7,
            ref_ax=ref_ax,
            label=r"Error (voxels, source_mask)",
        )

    title_x = 0.5 * (left + right)
    fig.suptitle(title, fontsize=_TITLE, fontweight="bold", x=title_x, y=0.98, ha="center")
    fig.text(title_x, 0.935, subtitle, ha="center", va="top", fontsize=_LABEL, color="black")
    fig.text(
        title_x,
        0.02,
        stats_note,
        ha="center",
        va="bottom",
        fontsize=_LABEL - 1,
        color="0.25",
        family="monospace",
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {save_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    eval_dir: Path,
    split: str,
    cfg: dict,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    show_progress: bool,
) -> tuple[dict[str, float], int]:
    ds = teu.HCPErrorMapDataset(
        eval_dir,
        split,
        u_scale=cfg["u_scale"],
        mask_u_pred=cfg["mask_u_pred"],
        image_norm=cfg["image_norm"],
        quantile_high=cfg["quantile_high"],
    )
    loader = teu.make_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    metrics = teu.evaluate(
        model,
        loader,
        device,
        use_amp=False,
        show_progress=show_progress,
        desc=f"eval {split}",
    )
    return metrics, len(ds)


def plot_training_curves_from_csv(
    metrics_csv: Path,
    save_path: Path | None,
    no_show: bool,
    run_label: str,
    *,
    train_loss: str = "mae",
    val_loss: str = "mae",
) -> bool:
    if not metrics_csv.is_file():
        return False
    epochs: list[int] = []
    train_vals: list[float] = []
    val_mae: list[float] = []
    val_mse: list[float] = []
    val_rmse: list[float] = []
    val_r: list[float] = []
    train_col = f"train_{train_loss}"
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if train_col not in fields and "train_loss" in fields:
            train_col = "train_loss"
        for row in reader:
            epochs.append(int(float(row["epoch"])))
            train_vals.append(float(row[train_col]))
            val_mae.append(float(row["val_mae"]))
            val_mse.append(
                float(row["val_mse"]) if "val_mse" in row and row["val_mse"] else float("nan")
            )
            val_rmse.append(float(row["val_rmse"]))
            val_r.append(float(row["val_pearson_r"]))
    if not epochs:
        return False

    sel_map = {"mae": val_mae, "mse": val_mse, "rmse": val_rmse}
    sel_vals = np.asarray(sel_map.get(val_loss, val_mae), dtype=float)
    finite = np.isfinite(sel_vals)
    best_i = 0 if not np.any(finite) else int(np.nanargmin(sel_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epochs,
        train_vals,
        label=f"Train {train_loss.upper()}",
        color="C0",
        marker=".",
        markersize=3,
    )
    ax.plot(epochs, val_mae, label="Val MAE", color="C1", marker=".", markersize=3)
    ax.plot(epochs, val_rmse, label="Val RMSE", color="C3", marker=".", markersize=3)
    ax.axvline(
        epochs[best_i],
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        label=f"Best Val {val_loss.upper()} (Ep {epochs[best_i]})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Error ({DISPLACEMENT_UNIT})")
    ax.set_title(f"Training Curves ({run_label})")
    ax.grid(True, alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(epochs, val_r, label="Val Pearson r", color="C2", marker=".", markersize=3)
    ax_r.set_ylabel("Pearson r")
    ax_r.set_ylim(-1.05, 1.05)
    ax_r.tick_params(axis="y", labelcolor="C2")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="center right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves: {save_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()
    return True


def generate_test_figures(
    *,
    model: torch.nn.Module,
    ds: teu.HCPErrorMapDataset,
    device: torch.device,
    eval_dir: Path,
    split: str,
    run_path: Path,
    seed: int,
    mmm_selection_csv: Path,
    no_show: bool,
    show_progress: bool,
) -> None:
    test_files = viz.collect_npz_files(eval_dir, split)
    if not test_files:
        print(f"WARNING: no NPZ under {eval_dir / split}", file=sys.stderr)
        return

    # --- random orthogonal: one sample per deformation class ---
    out_rand = run_path / "test_random_orthogonal"
    out_rand.mkdir(parents=True, exist_ok=True)
    groups, missing = viz.group_class_examples_optional(
        test_files, seed=seed + hash(split) % 10007
    )
    if missing:
        print(
            f"Warning: {split} missing class(es) {', '.join(missing)}; "
            "plotting available classes only.",
            file=sys.stderr,
        )
    it = tqdm(groups, desc="test_random_orthogonal", unit="fig", disable=not show_progress)
    for label, fp in it:
        print(f"  random / {label}: {fp.name}")
        err_pred = predict_error_map(fp, model, ds, device)
        _render_eval_orthogonal(
            fp,
            err_pred=err_pred,
            save_path=out_rand / f"{label}.png",
            no_show=no_show,
        )

    # --- mmm orthogonal: Phase-I CSV picks ---
    out_mmm = run_path / "test_mmm_orthogonal"
    out_mmm.mkdir(parents=True, exist_ok=True)
    picked = viz.load_mmm_picks_from_csv(
        mmm_selection_csv, eval_dir, split, u_metric="mean"
    )
    it = tqdm(picked, desc="test_mmm_orthogonal", unit="fig", disable=not show_progress)
    for fp, rank, score in it:
        print(f"  mmm / {rank}: mean ‖u‖={score:.3f}  {fp.name}")
        err_pred = predict_error_map(fp, model, ds, device)
        _render_eval_orthogonal(
            fp,
            err_pred=err_pred,
            save_path=out_mmm / f"{rank}.png",
            no_show=no_show,
            subtitle_extra=viz._mmm_rank_subtitle(rank, "mean"),
        )

    if no_show:
        plt.close("all")
    print(
        f"Figures: {len(groups)} under {out_rand.name}/, "
        f"{len(picked)} under {out_mmm.name}/"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate 3D HCP error-map U-Net on UniGrad synth NPZ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run-path", type=Path, default=DEFAULT_RUN_PATH)
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--eval-split", type=str, default="Test")
    p.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["figures", "metrics", "both"],
        help="Generate QC figures, Test metrics, or both (default: both). "
        "Figures are always written before metrics when mode=both.",
    )
    p.add_argument(
        "--mmm-selection-csv",
        type=Path,
        default=DEFAULT_MMM_CSV,
        help="Phase-I min/median/max selection CSV for test_mmm_orthogonal figures.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument(
        "--no-training-curves",
        action="store_true",
        help="Skip training_curves.png even when --mode includes figures.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = Path(args.run_path).resolve()
    eval_dir = Path(args.eval_dir)
    do_figures = args.mode in ("figures", "both")
    do_metrics = args.mode in ("metrics", "both")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_path.mkdir(parents=True, exist_ok=True)
    show_p = not args.no_progress

    checkpoint_path = run_path / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = load_train_config(ckpt, args.base_channels)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if do_figures:
        metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_path / "metrics.csv"
        run_cfg_path = run_path / "run_config.json"
        train_loss_name = "mae"
        val_loss_name = "mae"
        if run_cfg_path.is_file():
            try:
                run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
                train_loss_name = str(run_cfg.get("train_loss", run_cfg.get("loss", "mae")))
                if train_loss_name == "l1":
                    train_loss_name = "mae"
                val_loss_name = str(run_cfg.get("val_loss", "mae"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"NOTE: could not read {run_cfg_path}: {e}", file=sys.stderr)

        if not args.no_training_curves:
            if metrics_csv.is_file():
                plot_training_curves_from_csv(
                    metrics_csv,
                    run_path / "training_curves.png",
                    args.no_show,
                    run_path.name,
                    train_loss=train_loss_name,
                    val_loss=val_loss_name,
                )
            else:
                print(f"NOTE: no metrics.csv at {metrics_csv}", file=sys.stderr)

        ds_test = teu.HCPErrorMapDataset(
            eval_dir,
            args.eval_split,
            u_scale=cfg["u_scale"],
            mask_u_pred=cfg["mask_u_pred"],
            image_norm=cfg["image_norm"],
            quantile_high=cfg["quantile_high"],
        )
        generate_test_figures(
            model=model,
            ds=ds_test,
            device=device,
            eval_dir=eval_dir,
            split=args.eval_split,
            run_path=run_path,
            seed=args.seed,
            mmm_selection_csv=Path(args.mmm_selection_csv),
            no_show=args.no_show,
            show_progress=show_p,
        )

    if do_metrics:
        metrics, n_test = evaluate_split(
            model,
            eval_dir,
            args.eval_split,
            cfg,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            show_progress=show_p,
        )
        print(
            f"{args.eval_split} ({n_test} volumes): "
            f"MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  "
            f"r={metrics['pearson_r']:.4f}  ({DISPLACEMENT_UNIT}, source_mask)"
        )
        metrics_out = {
            "checkpoint": str(checkpoint_path),
            "eval_dir": str(eval_dir.resolve()),
            "eval_split": args.eval_split,
            "n_volumes": n_test,
            "masked_mae": metrics["mae"],
            "masked_rmse": metrics["rmse"],
            "masked_pearson_r": metrics["pearson_r"],
            "unit": DISPLACEMENT_UNIT,
            "mask": "source_mask",
            "config": cfg,
        }
        out_json = run_path / "test_metrics.json"
        out_json.write_text(json.dumps(metrics_out, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
