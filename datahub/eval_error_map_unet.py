#!/usr/bin/env python3
"""
Evaluate trained error-map U-Net: **Test** split for masked MSE/L1, random sample panels, and
min/median/max-by-mean-error panels; **Atlas** split for the same qualitative figures; optional
train/val curves from ``metrics.csv``.

- ``--eval-dir``: fiver root containing ``Test/`` and ``Atlas/``.
- ``--run-path``: training run directory. Weights are always ``run_path/best_model.pt``
  (single checkpoint). Also reads ``run_path/metrics.csv`` by default; writes
  ``training_curves.png``, ``test_error_pred_random.png``, ``test_error_pred_minmedmax.png``,
  ``atlas_error_pred_random.png``, ``atlas_error_pred_minmedmax.png``, and ``test_metrics.json`` there.

Example:
  python eval_error_map_unet.py --run-path ./runs/error_unet_run1 --eval-dir ./data/IXI_2D_unigrad_fiver --no-show
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Same package as train_error_map_unet.py
_DH = Path(__file__).resolve().parent
if str(_DH) not in sys.path:
    sys.path.insert(0, str(_DH))

import train_error_map_unet as teu

# Must match the filename written by train_error_map_unet.py (best_model.pt).
CHECKPOINT_FILENAME = "best_model.pt"


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def mean_error_over_slice(npz_path: Path) -> float:
    """Mean ``error_map`` over ``valid_mask`` if present, else over full slice."""
    with np.load(npz_path) as z:
        err = np.asarray(z["error_map"], dtype=np.float64)
        if "valid_mask" in z.files:
            m = np.asarray(z["valid_mask"], dtype=bool)
            if m.any():
                return float(np.mean(err[m]))
        return float(np.mean(err))


def select_min_median_max_by_mean_error(
    files: list[Path],
) -> list[tuple[Path, str, float]]:
    """Pick files with min / median / max mean(error_map); labels ``min``, ``median``, ``max``."""
    if not files:
        return []
    scored: list[tuple[Path, float]] = [(fp, mean_error_over_slice(fp)) for fp in files]
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


# Column headers (row 0); phi norms use mathtext double-bar notation.
_FIVER_COL_TITLES = (
    "fixed",
    "warped",
    r"$\|\phi_{\mathrm{true}}\|$",
    r"$\|\phi_{\mathrm{pred}}\|$",
    "error GT (px)",
    "error pred (px)",
)


def load_train_config(ckpt: dict) -> dict:
    c = ckpt.get("config") or {}
    return {
        "base_channels": int(c.get("base_channels", 32)),
        "image_norm": str(c.get("image_norm", "robust")),
        "quantile_high": float(c.get("quantile_high", 0.99)),
        "phi_scale": float(c.get("phi_scale", 64.0)),
    }


@torch.no_grad()
def evaluate_test_split(
    model: torch.nn.Module,
    eval_dir: Path,
    cfg: dict,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    show_progress: bool,
) -> tuple[float, float, int]:
    """Mean masked MSE and L1 over Test loader (same aggregation as training ``evaluate``)."""
    ds = teu.FiverErrorDataset(
        eval_dir,
        "Test",
        image_norm=cfg["image_norm"],
        quantile_high=cfg["quantile_high"],
        phi_scale=cfg["phi_scale"],
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=teu.collate_batch,
    )
    model.eval()
    sum_mse = 0.0
    sum_l1 = 0.0
    n = 0
    it = tqdm(loader, desc="Test", unit="batch", disable=not show_progress)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(x)
        sum_mse += float(teu.masked_mse(pred, y, mask))
        sum_l1 += float(teu.masked_l1(pred, y, mask))
        n += 1
    n = max(n, 1)
    return sum_mse / n, sum_l1 / n, len(ds)


def preprocess_from_npz(
    ds: teu.FiverErrorDataset,
    data: np.lib.npyio.NpzFile,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build model input (1,4,H,W) and return raw arrays for plotting."""
    image = np.asarray(data["image"], dtype=np.float32)
    warped = np.asarray(data["warped"], dtype=np.float32)
    phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
    err = np.asarray(data["error_map"], dtype=np.float32)
    phi_true = np.asarray(data["phi_true"], dtype=np.float32)

    image_n = ds._norm_image(image)
    warped_n = ds._norm_image(warped)
    phi_n = phi_pred / ds.phi_scale
    x = np.concatenate(
        [image_n[None, ...], warped_n[None, ...], phi_n],
        axis=0,
    )
    return torch.from_numpy(x).unsqueeze(0), image, warped, phi_true, phi_pred, err


def _left_axis_title_lines(
    fp: Path,
    mean_err: float,
    rank_tag: str | None,
    *,
    include_fixed_header: bool,
) -> str:
    name = fp.name
    head = f"[{rank_tag}] {name}" if rank_tag else name
    mean_line = f"mean error = {mean_err:.4f} px"
    if include_fixed_header:
        return f"fixed\n{head}\n{mean_line}"
    return f"{head}\n{mean_line}"


@torch.no_grad()
def plot_fiver_samples_grid(
    paths: list[Path],
    mean_errors: list[float],
    model: torch.nn.Module,
    ds_template: teu.FiverErrorDataset,
    device: torch.device,
    save_path: Path | None,
    no_show: bool,
    err_percentile: float,
    split_title: str,
    arrangement_detail: str,
    row_rank_tags: list[str | None] | None = None,
) -> None:
    """
    Six-panel rows: column titles on row 0 only; each row has filename + mean error on the first axis.
    ``row_rank_tags`` (e.g. min/median/max) prefix the filename when set.
    """
    if len(paths) != len(mean_errors):
        raise ValueError("paths and mean_errors must have the same length")
    if row_rank_tags is not None and len(row_rank_tags) != len(paths):
        raise ValueError("row_rank_tags must match paths length")
    model.eval()
    rows: list[
        tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for fp in paths:
        with np.load(fp) as data:
            x, image, warped, phi_true, phi_pred, err_true = preprocess_from_npz(ds_template, data)
        x = x.to(device)
        pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
        rows.append((fp, image, warped, phi_true, phi_pred, err_true, pred))

    all_vals = np.concatenate(
        [r[5].ravel() for r in rows] + [r[6].ravel() for r in rows],
        dtype=np.float64,
    )
    err_v = float(np.percentile(all_vals, err_percentile))
    if err_v <= 0:
        err_v = 1e-6

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 6, figsize=(18, 2.8 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    tags = row_rank_tags if row_rank_tags is not None else [None] * nrows

    for row, (fp, image, warped, phi_true, phi_pred, err_true, pred) in enumerate(rows):
        mag_t = phi_magnitude(phi_true.astype(np.float64))
        mag_p = phi_magnitude(phi_pred.astype(np.float64))
        m_err = mean_errors[row]
        tag = tags[row]

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(warped, cmap="gray")
        axes[row, 1].axis("off")

        im_pt = axes[row, 2].imshow(mag_t, cmap="hot", vmin=0.0)
        axes[row, 2].axis("off")
        fig.colorbar(im_pt, ax=axes[row, 2], fraction=0.046, pad=0.02)

        im_pp = axes[row, 3].imshow(mag_p, cmap="hot", vmin=0.0)
        axes[row, 3].axis("off")
        fig.colorbar(im_pp, ax=axes[row, 3], fraction=0.046, pad=0.02)

        im_et = axes[row, 4].imshow(err_true, cmap="hot", vmin=0.0, vmax=err_v)
        axes[row, 4].axis("off")
        fig.colorbar(im_et, ax=axes[row, 4], fraction=0.046, pad=0.02)

        im_ep = axes[row, 5].imshow(pred, cmap="hot", vmin=0.0, vmax=err_v)
        axes[row, 5].axis("off")
        fig.colorbar(im_ep, ax=axes[row, 5], fraction=0.046, pad=0.02)

        for col in range(6):
            ax = axes[row, col]
            if row == 0:
                if col == 0:
                    ax.set_title(
                        _left_axis_title_lines(fp, m_err, tag, include_fixed_header=True),
                        fontsize=8,
                    )
                else:
                    ax.set_title(_FIVER_COL_TITLES[col], fontsize=9)
            elif col == 0:
                ax.set_title(
                    _left_axis_title_lines(fp, m_err, tag, include_fixed_header=False),
                    fontsize=8,
                )

    fig.suptitle(
        f"{split_title} {arrangement_detail} - error U-Net - vmax err = {err_v:.4g} px "
        f"({err_percentile:g} pct)",
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


def plot_training_curves_from_csv(
    metrics_csv: Path,
    save_path: Path | None,
    no_show: bool,
    run_label: str,
) -> bool:
    """
    Plot ``train_mse`` and ``val_mse`` on the left y-axis, ``val_l1`` on the right (twin axis),
    vs epoch. MSE and L1 use different units/scales, so twin axes avoid squashing either curve.
    """
    if not metrics_csv.is_file():
        return False

    epochs: list[int] = []
    train_mse: list[float] = []
    val_mse: list[float] = []
    val_l1: list[float] = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(float(row["epoch"])))
            train_mse.append(float(row["train_mse"]))
            val_mse.append(float(row["val_mse"]))
            val_l1.append(float(row["val_l1"]))

    if not epochs:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_mse, label="train MSE", color="C0", marker=".", markersize=3)
    ax.plot(epochs, val_mse, label="val MSE", color="C1", marker=".", markersize=3)
    best_i = int(np.argmin(np.array(val_mse)))
    best_ep = epochs[best_i]
    ax.axvline(
        best_ep,
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        label=f"best val MSE (epoch {best_ep})",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.set_title(f"Training vs Validation ({run_label})")
    ax.grid(True, alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(epochs, val_l1, label="val L1 (px)", color="C2", marker=".", markersize=3)
    ax_r.set_ylabel("masked L1 (px)")
    ax_r.tick_params(axis="y", labelcolor="C2")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves: {save_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test-set metrics + Atlas visualization for error-map U-Net.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run-path",
        type=Path,
        required=True,
        help=f"Training run directory: loads {CHECKPOINT_FILENAME} and metrics.csv here; "
        "writes evaluation outputs here.",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("./data/IXI_2D_unigrad_fiver"),
        help="Fiver root with Test/ (test error) and Atlas/ (plot samples).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Override U-Net width if checkpoint has no config (default: 32).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--atlas-samples",
        type=int,
        default=3,
        metavar="N",
        help="Number of random *_fiver.npz files to plot for both Test and Atlas (default: 3).",
    )
    p.add_argument(
        "--err-percentile",
        type=float,
        default=99.0,
        help="Shared vmax for GT/pred error maps (from GT+pred samples shown).",
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Training metrics.csv. Default: run-path/metrics.csv",
    )
    p.add_argument(
        "--no-training-curves",
        action="store_true",
        help="Do not plot train/val curves from metrics.csv.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_path = Path(args.run_path).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_path / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    training_curves_path = run_path / "training_curves.png"
    test_vis_path = run_path / "test_error_pred_random.png"
    test_minmedmax_path = run_path / "test_error_pred_minmedmax.png"
    atlas_save_path = run_path / "atlas_error_pred_random.png"
    atlas_minmedmax_path = run_path / "atlas_error_pred_minmedmax.png"
    test_metrics_path = run_path / "test_metrics.json"

    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        metrics_csv = run_path / "metrics.csv"
    metrics_csv = Path(metrics_csv)

    if not args.no_training_curves:
        if metrics_csv.is_file():
            plot_training_curves_from_csv(
                metrics_csv,
                training_curves_path,
                args.no_show,
                run_path.name,
            )
        else:
            print(
                f"NOTE: training curves skipped - metrics.csv not found: {metrics_csv}",
                file=sys.stderr,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = load_train_config(ckpt)
    if args.base_channels is not None:
        cfg["base_channels"] = args.base_channels

    model = teu.UNet2D(in_channels=4, base=cfg["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    show_p = not args.no_progress
    test_mse, test_l1, n_test = evaluate_test_split(
        model,
        args.eval_dir,
        cfg,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        show_progress=show_p,
    )
    print(
        f"Test split ({n_test} files): masked MSE = {test_mse:.6f}  masked L1 = {test_l1:.6f} px"
    )

    metrics_out = {
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(Path(args.eval_dir).resolve()),
        "run_path": str(run_path),
        "metrics_csv": str(metrics_csv.resolve()) if metrics_csv.is_file() else None,
        "n_test_files": n_test,
        "test_masked_mse": test_mse,
        "test_masked_l1": test_l1,
        "preprocess_config": cfg,
    }
    test_metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {test_metrics_path}")

    rng = random.Random(args.seed)
    k = max(0, args.atlas_samples)

    test_dir = args.eval_dir / "Test"
    test_files = sorted(test_dir.glob("*_fiver.npz")) if test_dir.is_dir() else []
    if test_dir.is_dir() and test_files:
        ds_test = teu.FiverErrorDataset(
            args.eval_dir,
            "Test",
            image_norm=cfg["image_norm"],
            quantile_high=cfg["quantile_high"],
            phi_scale=cfg["phi_scale"],
        )
        if k > 0:
            kt = min(k, len(test_files))
            picked_test = rng.sample(test_files, kt)
            mean_errs_test = [mean_error_over_slice(p) for p in picked_test]
            plot_fiver_samples_grid(
                picked_test,
                mean_errs_test,
                model,
                ds_test,
                device,
                test_vis_path,
                args.no_show,
                err_percentile=args.err_percentile,
                split_title="Test",
                arrangement_detail=f"(random {kt})",
                row_rank_tags=None,
            )
        sel_test = select_min_median_max_by_mean_error(test_files)
        if sel_test:
            p_test = [t[0] for t in sel_test]
            m_test = [t[2] for t in sel_test]
            r_test = [t[1] for t in sel_test]
            plot_fiver_samples_grid(
                p_test,
                m_test,
                model,
                ds_test,
                device,
                test_minmedmax_path,
                args.no_show,
                err_percentile=args.err_percentile,
                split_title="Test",
                arrangement_detail="(min / median / max mean error over slice)",
                row_rank_tags=r_test,
            )
    elif not test_dir.is_dir():
        print(f"WARNING: Test split missing for plots: {test_dir}", file=sys.stderr)
    elif not test_files:
        print(f"WARNING: no *_fiver.npz in {test_dir} (skipping Test plot)", file=sys.stderr)

    atlas_dir = args.eval_dir / "Atlas"
    atlas_files = sorted(atlas_dir.glob("*_fiver.npz")) if atlas_dir.is_dir() else []
    if atlas_dir.is_dir() and atlas_files:
        ds_atlas = teu.FiverErrorDataset(
            args.eval_dir,
            "Atlas",
            image_norm=cfg["image_norm"],
            quantile_high=cfg["quantile_high"],
            phi_scale=cfg["phi_scale"],
        )
        if k > 0:
            ka = min(k, len(atlas_files))
            picked_atlas = rng.sample(atlas_files, ka)
            mean_errs_atlas = [mean_error_over_slice(p) for p in picked_atlas]
            plot_fiver_samples_grid(
                picked_atlas,
                mean_errs_atlas,
                model,
                ds_atlas,
                device,
                atlas_save_path,
                args.no_show,
                err_percentile=args.err_percentile,
                split_title="Atlas",
                arrangement_detail=f"(random {ka})",
                row_rank_tags=None,
            )
        sel_atlas = select_min_median_max_by_mean_error(atlas_files)
        if sel_atlas:
            p_at = [t[0] for t in sel_atlas]
            m_at = [t[2] for t in sel_atlas]
            r_at = [t[1] for t in sel_atlas]
            plot_fiver_samples_grid(
                p_at,
                m_at,
                model,
                ds_atlas,
                device,
                atlas_minmedmax_path,
                args.no_show,
                err_percentile=args.err_percentile,
                split_title="Atlas",
                arrangement_detail="(min / median / max mean error over slice)",
                row_rank_tags=r_at,
            )
    elif not atlas_dir.is_dir():
        print(f"WARNING: Atlas split missing for plots: {atlas_dir}", file=sys.stderr)
    elif not atlas_files:
        print(f"WARNING: no *_fiver.npz in {atlas_dir} (skipping Atlas plot)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
