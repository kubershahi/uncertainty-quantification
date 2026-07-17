#!/usr/bin/env python3
"""
Evaluate a trained HCP 3D error-map U-Net (Phase III).

Writes under ``--run-path``: ``training_curves.png``, ``test_metrics.json``,
and mid-slice QC figures for random / easy-normal-hard Test samples.

Example:
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/3d/error_unet_run1 --eval-dir datasets/error-map/unigrad-synth/hcp --no-show
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
if str(_DH) not in sys.path:
    sys.path.insert(0, str(_DH))

import train_unigrad_synth_unet as teu

CHECKPOINT_FILENAME = "best_model.pt"
PRODUCT_NAME = "UniGrad Synth HCP"
DISPLACEMENT_UNIT = "voxels"
RANK_METRIC_LABEL = "mean(error_map|source_mask)"

COLUMN_TITLES = (
    "source (fixed)",
    "moving (warped)",
    r"$\|u_{\mathrm{pred}}\|$",
    f"error GT ({DISPLACEMENT_UNIT})",
    f"error pred ({DISPLACEMENT_UNIT})",
)


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


def mean_error_in_source_mask(npz_path: Path) -> float:
    with np.load(npz_path) as z:
        if "u_error_map" in z.files:
            err = np.asarray(z["u_error_map"], dtype=np.float64)
        else:
            u_gt = np.asarray(z["u_gt"], dtype=np.float64)
            u_pred = np.asarray(z["u_pred"], dtype=np.float64)
            err = np.linalg.norm(u_gt - u_pred, axis=0)
        mask = np.asarray(z["source_mask"], dtype=bool)
    vals = err[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def select_easy_normal_hard(
    files: list[Path],
    *,
    show_progress: bool,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    it = tqdm(files, desc="rank volumes", unit="file", disable=not show_progress)
    scored = [(fp, mean_error_in_source_mask(fp)) for fp in it]
    scored = [(fp, s) for fp, s in scored if np.isfinite(s)]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 0:
        return []
    if n == 1:
        return [(scored[0][0], "easy", scored[0][1])]
    if n == 2:
        return [(scored[0][0], "easy", scored[0][1]), (scored[-1][0], "hard", scored[-1][1])]
    return [
        (scored[0][0], "easy", scored[0][1]),
        (scored[n // 2][0], "normal", scored[n // 2][1]),
        (scored[-1][0], "hard", scored[-1][1]),
    ]


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


def mid_axial_slice(vol: np.ndarray) -> np.ndarray:
    """``(X,Y,Z)`` → axial mid-slice ``(X,Y)`` then radiological ``rot90``."""
    z = vol.shape[2] // 2
    return np.rot90(vol[:, :, z])


@torch.no_grad()
def infer_volume_panels(
    fp: Path,
    model: torch.nn.Module,
    ds: teu.HCPErrorMapDataset,
    device: torch.device,
) -> tuple[np.ndarray, ...]:
    idx = ds.paths.index(fp)
    batch = ds[idx]
    with np.load(fp) as data:
        source = np.asarray(data["source"], dtype=np.float32)
        moving = np.asarray(data["moving"], dtype=np.float32)
        u_pred = np.asarray(data["u_pred"], dtype=np.float32)
        if "u_error_map" in data.files:
            err_gt = np.asarray(data["u_error_map"], dtype=np.float32)
        else:
            u_gt = np.asarray(data["u_gt"], dtype=np.float32)
            err_gt = np.linalg.norm(u_gt - u_pred, axis=0).astype(np.float32)

    x = batch["x"].unsqueeze(0).to(device)
    pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
    u_mag = np.linalg.norm(u_pred.astype(np.float64), axis=0).astype(np.float32)

    return (
        mid_axial_slice(source),
        mid_axial_slice(moving),
        mid_axial_slice(u_mag),
        mid_axial_slice(err_gt),
        mid_axial_slice(pred),
    )


def plot_samples_grid(
    items: list[tuple[Path, str, float]],
    model: torch.nn.Module,
    ds: teu.HCPErrorMapDataset,
    device: torch.device,
    save_path: Path | None,
    no_show: bool,
    err_percentile: float,
    arrangement: str,
    *,
    show_progress: bool,
) -> None:
    model.eval()
    rows: list[tuple[Path, str, float, tuple[np.ndarray, ...]]] = []
    it = tqdm(items, desc=f"plot {arrangement}", unit="vol", disable=not show_progress)
    for fp, tag, mean_err in it:
        panels = infer_volume_panels(fp, model, ds, device)
        rows.append((fp, tag, mean_err, panels))

    all_err = np.concatenate([r[3][3].ravel() for r in rows] + [r[3][4].ravel() for r in rows])
    err_v = max(float(np.percentile(all_err, err_percentile)), 1e-6)
    u_all = np.concatenate([r[3][2].ravel() for r in rows])
    u_v = max(float(np.percentile(u_all, err_percentile)), 1e-6)

    nrows = len(rows)
    ncol = len(COLUMN_TITLES)
    fig, axes = plt.subplots(nrows, ncol, figsize=(3.0 * ncol, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for row, (fp, tag, mean_err, panels) in enumerate(rows):
        src, mov, umag, err_gt, err_pr = panels
        specs = [
            (src, "gray", None, None),
            (mov, "gray", None, None),
            (umag, "hot", 0.0, u_v),
            (err_gt, "hot", 0.0, err_v),
            (err_pr, "hot", 0.0, err_v),
        ]
        for col, (img, cmap, vmin, vmax) in enumerate(specs):
            ax = axes[row, col]
            if vmin is None:
                ax.imshow(img, cmap=cmap, origin="upper")
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cbar.ax.tick_params(labelsize=7)
            ax.axis("off")
            if row == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=9, pad=4)
        label = f"[{tag}] {fp.stem}\n{RANK_METRIC_LABEL}={mean_err:.3f}"
        axes[row, 0].text(
            -0.04,
            0.5,
            label,
            rotation=90,
            va="center",
            ha="right",
            transform=axes[row, 0].transAxes,
            fontsize=8,
            fontweight="bold",
        )

    title = (
        f"{PRODUCT_NAME} · Test | random"
        if arrangement == "random"
        else f"{PRODUCT_NAME} · Test | ranked by {RANK_METRIC_LABEL}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.08, 0.02, 1, 0.95))
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
    if not metrics_csv.is_file():
        return False
    epochs: list[int] = []
    train_loss: list[float] = []
    val_mae: list[float] = []
    val_rmse: list[float] = []
    val_r: list[float] = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(float(row["epoch"])))
            train_loss.append(float(row["train_loss"]))
            val_mae.append(float(row["val_mae"]))
            val_rmse.append(float(row["val_rmse"]))
            val_r.append(float(row["val_pearson_r"]))
    if not epochs:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="train loss", color="C0", marker=".", markersize=3)
    ax.plot(epochs, val_mae, label="val MAE", color="C1", marker=".", markersize=3)
    ax.plot(epochs, val_rmse, label="val RMSE", color="C3", marker=".", markersize=3)
    best_i = int(np.argmin(np.array(val_mae)))
    ax.axvline(
        epochs[best_i],
        color="0.5",
        linestyle="--",
        linewidth=0.8,
        label=f"best MAE (ep {epochs[best_i]})",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"error ({DISPLACEMENT_UNIT})")
    ax.set_title(f"Training curves ({run_label})")
    ax.grid(True, alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(epochs, val_r, label="val Pearson r", color="C2", marker=".", markersize=3)
    ax_r.set_ylabel("Pearson r")
    ax_r.set_ylim(-1.05, 1.05)
    ax_r.tick_params(axis="y", labelcolor="C2")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate 3D HCP error-map U-Net on UniGrad synth NPZ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run-path",
        type=Path,
        default=Path("assets/runs/regression/unigrad-synth/3d/error_unet_run1"),
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("datasets/error-map/unigrad-synth/hcp"),
    )
    p.add_argument("--eval-split", type=str, default="Test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-random", type=int, default=3)
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--no-training-curves", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = Path(args.run_path).resolve()
    eval_dir = Path(args.eval_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_path.mkdir(parents=True, exist_ok=True)
    show_p = not args.no_progress

    checkpoint_path = run_path / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_path / "metrics.csv"
    if not args.no_training_curves:
        if metrics_csv.is_file():
            plot_training_curves_from_csv(
                metrics_csv,
                run_path / "training_curves.png",
                args.no_show,
                run_path.name,
            )
        else:
            print(f"NOTE: no metrics.csv at {metrics_csv}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = load_train_config(ckpt, args.base_channels)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

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
    (run_path / "test_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    test_dir = eval_dir / args.eval_split
    test_files = sorted(test_dir.glob(teu.DATA_GLOB)) if test_dir.is_dir() else []
    if not test_files:
        print(f"WARNING: no NPZ under {test_dir}", file=sys.stderr)
        return 0

    ds_test = teu.HCPErrorMapDataset(
        eval_dir,
        args.eval_split,
        u_scale=cfg["u_scale"],
        mask_u_pred=cfg["mask_u_pred"],
        image_norm=cfg["image_norm"],
        quantile_high=cfg["quantile_high"],
    )

    rng = random.Random(args.seed)
    if args.num_random > 0:
        k = min(args.num_random, len(test_files))
        picked = rng.sample(test_files, k)
        items = [(p, "", mean_error_in_source_mask(p)) for p in picked]
        plot_samples_grid(
            items,
            model,
            ds_test,
            device,
            run_path / "test_error_pred_random.png",
            args.no_show,
            args.err_percentile,
            "random",
            show_progress=show_p,
        )

    ranked = select_easy_normal_hard(test_files, show_progress=show_p)
    if ranked:
        plot_samples_grid(
            ranked,
            model,
            ds_test,
            device,
            run_path / "test_error_pred_easy_normal_hard.png",
            args.no_show,
            args.err_percentile,
            "easy_normal_hard",
            show_progress=show_p,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
