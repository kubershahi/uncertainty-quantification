#!/usr/bin/env python3
"""
Evaluate a trained 2D error-map U-Net on UniGrad Synth ``*_fiver.npz`` slices.

Writes under ``--run-path``: ``training_curves.png``, ``test_metrics.json``,
``test_error_pred_random.png``, ``test_error_pred_easy_normal_hard.png``.

Example:
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/2d/error_unet_run1 --eval-dir datasets/error-map/unigrad-synth/ixi_2d_fiver --no-show
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

_DH = Path(__file__).resolve().parent
if str(_DH) not in sys.path:
    sys.path.insert(0, str(_DH))

import train_unigrad_synth_unet as teu

CHECKPOINT_FILENAME = "best_model.pt"
PRODUCT_NAME = "UniGrad Synth"
DISPLACEMENT_UNIT = "pixels"
RANK_METRIC_LABEL = "mean(error_map)"

COLUMN_TITLES = (
    "subject (source)",
    "warped (target)",
    r"$\|\phi_{\mathrm{true}}\|$",
    r"$\|\phi_{\mathrm{pred}}\|$",
    f"error GT ({DISPLACEMENT_UNIT})",
    f"error pred ({DISPLACEMENT_UNIT})",
)


def phi_magnitude_2d(phi: np.ndarray) -> np.ndarray:
    if phi.ndim != 3 or phi.shape[0] != 2:
        raise ValueError(f"Expected phi (2, H, W), got {phi.shape}")
    return np.sqrt(np.sum(phi.astype(np.float64) ** 2, axis=0))


def mean_error_map_volume(npz_path: Path) -> float:
    with np.load(npz_path) as z:
        return float(np.mean(z["error_map"]))


def select_easy_normal_hard_by_mean_error(
    files: list[Path],
    *,
    show_progress: bool,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    it = tqdm(files, desc="rank slices", unit="file", disable=not show_progress)
    scored = [(fp, mean_error_map_volume(fp)) for fp in it]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 1:
        return [(scored[0][0], "easy", scored[0][1])]
    if n == 2:
        return [(scored[0][0], "easy", scored[0][1]), (scored[-1][0], "hard", scored[-1][1])]
    return [
        (scored[0][0], "easy", scored[0][1]),
        (scored[n // 2][0], "normal", scored[n // 2][1]),
        (scored[-1][0], "hard", scored[-1][1]),
    ]


def load_train_config(ckpt: dict, base_channels_override: int | None) -> dict:
    c = ckpt.get("config") or {}
    base = int(c.get("base_channels", 32))
    if base_channels_override is not None:
        base = base_channels_override
    return {
        "model": str(c.get("model", "UNet2D")),
        "in_channels": int(c.get("in_channels", 4)),
        "base_channels": base,
        "image_norm": str(c.get("image_norm", "robust")),
        "quantile_high": float(c.get("quantile_high", 0.99)),
        "phi_scale": float(c.get("phi_scale", 64.0)),
    }


def build_model(cfg: dict) -> teu.UNet2D:
    if cfg["model"] != "UNet2D":
        raise ValueError(f"Expected UNet2D checkpoint, got {cfg['model']!r}")
    return teu.UNet2D(in_channels=cfg["in_channels"], base=cfg["base_channels"])


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
) -> tuple[float, float, int]:
    ds = teu.FiverErrorDataset(
        eval_dir,
        split,
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
    it = tqdm(loader, desc=f"eval {split}", unit="batch", disable=not show_progress)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(x)
        sum_mse += float(teu.masked_mse(pred, y, mask))
        sum_l1 += float(teu.masked_l1(pred, y, mask))
        n += 1
    return sum_mse / max(n, 1), sum_l1 / max(n, 1), len(ds)


@torch.no_grad()
def infer_slices(
    fp: Path,
    model: torch.nn.Module,
    ds: teu.FiverErrorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = ds[ds.paths.index(fp)]
    with np.load(fp) as data:
        image = np.asarray(data["image"], dtype=np.float32)
        warped = np.asarray(data["warped"], dtype=np.float32)
        phi_true = np.asarray(data["phi_true"], dtype=np.float32)
        phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
        err_true = np.asarray(data["error_map"], dtype=np.float32)

    x = batch["x"].unsqueeze(0).to(device)
    pred = model(x).squeeze(0).squeeze(0).cpu().numpy()

    return (
        image,
        warped,
        phi_magnitude_2d(phi_true),
        phi_magnitude_2d(phi_pred),
        err_true,
        pred,
    )


def format_main_title(arrangement: str) -> str:
    if arrangement == "random":
        return f"{PRODUCT_NAME} · Test samples | random selection"
    return f"{PRODUCT_NAME} · Test samples | ranked by {RANK_METRIC_LABEL}"


def format_subtitle(nrows: int) -> str | None:
    if nrows > 1:
        return f"{nrows} slices"
    return None


def format_row_side_label(fp: Path, tag: str, mean_err: float) -> str:
    stem = fp.stem
    line1 = f"[{tag}] {stem}" if tag else stem
    line2 = f"{RANK_METRIC_LABEL} = {mean_err:.3f} {DISPLACEMENT_UNIT}"
    return f"{line1}\n{line2}"


def add_row_side_label(ax, text: str) -> None:
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


def set_column_headers(axes_row) -> None:
    for ax, title in zip(axes_row, COLUMN_TITLES):
        ax.set_title(title, fontsize=9, pad=4)


@torch.no_grad()
def plot_samples_grid(
    items: list[tuple[Path, str, float]],
    model: torch.nn.Module,
    ds: teu.FiverErrorDataset,
    device: torch.device,
    save_path: Path | None,
    no_show: bool,
    err_percentile: float,
    arrangement: str,
    *,
    show_progress: bool,
    phi_percentile: float,
) -> None:
    model.eval()
    rows_data: list[tuple[Path, str, float, tuple[np.ndarray, ...]]] = []

    it = tqdm(items, desc=f"plot {arrangement}", unit="slice", disable=not show_progress)
    for fp, tag, mean_err in it:
        panels = infer_slices(fp, model, ds, device)
        rows_data.append((fp, tag, mean_err, panels))

    all_err = np.concatenate([r[3][4].ravel() for r in rows_data] + [r[3][5].ravel() for r in rows_data])
    err_v = float(np.percentile(all_err, err_percentile))
    if err_v <= 0:
        err_v = 1e-6

    phi_slices = [r[3][2] for r in rows_data] + [r[3][3] for r in rows_data]
    phi_v = float(np.percentile(np.concatenate([p.ravel() for p in phi_slices]), phi_percentile))
    if phi_v <= 0:
        phi_v = 1e-6

    nrows = len(rows_data)
    ncol = 6
    fig, axes = plt.subplots(nrows, ncol, figsize=(3.0 * ncol, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    cbar_phi = f"displacement ({DISPLACEMENT_UNIT})"
    cbar_err = f"error ({DISPLACEMENT_UNIT})"

    for row, (fp, tag, mean_err, panels) in enumerate(rows_data):
        subj, atlas, phi_a, phi_b, err_gt, err_pred = panels
        images: list[tuple[np.ndarray, str, float | None, float | None, str | None]] = [
            (subj, "gray", None, None, None),
            (atlas, "gray", None, None, None),
            (phi_a, "hot", 0.0, phi_v, cbar_phi),
            (phi_b, "hot", 0.0, phi_v, cbar_phi),
            (err_gt, "hot", 0.0, err_v, cbar_err),
            (err_pred, "hot", 0.0, err_v, cbar_err),
        ]
        for col, (img, cmap, vmin, vmax, cbar_label) in enumerate(images):
            ax = axes[row, col]
            if vmin is None:
                ax.imshow(img, cmap=cmap, aspect="equal")
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                if cbar_label:
                    cbar.set_label(cbar_label, fontsize=8)
                cbar.ax.tick_params(labelsize=7)
            ax.axis("off")
        if row == 0:
            set_column_headers(axes[row])
        add_row_side_label(axes[row, 0], format_row_side_label(fp, tag, mean_err))

    main_title = format_main_title(arrangement)
    sub = format_subtitle(nrows)
    suptitle = f"{main_title}\n{sub}" if sub else main_title

    left = 0.11 if nrows > 1 else 0.08
    fig.tight_layout(rect=(left, 0.02, 1, 0.97), pad=0.35, h_pad=0.5, w_pad=0.15)
    fig.subplots_adjust(top=0.93)
    fig.suptitle(suptitle, fontsize=11, y=0.985)

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
    ax.axvline(epochs[best_i], color="0.5", linestyle="--", linewidth=0.8, label=f"best val (ep {epochs[best_i]})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("slice MSE")
    ax.set_title(f"Training vs validation ({run_label})")
    ax.grid(True, alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(epochs, val_l1, label=f"val L1 ({DISPLACEMENT_UNIT})", color="C2", marker=".", markersize=3)
    ax_r.set_ylabel(f"L1 ({DISPLACEMENT_UNIT})")
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
        description="Evaluate 2D error-map U-Net on UniGrad Synth fiver npz slices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run-path",
        type=Path,
        default=Path("assets/runs/regression/unigrad-synth/2d/error_unet_run1"),
    )
    p.add_argument("--eval-dir", type=Path, default=Path("datasets/error-map/unigrad-synth/ixi_2d_fiver"))
    p.add_argument("--eval-split", type=str, default="Test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-random", type=int, default=3)
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--phi-percentile", type=float, default=99.0)
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

    test_mse, test_l1, n_test = evaluate_split(
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
        f"{args.eval_split} ({n_test} slices): MSE = {test_mse:.6f}  "
        f"L1 = {test_l1:.6f} {DISPLACEMENT_UNIT}"
    )
    metrics_out = {
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(eval_dir.resolve()),
        "eval_split": args.eval_split,
        "n_slices": n_test,
        "masked_mse": test_mse,
        "masked_l1": test_l1,
        "unit": DISPLACEMENT_UNIT,
        "config": cfg,
    }
    (run_path / "test_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    test_dir = eval_dir / args.eval_split
    test_files = sorted(test_dir.glob(teu.FIVER_GLOB)) if test_dir.is_dir() else []
    if not test_files:
        print(f"WARNING: no fivers under {test_dir}", file=sys.stderr)
        return 0

    ds_test = teu.FiverErrorDataset(
        eval_dir,
        args.eval_split,
        image_norm=cfg["image_norm"],
        quantile_high=cfg["quantile_high"],
        phi_scale=cfg["phi_scale"],
    )

    rng = random.Random(args.seed)
    if args.num_random > 0:
        k = min(args.num_random, len(test_files))
        picked = rng.sample(test_files, k)
        items = [(p, "", mean_error_map_volume(p)) for p in picked]
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
            phi_percentile=args.phi_percentile,
        )

    ranked = select_easy_normal_hard_by_mean_error(test_files, show_progress=show_p)
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
        phi_percentile=args.phi_percentile,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
