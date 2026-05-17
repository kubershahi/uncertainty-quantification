#!/usr/bin/env python3
"""
Evaluate a trained 3D error-map U-Net on UniGrad IO ``.npz`` volumes.

- ``--eval-dir``: root with ``Test/`` (and optional ``Val/``) from ``create_unigrad_io_data.py``.
- ``--run-path``: training run with ``best_model.pt`` and ``metrics.csv``.

Writes under ``run-path``: ``training_curves.png``, ``test_metrics.json``,
``test_error_pred_random.png``, ``test_error_pred_easy_normal_hard.png`` (min/median/max
mean error_map).

Example:
  python experiments/regression/eval_error_map_unet.py --run-path assets/runs/error_map_unet_3d --eval-dir datasets/IXI_unigrad_io --no-show
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

import train_error_map_unet as teu

CHECKPOINT_FILENAME = "best_model.pt"

DISPLACEMENT_UNIT = "voxels"

_COL_TITLES = (
    "subject",
    "atlas (target)",
    r"$\|\phi_{\mathrm{pred}}\|$",
    f"error GT ({DISPLACEMENT_UNIT})",
    f"error pred ({DISPLACEMENT_UNIT})",
)


def phi_magnitude_slice(phi: np.ndarray, slice_z: int) -> np.ndarray:
    mag = np.sqrt(np.sum(phi.astype(np.float64) ** 2, axis=0))
    return mag[slice_z]


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
    it = tqdm(files, desc="rank volumes", unit="file", disable=not show_progress)
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
        "model": str(c.get("model", "UNet3D")),
        "in_channels": int(c.get("in_channels", 5)),
        "base_channels": base,
        "image_norm": str(c.get("image_norm", "robust")),
        "quantile_high": float(c.get("quantile_high", 0.99)),
        "phi_scale": float(c.get("phi_scale", 64.0)),
    }


def build_model(cfg: dict) -> torch.nn.Module:
    if cfg["model"] != "UNet3D":
        raise ValueError(f"Unsupported checkpoint model {cfg['model']!r}; expected UNet3D")
    return teu.UNet3D(in_channels=cfg["in_channels"], base=cfg["base_channels"])


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
    ds = teu.UniGradIOErrorDataset(
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
def infer_and_slice(
    fp: Path,
    model: torch.nn.Module,
    ds: teu.UniGradIOErrorDataset,
    device: torch.device,
    slice_z: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = ds.paths.index(fp)
    batch = ds[idx]
    with np.load(fp) as data:
        source = teu.volume_hw_d_to_dhw(np.asarray(data["source"]))
        target = teu.volume_hw_d_to_dhw(np.asarray(data["target"]))
        phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
        err_true = np.asarray(data["error_map"], dtype=np.float32)

    z = slice_z if slice_z is not None else err_true.shape[0] // 2
    z = int(np.clip(z, 0, err_true.shape[0] - 1))

    x = batch["x"].unsqueeze(0).to(device)
    pred = model(x).squeeze(0).squeeze(0).cpu().numpy()

    return (
        source[:, :, z],
        target[:, :, z],
        phi_magnitude_slice(phi_pred, z),
        err_true[z],
        pred[z],
    )


@torch.no_grad()
def plot_io_samples_grid(
    items: list[tuple[Path, str, float]],
    model: torch.nn.Module,
    ds: teu.UniGradIOErrorDataset,
    device: torch.device,
    save_path: Path | None,
    no_show: bool,
    err_percentile: float,
    split_title: str,
    arrangement_detail: str,
    *,
    show_progress: bool,
    slice_z: int | None,
) -> None:
    model.eval()
    rows: list[tuple[Path, str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    it = tqdm(items, desc=f"plot {split_title}", unit="subject", disable=not show_progress)
    for fp, tag, mean_err in it:
        subj, atlas, mag_p, err_gt, err_pred = infer_and_slice(fp, model, ds, device, slice_z)
        rows.append((fp, tag, mean_err, subj, atlas, mag_p, err_gt, err_pred))

    all_err = np.concatenate([r[6].ravel() for r in rows] + [r[7].ravel() for r in rows])
    err_v = float(np.percentile(all_err, err_percentile))
    if err_v <= 0:
        err_v = 1e-6

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 5, figsize=(16, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for row, (fp, tag, mean_err, subj, atlas, mag_p, err_gt, err_pred) in enumerate(rows):
        images = [
            (subj, "gray", None, None),
            (atlas, "gray", None, None),
            (mag_p, "hot", 0.0, None),
            (err_gt, "hot", 0.0, err_v),
            (err_pred, "hot", 0.0, err_v),
        ]
        for col, (img, cmap, vmin, vmax) in enumerate(images):
            ax = axes[row, col]
            if vmax is None and vmin is None:
                ax.imshow(img, cmap=cmap)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.axis("off")
            if row == 0:
                ax.set_title(_COL_TITLES[col], fontsize=9)
            if col == 0:
                rank_line = f"[{tag}] " if tag else ""
                ax.set_ylabel(
                    f"{rank_line}{fp.stem}\nmean(error_map)={mean_err:.3f} {DISPLACEMENT_UNIT}",
                    fontsize=8,
                )

    with np.load(rows[0][0]) as zdata:
        d = int(np.asarray(zdata["error_map"]).shape[0])
    z_show = slice_z if slice_z is not None else d // 2
    z_show = int(np.clip(z_show, 0, d - 1))
    fig.suptitle(
        f"{split_title} {arrangement_detail} · axial z={z_show} · err vmax={err_v:.3g} {DISPLACEMENT_UNIT}",
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
    ax.set_ylabel("volume MSE")
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
        description="Evaluate 3D error-map U-Net on UniGrad IO npz volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run-path", type=Path, required=True)
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("datasets/IXI_unigrad_io"),
    )
    p.add_argument("--eval-split", type=str, default="Test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-random", type=int, default=3, help="Random Test volumes to plot.")
    p.add_argument("--slice-index", type=int, default=None, metavar="Z")
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (enabled by default).",
    )
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--no-training-curves", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_path = Path(args.run_path).resolve()
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
        args.eval_dir,
        args.eval_split,
        cfg,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        show_progress=show_p,
    )
    print(
        f"{args.eval_split} ({n_test} volumes): MSE = {test_mse:.6f}  "
        f"L1 = {test_l1:.6f} {DISPLACEMENT_UNIT}"
    )
    metrics_out = {
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(Path(args.eval_dir).resolve()),
        "eval_split": args.eval_split,
        "n_volumes": n_test,
        "masked_mse": test_mse,
        "masked_l1": test_l1,
        "config": cfg,
    }
    (run_path / "test_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    test_dir = Path(args.eval_dir) / args.eval_split
    test_files = sorted(test_dir.glob(teu.DATA_GLOB)) if test_dir.is_dir() else []
    if not test_files:
        print(f"WARNING: no volumes under {test_dir}", file=sys.stderr)
        return 0

    ds_test = teu.UniGradIOErrorDataset(
        args.eval_dir,
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
        plot_io_samples_grid(
            items,
            model,
            ds_test,
            device,
            run_path / "test_error_pred_random.png",
            args.no_show,
            args.err_percentile,
            args.eval_split,
            f"(random {k})",
            show_progress=show_p,
            slice_z=args.slice_index,
        )

    ranked = select_easy_normal_hard_by_mean_error(test_files, show_progress=show_p)
    plot_io_samples_grid(
        ranked,
        model,
        ds_test,
        device,
        run_path / "test_error_pred_easy_normal_hard.png",
        args.no_show,
        args.err_percentile,
        args.eval_split,
        "(easy / normal / hard by mean error_map)",
        show_progress=show_p,
        slice_z=args.slice_index,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
