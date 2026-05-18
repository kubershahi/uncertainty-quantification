#!/usr/bin/env python3
"""
Evaluate a trained 3D IO error-map U-Net on UniGrad IO data from ``create_unigrad_io_data.py``.

Default writes under ``--run-path``:

- ``training_curves.png`` (from ``metrics.csv``)
- ``test_metrics.json``
- ``test_error_pred_random.png`` — random Test subjects, 5 columns per row (see below)
- ``test_error_pred_easy_normal_hard.png`` — lowest / median / highest mean ``error_map`` in atlas mask

Each QC figure is one axial slice (default z = depth/4) with shared color scales across rows:
subject source, atlas target, |φ_pred|, GT error map, U-Net predicted error map (voxels).

Use ``--curves-only`` to skip Test eval (no GPU). Checkpoints from ``torch.compile`` training load via ``_orig_mod`` key fix.

Example:
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run1 --eval-dir datasets/IXI_unigrad_io --no-show

python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run1 --eval-dir datasets/IXI_unigrad_io --curves-only --no-show

python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run1 --eval-dir datasets/IXI_unigrad_io --curves-only --no-show --val-plot-min-epoch 5
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

import train_unigrad_io_unet as teu

CHECKPOINT_FILENAME = "best_model.pt"
PRODUCT_NAME = "UniGrad IO"
DISPLACEMENT_UNIT = "voxels"
RANK_METRIC_LABEL = "mean(error_map)"

COLUMN_TITLES = (
    "subject (source)",
    "atlas (target)",
    r"zero-shot ($\|\phi_{\mathrm{pred}}\|$)",
    f"error GT ({DISPLACEMENT_UNIT})",
    f"error pred ({DISPLACEMENT_UNIT})",
)


def phi_magnitude_slice(phi: np.ndarray, slice_z: int) -> np.ndarray:
    mag = np.sqrt(np.sum(phi.astype(np.float64) ** 2, axis=0))
    return mag[slice_z]


def mean_error_map_volume(npz_path: Path, mask_dhw: np.ndarray | None = None) -> float:
    with np.load(npz_path) as z:
        err = np.asarray(z["error_map"])
    if mask_dhw is not None:
        if mask_dhw.shape != err.shape:
            raise ValueError(f"{npz_path.name}: mask {mask_dhw.shape} vs error_map {err.shape}")
        return float(np.mean(err[mask_dhw]))
    return float(np.mean(err))


def select_easy_normal_hard_by_mean_error(
    files: list[Path],
    *,
    mask_dhw: np.ndarray | None,
    show_progress: bool,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    it = tqdm(files, desc="rank volumes", unit="file", disable=not show_progress)
    scored = [(fp, mean_error_map_volume(fp, mask_dhw)) for fp in it]
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


def build_model(cfg: dict) -> teu.UNet3D:
    if cfg["model"] != "UNet3D":
        raise ValueError(f"Expected UNet3D checkpoint, got {cfg['model']!r}")
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
def infer_slices(
    fp: Path,
    model: torch.nn.Module,
    ds: teu.UniGradIOErrorDataset,
    device: torch.device,
    slice_z: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    batch = ds[ds.paths.index(fp)]
    with np.load(fp) as data:
        source_hw = np.asarray(data["source"])
        if "target" in data.files:
            atlas_hw = np.asarray(data["target"])
        else:
            atlas_hw = ds.atlas_target_hw_d
        phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
        err_true = np.asarray(data["error_map"], dtype=np.float32)

    depth = int(err_true.shape[0])
    z = teu.default_slice_index(depth) if slice_z is None else int(slice_z)
    z = int(np.clip(z, 0, depth - 1))

    x = batch["x"].unsqueeze(0).to(device)
    pred = model(x).squeeze(0).squeeze(0).cpu().numpy()

    return (
        source_hw[:, :, z],
        atlas_hw[:, :, z],
        phi_magnitude_slice(phi_pred, z),
        err_true[z],
        pred[z],
        z,
        depth,
    )


def format_run_caption(run_label: str) -> str:
    return f"({run_label})"


def format_test_qc_title(arrangement: str, run_label: str) -> str:
    if arrangement == "random":
        line1 = f"{PRODUCT_NAME} Test QC | random selection"
    else:
        line1 = f"{PRODUCT_NAME} Test QC | ranked by {RANK_METRIC_LABEL}"
    return f"{line1}\n{format_run_caption(run_label)}"


def format_training_curves_title(run_label: str) -> str:
    return f"{PRODUCT_NAME} | Train vs Val\n{format_run_caption(run_label)}"


def format_slice_caption(z: int, depth: int, nrows: int) -> str:
    line = f"axial slice z = {z} / {depth - 1}"
    if nrows > 1:
        line += f"  ·  {nrows} subjects"
    return line


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
    ds: teu.UniGradIOErrorDataset,
    device: torch.device,
    save_path: Path | None,
    no_show: bool,
    err_percentile: float,
    arrangement: str,
    run_label: str,
    *,
    show_progress: bool,
    slice_z: int | None,
    phi_percentile: float,
) -> None:
    model.eval()
    rows_data: list[tuple[Path, str, float, tuple[np.ndarray, ...]]] = []
    z_show: int | None = None
    depth: int | None = None

    it = tqdm(items, desc=f"plot {arrangement}", unit="subject", disable=not show_progress)
    for fp, tag, mean_err in it:
        subj, atlas, phi_a, err_gt, err_pred, z_i, d_i = infer_slices(
            fp, model, ds, device, slice_z
        )
        if z_show is None:
            z_show, depth = z_i, d_i
        rows_data.append((fp, tag, mean_err, (subj, atlas, phi_a, err_gt, err_pred)))

    all_err = np.concatenate([r[3][3].ravel() for r in rows_data] + [r[3][4].ravel() for r in rows_data])
    err_v = float(np.percentile(all_err, err_percentile))
    if err_v <= 0:
        err_v = 1e-6

    phi_slices = [r[3][2] for r in rows_data]
    phi_v = float(np.percentile(np.concatenate([p.ravel() for p in phi_slices]), phi_percentile))
    if phi_v <= 0:
        phi_v = 1e-6

    nrows = len(rows_data)
    ncol = 5
    fig, axes = plt.subplots(nrows, ncol, figsize=(3.0 * ncol, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    cbar_phi = f"displacement ({DISPLACEMENT_UNIT})"
    cbar_err = f"‖Δφ‖ ({DISPLACEMENT_UNIT})"

    for row, (fp, tag, mean_err, panels) in enumerate(rows_data):
        subj, atlas, phi_a, err_gt, err_pred = panels
        images: list[tuple[np.ndarray, str, float | None, float | None, str | None]] = [
            (subj, "gray", None, None, None),
            (atlas, "gray", None, None, None),
            (phi_a, "hot", 0.0, phi_v, cbar_phi),
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

    assert z_show is not None and depth is not None
    suptitle = (
        f"{format_test_qc_title(arrangement, run_label)}\n"
        f"{format_slice_caption(z_show, depth, nrows)}"
    )

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


def resolve_val_plot_min_epoch(
    epochs: list[int],
    run_path: Path,
    *,
    val_plot_min_epoch: int | None,
    val_plot_min_frac: float,
) -> int:
    if val_plot_min_epoch is not None and val_plot_min_epoch > 0:
        return min(val_plot_min_epoch, max(epochs) if epochs else 1)
    cfg_path = run_path / "run_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("val_start_epoch"):
                return min(int(cfg["val_start_epoch"]), max(epochs) if epochs else 1)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if val_plot_min_frac > 0.0 and epochs:
        return min(max(1, int(np.ceil(val_plot_min_frac * max(epochs)))), max(epochs))
    return 1


def plot_training_curves_from_csv(
    metrics_csv: Path,
    save_path: Path | None,
    no_show: bool,
    run_label: str,
    *,
    run_path: Path | None = None,
    val_plot_min_epoch: int | None = None,
    val_plot_min_frac: float = 0.1,
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

    plot_root = run_path if run_path is not None else (metrics_csv.parent if metrics_csv else Path("."))
    vmin_ep = resolve_val_plot_min_epoch(
        epochs,
        plot_root,
        val_plot_min_epoch=val_plot_min_epoch,
        val_plot_min_frac=val_plot_min_frac,
    )
    ep_arr = np.asarray(epochs, dtype=int)
    val_mse_arr = np.asarray(val_mse, dtype=float)
    val_l1_arr = np.asarray(val_l1, dtype=float)
    val_mask = (ep_arr >= vmin_ep) & np.isfinite(val_mse_arr)
    val_ep_plot = ep_arr[val_mask]
    val_mse_plot = val_mse_arr[val_mask]
    val_l1_plot = val_l1_arr[val_mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_mse, label="train MSE", color="C0", marker=".", markersize=3)
    y_for_scale: list[float] = list(train_mse)
    if val_ep_plot.size:
        ax.plot(val_ep_plot, val_mse_plot, label="val MSE", color="C1", marker=".", markersize=3)
        ax.plot(
            val_ep_plot,
            val_l1_plot,
            label=f"val L1 ({DISPLACEMENT_UNIT})",
            color="C2",
            marker=".",
            markersize=3,
        )
        y_for_scale.extend(val_mse_plot.tolist())
        y_for_scale.extend(val_l1_plot.tolist())
        best_i = int(np.argmin(val_mse_plot))
        ax.axvline(
            val_ep_plot[best_i],
            color="0.5",
            linestyle="--",
            linewidth=0.8,
            label=f"best val (ep {val_ep_plot[best_i]})",
        )
    else:
        ax.plot([], [], label="val MSE", color="C1", marker=".", markersize=3)
        ax.plot([], [], label=f"val L1 ({DISPLACEMENT_UNIT})", color="C2", marker=".", markersize=3)

    y_hi = float(np.nanpercentile(np.asarray(y_for_scale, dtype=float), 99))
    if y_hi > 0:
        ax.set_ylim(0.0, y_hi * 1.05)

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"volume loss (MSE; L1 in {DISPLACEMENT_UNIT})")
    ax.set_title(format_training_curves_title(run_label), fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
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
        description="Training curves + Test metrics and QC figures for IO error-map U-Net.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--run-path",
        type=Path,
        default=Path("assets/runs/3d/unigrad-io/error_unet_run1"),
    )
    p.add_argument("--eval-dir", type=Path, default=Path("datasets/IXI_unigrad_io"))
    p.add_argument("--eval-split", type=str, default="Test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-random", type=int, default=3)
    p.add_argument("--slice-index", type=int, default=None, metavar="Z")
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--phi-percentile", type=float, default=99.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--no-training-curves", action="store_true")
    p.add_argument(
        "--curves-only",
        action="store_true",
        help="Only plot training_curves.png; skip Test metrics and figures.",
    )
    p.add_argument(
        "--val-plot-min-epoch",
        type=int,
        default=None,
        metavar="N",
        help="Plot val curves only from epoch N (default: run_config val_start_epoch or --val-plot-min-frac).",
    )
    p.add_argument(
        "--val-plot-min-frac",
        type=float,
        default=0.1,
        help="If val plot min epoch unset: hide val before ceil(frac * epochs) (0 = show all).",
    )
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

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_path / "metrics.csv"

    if not args.no_training_curves:
        if metrics_csv.is_file():
            if not plot_training_curves_from_csv(
                metrics_csv,
                run_path / "training_curves.png",
                args.no_show,
                run_path.name,
                run_path=run_path,
                val_plot_min_epoch=args.val_plot_min_epoch,
                val_plot_min_frac=args.val_plot_min_frac,
            ):
                print(f"ERROR: failed to plot from {metrics_csv}", file=sys.stderr)
                return 2
        else:
            print(f"WARNING: no metrics.csv at {metrics_csv}", file=sys.stderr)

    if args.curves_only:
        return 0

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
    teu.load_checkpoint_state_dict(model, ckpt["model_state"])
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
        f"{args.eval_split} ({n_test} volumes): MSE = {test_mse:.6f}  "
        f"L1 = {test_l1:.6f} {DISPLACEMENT_UNIT}"
    )
    metrics_out = {
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(eval_dir.resolve()),
        "eval_split": args.eval_split,
        "atlas_mask_file": teu.ATLAS_MASK_FILENAME,
        "n_volumes": n_test,
        "masked_mse": test_mse,
        "masked_l1": test_l1,
        "unit": DISPLACEMENT_UNIT,
        "config": cfg,
    }
    (run_path / "test_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    test_dir = eval_dir / args.eval_split
    test_files = sorted(test_dir.glob(teu.DATA_GLOB)) if test_dir.is_dir() else []
    if not test_files:
        print(f"WARNING: no volumes under {test_dir}", file=sys.stderr)
        return 0

    ds_test = teu.UniGradIOErrorDataset(
        eval_dir,
        args.eval_split,
        image_norm=cfg["image_norm"],
        quantile_high=cfg["quantile_high"],
        phi_scale=cfg["phi_scale"],
    )
    eval_mask = ds_test.valid_mask_dhw

    rng = random.Random(args.seed)
    if args.num_random > 0:
        k = min(args.num_random, len(test_files))
        picked = rng.sample(test_files, k)
        items = [(p, "", mean_error_map_volume(p, eval_mask)) for p in picked]
        plot_samples_grid(
            items,
            model,
            ds_test,
            device,
            run_path / "test_error_pred_random.png",
            args.no_show,
            args.err_percentile,
            "random",
            run_path.name,
            show_progress=show_p,
            slice_z=args.slice_index,
            phi_percentile=args.phi_percentile,
        )

    ranked = select_easy_normal_hard_by_mean_error(
        test_files, mask_dhw=eval_mask, show_progress=show_p
    )
    plot_samples_grid(
        ranked,
        model,
        ds_test,
        device,
        run_path / "test_error_pred_easy_normal_hard.png",
        args.no_show,
        args.err_percentile,
        "easy_normal_hard",
        run_path.name,
        show_progress=show_p,
        slice_z=args.slice_index,
        phi_percentile=args.phi_percentile,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
