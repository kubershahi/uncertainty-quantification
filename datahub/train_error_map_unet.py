#!/usr/bin/env python3
"""
Train a 2D regression U-Net to predict per-pixel registration error magnitude.

Fiver npz (see create_unigrad_data.py):
  image, warped, phi_pred, error_map, valid_mask, ...

Input channels (default): fixed image, warped image, phi_pred (2 components) -> 4 channels.
Target: error_map (scalar ‖φ_true − φ_pred‖ per pixel).

Loss is averaged over pixels where valid_mask is True.

Per-epoch train/val metrics are appended to ``metrics.csv`` under ``--out-dir`` (for plotting).

Progress: ``tqdm`` bars for epochs, train batches, and val batches (disable with ``--no-progress``).

MSE is masked to the interior ``valid_mask`` only. Optional ``--smooth-weight`` adds total-variation
style smoothing on edges that touch the exterior or boundary (not on two fully interior edges)
so predictions stay calm where there is no supervision.

Default ``--image-norm robust``: per-slice min + quantile hi → [0,1] for ``image``/``warped`` (safe across datasets).
If inputs are already scaled consistently (e.g. IXI fivers), optionally ``--image-norm none``.

Example:
  python train_error_map_unet.py --data-dir ./data/IXI_2D_unigrad_fiver \\
    --epochs 50 --batch-size 8 --out-dir ./runs/error_unet_run1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_fiver_paths(root: Path, split: str) -> list[Path]:
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(f"Missing split directory: {d}")
    files = sorted(d.glob("*_fiver.npz"))
    if not files:
        raise FileNotFoundError(f"No *_fiver.npz under {d}")
    return files


class FiverErrorDataset(Dataset):
    """Loads *_fiver.npz; builds (4, H, W) input and (1, H, W) target."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        image_norm: str = "robust",
        quantile_high: float = 0.99,
        phi_scale: float = 64.0,
    ):
        self.paths = collect_fiver_paths(root, split)
        if image_norm not in ("none", "robust"):
            raise ValueError("image_norm must be 'none' or 'robust'")
        self.image_norm = image_norm
        self.quantile_high = quantile_high
        self.phi_scale = float(phi_scale)

    def __len__(self) -> int:
        return len(self.paths)

    def _norm_image(self, x: np.ndarray) -> np.ndarray:
        """``none``: use stored intensities (float32). ``robust``: per-slice min + quantile hi -> [0,1]."""
        x = x.astype(np.float32)
        if self.image_norm == "none":
            return x
        lo = float(np.min(x))
        hi = float(np.quantile(x.reshape(-1), self.quantile_high))
        if hi <= lo:
            hi = lo + 1e-5
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        with np.load(path) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            warped = np.asarray(data["warped"], dtype=np.float32)
            phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
            err = np.asarray(data["error_map"], dtype=np.float32)
            if "valid_mask" in data.files:
                mask = np.asarray(data["valid_mask"], dtype=np.bool_)
            else:
                mask = np.ones(image.shape[:2], dtype=np.bool_)

        image_n = self._norm_image(image)
        warped_n = self._norm_image(warped)
        phi_n = phi_pred / self.phi_scale

        x = np.concatenate(
            [
                image_n[None, ...],
                warped_n[None, ...],
                phi_n,
            ],
            axis=0,
        )
        y = err[None, ...]

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "path": str(path),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B,1,H,W); mask: (B,H,W) bool."""
    m = mask.unsqueeze(1).float()
    diff = (pred - target) ** 2 * m
    denom = m.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(1).float()
    diff = torch.abs(pred - target) * m
    denom = m.sum().clamp_min(1.0)
    return diff.sum() / denom


def masked_mse_plus_boundary_smoothness(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_valid: torch.Tensor,
    *,
    smooth_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interior MSE + optional smoothness on non-interior edges.

    ``mask_valid`` is True on supervised interior pixels. Smoothness is mean |∇pred| on
    horizontal/vertical edges where it is not the case that both endpoints are valid.
    """
    mse = masked_mse(pred, target, mask_valid)
    if smooth_weight <= 0.0:
        z = torch.zeros_like(mse)
        return mse, z, mse

    m = mask_valid
    if m.dim() == 4:
        m = m.squeeze(1)
    m = m.float()

    gx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    m1, m2 = m[:, :-1, :], m[:, 1:, :]
    w_x = 1.0 - (m1 * m2)
    tv_x = (gx.abs() * w_x.unsqueeze(1)).sum() / w_x.sum().clamp_min(1.0)

    gy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    m1, m2 = m[:, :, :-1], m[:, :, 1:]
    w_y = 1.0 - (m1 * m2)
    tv_y = (gy.abs() * w_y.unsqueeze(1)).sum() / w_y.sum().clamp_min(1.0)

    smooth = 0.5 * (tv_x + tv_y)
    total = mse + smooth_weight * smooth
    return mse, smooth, total


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """Classic U-Net; in_channels=4 (image, warped, phi_xy), out_channels=1."""

    def __init__(self, in_channels: int = 4, base: int = 32) -> None:
        super().__init__()
        b = base
        self.down1 = DoubleConv(in_channels, b)
        self.down2 = DoubleConv(b, b * 2)
        self.down3 = DoubleConv(b * 2, b * 4)
        self.down4 = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bot = DoubleConv(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.conv4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.conv3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.conv2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.conv1 = DoubleConv(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)
        c4 = self.down4(p3)
        p4 = self.pool(c4)
        x5 = self.bot(p4)
        x = self.up4(x5)
        x = self.conv4(torch.cat([x, c4], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, c3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, c2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, c1], dim=1))
        return self.out(x)


def collate_batch(samples: list[dict]) -> dict:
    return {
        "x": torch.stack([s["x"] for s in samples], dim=0),
        "y": torch.stack([s["y"] for s in samples], dim=0),
        "mask": torch.stack([s["mask"] for s in samples], dim=0),
        "path": [s["path"] for s in samples],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    show_progress: bool = True,
    desc: str = "val",
) -> tuple[float, float]:
    model.eval()
    sum_mse = 0.0
    sum_l1 = 0.0
    n = 0
    it = tqdm(loader, desc=desc, leave=False, unit="batch", disable=not show_progress)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            mse = masked_mse(pred, y, mask)
            l1 = masked_l1(pred, y, mask)
        sum_mse += float(mse)
        sum_l1 += float(l1)
        n += 1
    return sum_mse / max(n, 1), sum_l1 / max(n, 1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    smooth_weight: float,
    show_progress: bool = True,
    epoch: int = 1,
    total_epochs: int = 1,
) -> tuple[float, float, float]:
    model.train()
    sum_mse = 0.0
    sum_smooth = 0.0
    sum_total = 0.0
    steps = 0
    desc = f"train {epoch}/{total_epochs}"
    it = tqdm(loader, desc=desc, leave=False, unit="batch", disable=not show_progress)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            mse, smooth, loss = masked_mse_plus_boundary_smoothness(
                pred, y, mask, smooth_weight=smooth_weight
            )
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sum_mse += float(mse)
        sum_smooth += float(smooth)
        sum_total += float(loss)
        steps += 1
        if show_progress:
            it.set_postfix(mse=float(mse), total=float(loss))
    n = max(steps, 1)
    return sum_mse / n, sum_smooth / n, sum_total / n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net to predict registration error_map from fiver npz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/IXI_2D_unigrad_fiver"),
        help="Root with Train/Val (and optional Test) subfolders.",
    )
    p.add_argument("--train-split", type=str, default="Train")
    p.add_argument("--val-split", type=str, default="Val")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument(
        "--image-norm",
        type=str,
        default="robust",
        choices=["none", "robust"],
        help="robust (default): per-slice min + quantile-hi rescale image/warped to [0,1]. "
        "none: use raw float32 from npz (skip rescaling).",
    )
    p.add_argument(
        "--quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile for --image-norm robust (ignored for none).",
    )
    p.add_argument(
        "--phi-scale",
        type=float,
        default=64.0,
        help="Divide phi_pred by this (pixels). Must be > 0.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("./runs/error_map_unet"))
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA.")
    p.add_argument(
        "--smooth-weight",
        type=float,
        default=0.05,
        metavar="W",
        help="Weight for boundary/exterior TV smoothness (0 to disable). "
        "Not applied to edges between two interior (valid_mask) pixels.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (plain log lines only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phi_scale <= 0:
        raise ValueError("--phi-scale must be positive.")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_ds = FiverErrorDataset(
        args.data_dir,
        args.train_split,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
        phi_scale=args.phi_scale,
    )
    val_ds = FiverErrorDataset(
        args.data_dir,
        args.val_split,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
        phi_scale=args.phi_scale,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = UNet2D(in_channels=4, base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_dir": str(args.data_dir.resolve()),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train_files": len(train_ds),
        "val_files": len(val_ds),
        "phi_scale": args.phi_scale,
        "image_norm": args.image_norm,
        "quantile_high": args.quantile_high,
        "base_channels": args.base_channels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "metrics_csv": "metrics.csv",
        "smooth_weight": args.smooth_weight,
        "show_progress": not args.no_progress,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_path = args.out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["epoch", "train_mse", "train_smooth", "train_total", "val_mse", "val_l1", "elapsed_s"]
        )

    best_val = float("inf")
    best_path = args.out_dir / "best_model.pt"

    show_p = not args.no_progress
    t0 = time.time()
    epoch_loop = range(1, args.epochs + 1)
    if show_p:
        epoch_loop = tqdm(
            epoch_loop,
            desc="epoch",
            unit="ep",
            total=args.epochs,
            position=0,
            leave=True,
        )
    for epoch in epoch_loop:
        tr_mse, tr_smooth, tr_total = train_epoch(
            model,
            train_loader,
            opt,
            device,
            use_amp=use_amp,
            scaler=scaler if use_amp else None,
            smooth_weight=args.smooth_weight,
            show_progress=show_p,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_mse, val_l1 = evaluate(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            show_progress=show_p,
            desc=f"val {epoch}/{args.epochs}",
        )
        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_mse={tr_mse:.6f}  train_smooth={tr_smooth:.6f}  train_total={tr_total:.6f}  "
            f"val_mse={val_mse:.6f}  val_l1={val_l1:.6f}  "
            f"elapsed={dt:.1f}s"
        )
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, tr_mse, tr_smooth, tr_total, val_mse, val_l1, dt])
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "val_l1": val_l1,
                    "config": meta,
                },
                best_path,
            )
            print(f"  saved best to {best_path}")

    print(f"Done. Best val MSE={best_val:.6f} -> {best_path}")
    print(f"Metrics log: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
