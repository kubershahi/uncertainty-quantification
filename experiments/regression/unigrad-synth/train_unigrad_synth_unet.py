#!/usr/bin/env python3
"""
Train a 3D U-Net to regress the UniGradICON registration error map on HCP synth NPZ.

Phase II outputs (``create_unigrad_synth_data.py``) under ``Train|Val|Test/*.npz``:
  ``source``, ``moving``, ``u_gt``, ``u_pred`` (3, X, Y, Z), ``u_error_map``, ``source_mask``, …

Model input (5 channels, ``N×C×X×Y×Z``):
  ``[source, moving, u_pred_x / u_scale, u_pred_y / u_scale, u_pred_z / u_scale]``
Target (1 channel): ``‖u_gt − u_pred‖`` per voxel (full 3D magnitude).
Loss / metrics: masked by ``source_mask`` (brain tissue on the source grid).

Ablation (``--mask-u-pred``): OFF by default (raw u_pred channels keep extra-cranial
context). When set, ``u_pred_{x,y,z}`` are zeroed outside ``source_mask`` before the U-Net.

Logging: per-epoch CSV (``metrics.csv``, plotted by the eval script) plus optional
Weights & Biases (``--wandb``). Best checkpoint by validation masked MAE.

Optimizer AdamW; ``ReduceLROnPlateau`` on val MAE; optional early stopping.

Example:
python experiments/regression/unigrad-synth/train_unigrad_synth_unet.py --data-dir datasets/error-map/unigrad-synth/hcp --batch-size 1 --out-dir assets/runs/regression/unigrad-synth/3d/error_unet_run1

Example (ablation, masked u_pred + wandb):
python experiments/regression/unigrad-synth/train_unigrad_synth_unet.py --data-dir datasets/error-map/unigrad-synth/hcp --batch-size 1 --mask-u-pred --wandb --wandb-project unc-quan --wandb-run-name error_unet_run1_masked --out-dir assets/runs/regression/unigrad-synth/3d/error_unet_run1_masked
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_GLOB = "*.npz"
HCP_REQUIRED_KEYS = frozenset(
    {
        "source",
        "moving",
        "u_gt",
        "u_pred",
        "source_mask",
    }
)
DEFAULT_EPOCHS = 50
DEFAULT_EARLY_STOP_PATIENCE = 10
IN_CHANNELS = 5  # source, moving, u_pred_x, u_pred_y, u_pred_z
SPATIAL_MULTIPLE = 16  # 4× MaxPool3d(2) in UNet3D


def pad_spatial_to_multiple(
    arr: np.ndarray,
    *,
    multiple: int = SPATIAL_MULTIPLE,
    is_mask: bool = False,
) -> np.ndarray:
    """Pad trailing 3 spatial dims so each is divisible by ``multiple`` (pad after)."""
    if arr.ndim < 3:
        raise ValueError(f"expected >=3D array, got {arr.shape}")
    spatial = arr.shape[-3:]
    pads = [((multiple - s % multiple) % multiple) for s in spatial]
    if pads == [0, 0, 0]:
        return arr
    pad_width = [(0, 0)] * (arr.ndim - 3) + [(0, pads[0]), (0, pads[1]), (0, pads[2])]
    constant = False if is_mask else 0
    return np.pad(arr, pad_width, mode="constant", constant_values=constant)


def amp_autocast(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except AttributeError:
        return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_hcp_npz_paths(root: Path, split: str) -> list[Path]:
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(f"Missing split directory: {d}")
    files = sorted(d.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No {DATA_GLOB} under {d}")
    return files


class HCPErrorMapDataset(Dataset):
    """HCP Phase-II NPZ → (5, X, Y, Z) input, (1, X, Y, Z) target, source_mask."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        u_scale: float = 64.0,
        mask_u_pred: bool = False,
        image_norm: str = "none",
        quantile_high: float = 0.99,
    ):
        self.paths = collect_hcp_npz_paths(root, split)
        if image_norm not in ("none", "robust"):
            raise ValueError("image_norm must be 'none' or 'robust'")
        if u_scale <= 0:
            raise ValueError("u_scale must be positive")
        self.u_scale = float(u_scale)
        self.mask_u_pred = bool(mask_u_pred)
        self.image_norm = image_norm
        self.quantile_high = float(quantile_high)
        self._validate_first()

    def _validate_first(self) -> None:
        with np.load(self.paths[0]) as data:
            missing = HCP_REQUIRED_KEYS - set(data.files)
            if missing:
                raise KeyError(
                    f"{self.paths[0].name}: missing {sorted(missing)}; "
                    "expected Phase-II HCP keys from create_unigrad_synth_data.py"
                )

    def __len__(self) -> int:
        return len(self.paths)

    def _norm_volume(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.image_norm == "none":
            return x
        lo = float(np.min(x))
        hi = float(np.quantile(x.reshape(-1), self.quantile_high))
        if hi <= lo:
            hi = lo + 1e-5
        x = np.clip(x, lo, hi)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        with np.load(path) as data:
            source = np.asarray(data["source"], dtype=np.float32)
            moving = np.asarray(data["moving"], dtype=np.float32)
            u_gt = np.asarray(data["u_gt"], dtype=np.float32)
            u_pred = np.asarray(data["u_pred"], dtype=np.float32)
            source_mask = np.asarray(data["source_mask"], dtype=np.bool_)
            if "u_error_map" in data.files:
                err = np.asarray(data["u_error_map"], dtype=np.float32)
            else:
                err = np.linalg.norm(
                    u_gt.astype(np.float64) - u_pred.astype(np.float64), axis=0
                ).astype(np.float32)

        if source.ndim != 3 or moving.shape != source.shape:
            raise ValueError(
                f"{path.name}: source/moving must be (X,Y,Z), got {source.shape}/{moving.shape}"
            )
        if u_gt.shape != (3, *source.shape) or u_pred.shape != (3, *source.shape):
            raise ValueError(
                f"{path.name}: u_gt/u_pred must be (3,X,Y,Z), got {u_gt.shape}/{u_pred.shape}"
            )
        if source_mask.shape != source.shape:
            raise ValueError(f"{path.name}: source_mask {source_mask.shape} vs source {source.shape}")

        source_n = self._norm_volume(source)
        moving_n = self._norm_volume(moving)
        u_scaled = (u_pred / self.u_scale).astype(np.float32)  # (3, X, Y, Z)
        if self.mask_u_pred:
            u_scaled = u_scaled * source_mask.astype(np.float32)[None, ...]

        x = np.concatenate([source_n[None], moving_n[None], u_scaled], axis=0)  # (5, X, Y, Z)
        y = err[None, ...]

        # Pad so UNet3D skip-concat shapes match (4 pool levels -> divisible by 16).
        x = pad_spatial_to_multiple(x)
        y = pad_spatial_to_multiple(y)
        source_mask = pad_spatial_to_multiple(source_mask, is_mask=True)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "mask": torch.from_numpy(np.ascontiguousarray(source_mask)),
            "path": str(path),
        }


def _mask_bcxyz(mask: torch.Tensor) -> torch.Tensor:
    """``(N, X, Y, Z)`` bool/any → ``(N, 1, X, Y, Z)`` float."""
    m = mask.float()
    if m.dim() == 4:
        m = m.unsqueeze(1)
    return m


def masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _mask_bcxyz(mask)
    return (loss_map * m).sum() / (m.sum() + 1e-8)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(pred - target), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_l1(pred, target, mask)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask).clamp_min(0.0))


def masked_pearson_r(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson r over masked voxels (per forward call)."""
    m = _mask_bcxyz(mask).bool()
    a = pred[m].float()
    b = target[m].float()
    if a.numel() < 2:
        return pred.new_tensor(0.0)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-8)
    return (a * b).sum() / denom


def masked_regression_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, loss: str
) -> torch.Tensor:
    if loss == "l1":
        return masked_l1(pred, target, mask)
    if loss == "mse":
        return masked_mse(pred, target, mask)
    raise ValueError(f"Unknown loss {loss!r}; expected 'l1' or 'mse'")


def _norm3d(num_channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm (robust to batch size 1 for full 3D volumes)."""
    g = groups
    while g > 1 and num_channels % g != 0:
        g //= 2
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """3D U-Net: 5 in-ch (source, moving, u_pred_xyz) -> 1 out-ch (error map)."""

    def __init__(self, in_channels: int = IN_CHANNELS, base: int = 32) -> None:
        super().__init__()
        b = base
        self.down1 = DoubleConv3d(in_channels, b)
        self.down2 = DoubleConv3d(b, b * 2)
        self.down3 = DoubleConv3d(b * 2, b * 4)
        self.down4 = DoubleConv3d(b * 4, b * 8)
        self.pool = nn.MaxPool3d(2)
        self.bot = DoubleConv3d(b * 8, b * 16)
        self.up4 = nn.ConvTranspose3d(b * 16, b * 8, 2, stride=2)
        self.conv4 = DoubleConv3d(b * 16, b * 8)
        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.conv3 = DoubleConv3d(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.conv2 = DoubleConv3d(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.conv1 = DoubleConv3d(b * 2, b)
        self.out = nn.Conv3d(b, 1, 1)

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


def eager_module(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return eager_module(model).state_dict()


def make_dataloader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kw: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_batch,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    show_progress: bool = True,
    desc: str = "val",
) -> dict[str, float]:
    """Masked MAE / RMSE / Pearson r inside ``source_mask`` (averaged over batches)."""
    model.eval()
    sum_mae = 0.0
    sum_rmse = 0.0
    sum_r = 0.0
    n = 0
    it = tqdm(loader, desc=desc, leave=False, unit="batch", disable=not show_progress)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with amp_autocast(use_amp):
            pred = model(x)
            sum_mae += float(masked_mae(pred, y, mask))
            sum_rmse += float(masked_rmse(pred, y, mask))
            sum_r += float(masked_pearson_r(pred, y, mask))
        n += 1
        it.set_postfix(mae=f"{sum_mae / max(n, 1):.4f}")
    n = max(n, 1)
    return {"mae": sum_mae / n, "rmse": sum_rmse / n, "pearson_r": sum_r / n}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_name: str,
    use_amp: bool,
    scaler,
    show_progress: bool = True,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    model.train()
    sum_loss = 0.0
    steps = 0
    it = tqdm(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=not show_progress,
    )
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            pred = model(x)
            loss = masked_regression_loss(pred, y, mask, loss=loss_name)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sum_loss += float(loss)
        steps += 1
        it.set_postfix(loss=f"{float(loss):.4f}")
    return sum_loss / max(steps, 1)


def init_wandb(args: argparse.Namespace, meta: dict) -> object | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError("wandb is required for --wandb (pip install wandb)") from e

    run_name = args.wandb_run_name or Path(args.out_dir).name
    init_kwargs: dict = {
        "project": args.wandb_project,
        "name": run_name,
        "config": meta,
        "dir": str(Path(args.out_dir).resolve()),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if tags:
        init_kwargs["tags"] = tags
    return wandb.init(**init_kwargs)


def log_wandb_epoch(
    wandb_run: object | None,
    *,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
    lr: float,
    elapsed_s: float,
    best_val_mae: float,
) -> None:
    if wandb_run is None:
        return
    import wandb

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "lr": lr,
            "elapsed_s": elapsed_s,
            "best_val_mae": best_val_mae,
        },
        step=epoch,
    )


def finish_wandb(
    wandb_run: object | None,
    *,
    best_val_mae: float,
    best_epoch: int,
    best_checkpoint: Path,
) -> None:
    if wandb_run is None:
        return
    import wandb

    wandb.run.summary["best_val_mae"] = best_val_mae
    wandb.run.summary["best_epoch"] = best_epoch
    if best_checkpoint.is_file():
        wandb.save(str(best_checkpoint), base_path=str(best_checkpoint.parent))
    wandb.finish()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3D U-Net for HCP UniGrad synth error-map regression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/error-map/unigrad-synth/hcp"),
        help="Root with Train/Val/Test/*.npz (Phase II HCP error-map NPZ).",
    )
    p.add_argument("--train-split", type=str, default="Train")
    p.add_argument("--val-split", type=str, default="Val")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--u-scale", type=float, default=64.0, help="Divide u_pred_{x,y,z} by this scale.")
    p.add_argument(
        "--image-norm",
        type=str,
        default="none",
        choices=["none", "robust"],
        help="Intensity norm (default none: Phase I already masked-z-scored).",
    )
    p.add_argument("--quantile-high", type=float, default=0.99)
    p.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
    p.add_argument(
        "--mask-u-pred",
        action="store_true",
        help="Ablation: zero u_pred_{x,y,z} outside source_mask before the U-Net (default: off).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = 4 on CUDA, 0 on CPU).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/runs/regression/unigrad-synth/3d/error_unet_run1"),
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile(UNet3D) when supported.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--max-train-files", type=int, default=None, metavar="N")
    p.add_argument("--max-val-files", type=int, default=None, metavar="N")
    # LR schedule + early stopping (mirrors unigrad-io trainer)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "none"],
        help="Reduce LR when val MAE plateaus (default) or fixed LR.",
    )
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop after this many epochs without val-MAE improvement (0 = off).",
    )
    p.add_argument("--early-stop-min-delta", type=float, default=0.005)
    # Weights & Biases
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="unigrad-synth-hcp")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.u_scale <= 0:
        raise ValueError("--u-scale must be positive.")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = make_grad_scaler(use_amp)
    if args.num_workers < 0:
        args.num_workers = 4 if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    show_p = not args.no_progress

    train_ds = HCPErrorMapDataset(
        args.data_dir,
        args.train_split,
        u_scale=args.u_scale,
        mask_u_pred=args.mask_u_pred,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
    )
    val_ds = HCPErrorMapDataset(
        args.data_dir,
        args.val_split,
        u_scale=args.u_scale,
        mask_u_pred=args.mask_u_pred,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
    )
    if args.max_train_files is not None and args.max_train_files > 0:
        train_ds.paths = train_ds.paths[: int(args.max_train_files)]
    if args.max_val_files is not None and args.max_val_files > 0:
        val_ds.paths = val_ds.paths[: int(args.max_val_files)]

    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin
    )
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )

    model = UNet3D(in_channels=IN_CHANNELS, base=args.base_channels).to(device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("Using torch.compile(UNet3D)")
        except Exception as exc:
            print(f"WARNING: torch.compile failed ({exc}); using eager mode.", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": "UNet3D",
        "norm": "GroupNorm",
        "in_channels": IN_CHANNELS,
        "out_channels": 1,
        "inputs": ["source", "moving", "u_pred_x", "u_pred_y", "u_pred_z"],
        "target": "||u_gt - u_pred|| (u_error_map)",
        "mask": "source_mask",
        "mask_u_pred": bool(args.mask_u_pred),
        "loss": args.loss,
        "data_dir": str(args.data_dir.resolve()),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train_files": len(train_ds),
        "val_files": len(val_ds),
        "u_scale": args.u_scale,
        "image_norm": args.image_norm,
        "quantile_high": args.quantile_high,
        "base_channels": args.base_channels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "lr_scheduler": args.lr_scheduler,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "min_lr": args.min_lr,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "device": str(device),
        "num_workers": args.num_workers,
        "compile": args.compile,
        "best_metric": "val_mae",
        "metrics_csv": "metrics.csv",
        "wandb": args.wandb,
        "wandb_project": args.wandb_project if args.wandb else None,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    wandb_run = init_wandb(args, meta)

    metrics_path = args.out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_mae", "val_rmse", "val_pearson_r", "lr", "elapsed_s"]
        )

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    best_path = args.out_dir / "best_model.pt"
    t0 = time.time()

    print(
        f"Train {len(train_ds)} / Val {len(val_ds)} on {device} | "
        f"in_ch={IN_CHANNELS} | mask_u_pred={args.mask_u_pred} | loss={args.loss}"
    )

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model,
            train_loader,
            opt,
            device,
            loss_name=args.loss,
            use_amp=use_amp,
            scaler=scaler if use_amp else None,
            show_progress=show_p,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_m = evaluate(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            show_progress=show_p,
            desc=f"val {epoch}/{args.epochs}",
        )
        if scheduler is not None:
            scheduler.step(val_m["mae"])
        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{args.epochs}  train_loss={tr_loss:.6f}  "
            f"val_mae={val_m['mae']:.6f}  val_rmse={val_m['rmse']:.6f}  "
            f"val_r={val_m['pearson_r']:.4f}  lr={lr_now:.2e}  elapsed={dt:.1f}s"
        )
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, val_m["mae"], val_m["rmse"], val_m["pearson_r"], lr_now, dt]
            )
        improved = val_m["mae"] < best_val_mae - args.early_stop_min_delta
        if improved:
            best_val_mae = val_m["mae"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state": checkpoint_state_dict(model),
                    "epoch": epoch,
                    "val_mae": val_m["mae"],
                    "val_rmse": val_m["rmse"],
                    "val_pearson_r": val_m["pearson_r"],
                    "config": meta,
                },
                best_path,
            )
            print(f"  saved best (val_mae={best_val_mae:.6f}) -> {best_path}")
        else:
            epochs_without_improve += 1

        log_wandb_epoch(
            wandb_run,
            epoch=epoch,
            train_loss=tr_loss,
            val_metrics=val_m,
            lr=lr_now,
            elapsed_s=dt,
            best_val_mae=best_val_mae,
        )

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping: no val MAE improvement > {args.early_stop_min_delta} "
                f"for {args.early_stop_patience} epoch(s) "
                f"(best epoch {best_epoch}, val_mae={best_val_mae:.6f})."
            )
            break

    finish_wandb(
        wandb_run,
        best_val_mae=best_val_mae,
        best_epoch=best_epoch,
        best_checkpoint=best_path,
    )
    print(f"Done. Best val MAE={best_val_mae:.6f} at epoch {best_epoch} -> {best_path}")
    print(f"Metrics log: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
