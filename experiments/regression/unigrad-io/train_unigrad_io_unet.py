#!/usr/bin/env python3
"""
Train a 3D U-Net to predict per-voxel IO error magnitude from UniGrad ICON IO volumes.

Layout from ``create_unigrad_io_data.py``:

  <data-dir>/atlas_valid_mask.npz  — shared ``atlas`` (H, W, D), ``valid_mask`` (D, H, W)
  <data-dir>/Train|Val|Test/<subject>.npz per volume:
    ``source`` (H, W, D), ``phi_pred`` / ``phi_predio`` (3, D, H, W),
    ``error_map`` (D, H, W), ``io_iterations``

Model input (5 channels, ``N×C×D×H×W``): robust-normalized subject + atlas,
``phi_pred / phi_scale``. Loss and metrics use ``valid_mask`` only.

Optimizer: AdamW (default ``lr=1e-3``). LR schedule: ``ReduceLROnPlateau`` on ``val_{loss}``.
Early stopping: no improvement > ``--early-stop-min-delta`` (default 0.005) for
``--early-stop-patience`` val checks (default 10).

Loss: ``total = regression_loss + smooth_weight * tv_3d(pred)``. Regression is one of
``--loss mse`` (default), ``l1``, or ``huber`` (``--huber-delta``, default 1.0 in same units as
``error_map``). Default ``smooth_weight=0`` (no TV).

Speed (full volumes are expensive; default is one 3D U-Net step per subject per epoch):

- ``--num-workers 4`` — parallel NPZ load + normalize (default on CUDA).
- Cached normalized atlas (not recomputed every subject).
- ``--val-every 5`` — skip most val passes (~12% of epoch time at 58 val / 461 steps).
- ``--val-start-frac 0.1`` — first validation at epoch ``ceil(0.1 * epochs)`` (avoids early unstable val spikes on curves).
- ``--compile`` — ``torch.compile`` when supported.
- Try ``--batch-size 2`` if GPU memory allows (same H×W×D per volume).

True multi-GPU needs DDP (not implemented); one epoch ≈ 403 train + 58 val forward passes at batch 1.

Example (MSE, default):
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 50 --batch-size 2 --out-dir assets/runs/unigrad-io/error_unet_run2

Example (L1 re-run with early stop, run4b):
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 60 --batch-size 2 --loss l1 --val-every 3 --early-stop-patience 8 --early-stop-min-delta 0.005 --compile --wandb --wandb-project unc-quan --wandb-run-name error_unet_run4b --out-dir assets/runs/unigrad-io/error_unet_run4b

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ATLAS_MASK_FILENAME = "atlas_valid_mask.npz"
DATA_GLOB = "*.npz"
SUBJECT_NPZ_KEYS = frozenset({"source", "phi_pred", "phi_predio", "error_map"})


def default_slice_index(depth: int) -> int:
    return depth // 2


def resolve_val_start_epoch(
    epochs: int,
    *,
    val_start_epoch: int | None = None,
    val_start_frac: float = 0.0,
) -> int:
    """First epoch (1-based) at which validation, LR schedule, and best checkpoint apply."""
    if val_start_epoch is not None and val_start_epoch > 0:
        return min(int(val_start_epoch), epochs)
    if val_start_frac > 0.0:
        return min(max(1, int(np.ceil(val_start_frac * epochs))), epochs)
    return 1


def count_val_passes(total_epochs: int, val_every: int, val_start_epoch: int) -> int:
    n = 0
    for e in range(1, total_epochs + 1):
        if e < val_start_epoch:
            continue
        if (e % val_every == 0) or (e == total_epochs):
            n += 1
    return n


def load_atlas_target_hw_d(data_root: Path) -> np.ndarray | None:
    path = Path(data_root) / ATLAS_MASK_FILENAME
    if not path.is_file():
        return None
    with np.load(path) as z:
        key = "atlas" if "atlas" in z.files else "target"
        if key not in z.files:
            return None
        atlas = np.asarray(z[key])
    if atlas.ndim != 3:
        raise ValueError(f"{path}: atlas must be (H, W, D), got {atlas.shape}")
    return atlas


def load_atlas_valid_mask_dhw(data_root: Path) -> np.ndarray | None:
    path = Path(data_root) / ATLAS_MASK_FILENAME
    if not path.is_file():
        return None
    with np.load(path) as z:
        if "valid_mask" not in z.files:
            return None
        mask = np.asarray(z["valid_mask"])
    if mask.ndim != 3:
        raise ValueError(f"{path}: valid_mask must be (D, H, W), got {mask.shape}")
    return mask.astype(np.bool_, copy=False)

DEFAULT_EPOCHS = 25
DEFAULT_EARLY_STOP_PATIENCE = 8


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


def collect_io_npz_paths(root: Path, split: str) -> list[Path]:
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(f"Missing split directory: {d}")
    files = sorted(d.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No {DATA_GLOB} under {d}")
    return files


def volume_hw_d_to_dhw(vol: np.ndarray) -> np.ndarray:
    """``(H, W, D)`` → ``(D, H, W)``."""
    if vol.ndim != 3:
        raise ValueError(f"Expected (H, W, D), got {vol.shape}")
    return np.transpose(vol, (2, 0, 1)).astype(np.float32, copy=False)


def validate_subject_npz(path: Path) -> None:
    with np.load(path) as data:
        missing = SUBJECT_NPZ_KEYS - set(data.files)
        if missing:
            raise KeyError(
                f"{path.name}: missing {sorted(missing)}; expected keys "
                f"{sorted(SUBJECT_NPZ_KEYS)} from create_unigrad_io_data.py"
            )


class UniGradIOErrorDataset(Dataset):
    """3D IO npz: (5, D, H, W) input, (1, D, H, W) target."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        image_norm: str = "robust",
        quantile_high: float = 0.99,
        phi_scale: float = 64.0,
    ):
        self.paths = collect_io_npz_paths(root, split)
        validate_subject_npz(self.paths[0])
        if image_norm not in ("none", "robust"):
            raise ValueError("image_norm must be 'none' or 'robust'")
        self.image_norm = image_norm
        self.quantile_high = quantile_high
        self.phi_scale = float(phi_scale)
        self.data_root = Path(root)
        self.atlas_target_hw_d = load_atlas_target_hw_d(root)
        self.valid_mask_dhw = load_atlas_valid_mask_dhw(root)
        atlas_npz = root / ATLAS_MASK_FILENAME
        if self.atlas_target_hw_d is None:
            raise FileNotFoundError(
                f"Missing shared atlas in {atlas_npz}. "
                "Run create_unigrad_io_data.py with --output-path pointing at this data-dir."
            )
        if self.valid_mask_dhw is None:
            raise FileNotFoundError(
                f"Missing valid_mask in {atlas_npz}. "
                "Regenerate data with create_unigrad_io_data.py."
            )
        self._atlas_target_dhw = volume_hw_d_to_dhw(self.atlas_target_hw_d)
        self._atlas_target_n = self._norm_volume(self._atlas_target_dhw)

    def __len__(self) -> int:
        return len(self.paths)

    def _norm_volume(self, x: np.ndarray) -> np.ndarray:
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
            source = volume_hw_d_to_dhw(np.asarray(data["source"]))
            if "target" in data.files:
                target = volume_hw_d_to_dhw(np.asarray(data["target"]))
                target_n = self._norm_volume(target)
            else:
                target_n = self._atlas_target_n
            phi_pred = np.asarray(data["phi_pred"], dtype=np.float32)
            err = np.asarray(data["error_map"], dtype=np.float32)

        if phi_pred.ndim != 4 or phi_pred.shape[0] != 3:
            raise ValueError(f"{path.name}: phi_pred must be (3, D, H, W), got {phi_pred.shape}")
        if err.ndim != 3:
            raise ValueError(f"{path.name}: error_map must be (D, H, W), got {err.shape}")
        d, h, w = err.shape
        if phi_pred.shape[1:] != (d, h, w):
            raise ValueError(
                f"{path.name}: phi_pred {phi_pred.shape[1:]} vs error_map {(d, h, w)}"
            )

        source_n = self._norm_volume(source)
        phi_n = phi_pred / self.phi_scale

        x = np.concatenate(
            [source_n[None, ...], target_n[None, ...], phi_n],
            axis=0,
        )
        y = err[None, ...]
        if self.valid_mask_dhw is not None:
            mask = self.valid_mask_dhw
            if mask.shape != (d, h, w):
                raise ValueError(
                    f"{path.name}: valid_mask {mask.shape} vs error_map {(d, h, w)}; "
                    f"regenerate {ATLAS_MASK_FILENAME}"
                )
        else:
            mask = np.ones((d, h, w), dtype=np.bool_)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "path": str(path),
        }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(1).float()
    diff = (pred - target) ** 2 * m
    return diff.sum() / m.sum().clamp_min(1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(1).float()
    diff = torch.abs(pred - target) * m
    return diff.sum() / m.sum().clamp_min(1.0)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    m = mask.unsqueeze(1).float()
    raw = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (raw * m).sum() / m.sum().clamp_min(1.0)


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss == "mse":
        return masked_mse(pred, target, mask)
    if loss == "l1":
        return masked_l1(pred, target, mask)
    if loss == "huber":
        return masked_huber(pred, target, mask, huber_delta)
    raise ValueError(f"Unknown --loss {loss!r}; expected mse, l1, huber.")


def masked_tv_smoothness_3d(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    if m.dim() == 5:
        m = m.squeeze(1)

    def edge_tv(diff: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (diff.abs() * w.unsqueeze(1)).sum() / w.sum().clamp_min(1.0)

    gd = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    w_d = 1.0 - (m[:, :-1, :, :] * m[:, 1:, :, :])
    tv_d = edge_tv(gd, w_d)

    gh = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    w_h = 1.0 - (m[:, :, :-1, :] * m[:, :, 1:, :])
    tv_h = edge_tv(gh, w_h)

    gw = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    w_w = 1.0 - (m[:, :, :, :-1] * m[:, :, :, 1:])
    tv_w = edge_tv(gw, w_w)

    return (tv_d + tv_h + tv_w) / 3.0


def masked_regression_plus_smoothness_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss: str,
    smooth_weight: float,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reg = masked_regression_loss(pred, target, mask, loss=loss, huber_delta=huber_delta)
    if smooth_weight <= 0.0:
        z = torch.zeros_like(reg)
        return reg, z, reg
    smooth = masked_tv_smoothness_3d(pred, mask)
    total = reg + smooth_weight * smooth
    return reg, smooth, total


def masked_mse_plus_smoothness_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    smooth_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return masked_regression_plus_smoothness_3d(
        pred,
        target,
        mask,
        loss="mse",
        smooth_weight=smooth_weight,
        huber_delta=1.0,
    )


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """3D U-Net: 5 in-ch (subject, atlas, phi_pred×3) → 1 out-ch (error_map)."""

    def __init__(self, in_channels: int = 5, base: int = 32) -> None:
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
    """Underlying module when ``torch.compile`` wrapped (``_orig_mod``)."""
    return getattr(model, "_orig_mod", model)


def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save/load-friendly weights (no ``_orig_mod.`` prefix from ``torch.compile``)."""
    return eager_module(model).state_dict()


def load_checkpoint_state_dict(model: nn.Module, state_dict: dict) -> None:
    """Load weights from ``best_model.pt`` (handles legacy ``_orig_mod.*`` keys)."""
    fixed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            fixed[key.removeprefix("_orig_mod.")] = value
        else:
            fixed[key] = value
    eager_module(model).load_state_dict(fixed, strict=True)


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


def subset_dataset_paths(ds: UniGradIOErrorDataset, max_files: int | None) -> None:
    if max_files is not None and max_files > 0 and max_files < len(ds.paths):
        ds.paths = ds.paths[: int(max_files)]


@torch.no_grad()
def evaluate_primary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss: str,
    use_amp: bool,
    huber_delta: float,
    show_progress: bool = True,
    desc: str = "val",
    overall_pbar: tqdm | None = None,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    it = tqdm(loader, desc=desc, leave=False, unit="batch", disable=not show_progress, position=2)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with amp_autocast(use_amp):
            pred = model(x)
            total += float(
                masked_regression_loss(pred, y, mask, loss=loss, huber_delta=huber_delta)
            )
        n += 1
        if overall_pbar is not None:
            overall_pbar.update(1)
            overall_pbar.set_postfix(phase="val", val=f"{total / max(n, 1):.3f}")
    return total / max(n, 1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler,
    loss: str,
    smooth_weight: float,
    huber_delta: float,
    show_progress: bool = True,
    epoch: int = 1,
    total_epochs: int = 1,
    overall_pbar: tqdm | None = None,
) -> float:
    model.train()
    sum_reg = 0.0
    steps = 0
    desc = f"train {epoch}/{total_epochs}"
    it = tqdm(loader, desc=desc, leave=False, unit="batch", disable=not show_progress, position=2)
    for batch in it:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            pred = model(x)
            reg, smooth, total = masked_regression_plus_smoothness_3d(
                pred, y, mask, loss=loss, smooth_weight=smooth_weight, huber_delta=huber_delta
            )
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            opt.step()
        sum_reg += float(reg)
        steps += 1
        if show_progress:
            it.set_postfix(reg=float(reg), total=float(total))
        if overall_pbar is not None:
            overall_pbar.update(1)
            overall_pbar.set_postfix(
                phase="train",
                epoch=f"{epoch}/{total_epochs}",
                reg=f"{float(reg):.3f}",
            )
    return sum_reg / max(steps, 1)


ALL_LOSSES = ("mse", "l1", "huber")


def primary_train_col(loss: str) -> str:
    return f"train_{loss}"


def primary_val_col(loss: str) -> str:
    return f"val_{loss}"


def metrics_csv_header(loss: str) -> list[str]:
    return [
        "epoch",
        "loss",
        primary_train_col(loss),
        primary_val_col(loss),
        "elapsed_s",
    ]


def metric_plot_label(split_loss: str) -> str:
    """Legend / axis label, e.g. train_l1, val_mse."""
    return split_loss


def loss_display_name(loss: str) -> str:
    if loss == "mse":
        return "MSE"
    if loss == "l1":
        return "L1"
    if loss == "huber":
        return "Huber"
    raise ValueError(f"Unknown loss {loss!r}")


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
    loss_name: str,
    train: float,
    val: float,
    lr: float,
    elapsed_s: float,
    best_val_metric: float,
) -> None:
    if wandb_run is None:
        return
    import wandb

    train_col = primary_train_col(loss_name)
    val_col = primary_val_col(loss_name)
    wandb.log(
        {
            "epoch": epoch,
            "loss": loss_name,
            train_col: train,
            val_col: val,
            "lr": lr,
            "elapsed_s": elapsed_s,
            f"best_{val_col}": best_val_metric,
        },
        step=epoch,
    )


def finish_wandb(
    wandb_run: object | None,
    *,
    best_val_metric: float,
    best_epoch: int,
    best_checkpoint: Path,
    training_loss: str,
) -> None:
    if wandb_run is None:
        return
    import wandb

    wandb.run.summary[f"best_{primary_val_col(training_loss)}"] = best_val_metric
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["training_loss"] = training_loss
    if best_checkpoint.is_file():
        wandb.save(str(best_checkpoint), base_path=str(best_checkpoint.parent))
    wandb.finish()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3D U-Net for IO error_map from UniGrad IO npz volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/IXI_unigrad_io"),
        help="Root with Train/Val/Test/*.npz from create_unigrad_io_data.py.",
    )
    p.add_argument("--train-split", type=str, default="Train")
    p.add_argument("--val-split", type=str, default="Val")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--image-norm", type=str, default="robust", choices=["none", "robust"])
    p.add_argument("--quantile-high", type=float, default=0.99)
    p.add_argument("--phi-scale", type=float, default=64.0)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = 4 on CUDA, 0 on CPU).",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs (1 = every epoch).",
    )
    p.add_argument(
        "--val-start-epoch",
        type=int,
        default=None,
        metavar="N",
        help="First epoch to run val / LR schedule / best checkpoint (overrides --val-start-frac).",
    )
    p.add_argument(
        "--val-start-frac",
        type=float,
        default=0.1,
        help="If --val-start-epoch unset: first val at ceil(frac * epochs). 0 = epoch 1.",
    )
    p.add_argument(
        "--max-train-files",
        type=int,
        default=None,
        help="Cap train subjects (debug / smoke).",
    )
    p.add_argument(
        "--max-val-files",
        type=int,
        default=None,
        help="Cap val subjects (debug / smoke).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(UNet3D) when supported (PyTorch 2+).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/runs/3d/unigrad-io/error_unet_run1"),
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--smooth-weight",
        type=float,
        default=0.0,
        help="Weight on masked 3D TV of pred (0 = no TV; >0 penalises |Δpred| at mask edges).",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "l1", "huber"],
        help="Masked regression term in total loss (before TV). Val + checkpoint use the matching metric.",
    )
    p.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber delta (same units as error_map voxels); used for --loss huber and for val/huber.",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Stop after this many val checks without meaningful improvement (0 = off).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.005,
        help="Min decrease in val loss (matching --loss) to reset early-stop counter.",
    )
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "none"],
        help="Reduce LR when primary val loss plateaus (default), or fixed LR.",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Epochs without val improvement before ReduceLROnPlateau cuts LR.",
    )
    p.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="LR multiply factor when plateau scheduler fires.",
    )
    p.add_argument("--min-lr", type=float, default=1e-6, help="Floor for scheduled LR.")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases (requires wandb package + login).",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="unigrad-io",
        help="W&B project name (with --wandb).",
    )
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity / team.")
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: out-dir folder name).",
    )
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags, e.g. '3d,unet,masked'.",
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phi_scale <= 0:
        raise ValueError("--phi-scale must be positive.")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = make_grad_scaler(use_amp)
    if args.num_workers < 0:
        args.num_workers = 4 if device.type == "cuda" else 0
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")
    if args.huber_delta <= 0:
        raise ValueError("--huber-delta must be positive")
    if args.val_start_frac < 0.0:
        raise ValueError("--val-start-frac must be >= 0")
    val_start_epoch = resolve_val_start_epoch(
        args.epochs,
        val_start_epoch=args.val_start_epoch,
        val_start_frac=args.val_start_frac,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = UniGradIOErrorDataset(
        args.data_dir,
        args.train_split,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
        phi_scale=args.phi_scale,
    )
    val_ds = UniGradIOErrorDataset(
        args.data_dir,
        args.val_split,
        image_norm=args.image_norm,
        quantile_high=args.quantile_high,
        phi_scale=args.phi_scale,
    )
    subset_dataset_paths(train_ds, args.max_train_files)
    subset_dataset_paths(val_ds, args.max_val_files)

    pin = device.type == "cuda"
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = UNet3D(in_channels=5, base=args.base_channels).to(device)
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
            opt,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": "UNet3D",
        "in_channels": 5,
        "out_channels": 1,
        "inputs": ["source", "atlas", "phi_pred"],
        "atlas_mask_file": ATLAS_MASK_FILENAME,
        "target": "error_map",
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
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "optimizer": "AdamW",
        "lr_scheduler": args.lr_scheduler,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "min_lr": args.min_lr,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "val_every": args.val_every,
        "val_start_epoch": val_start_epoch,
        "val_start_frac": args.val_start_frac,
        "val_start_epoch_arg": args.val_start_epoch,
        "num_workers": args.num_workers,
        "compile": args.compile,
        "max_train_files": args.max_train_files,
        "max_val_files": args.max_val_files,
        "wandb": args.wandb,
        "wandb_project": args.wandb_project if args.wandb else None,
        "show_progress": not args.no_progress,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    wandb_run = init_wandb(args, meta)

    metrics_path = args.out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(metrics_csv_header(args.loss))

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    last_val = float("nan")
    best_path = args.out_dir / "best_model.pt"
    show_p = not args.no_progress
    t0 = time.time()
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    val_epochs = count_val_passes(args.epochs, args.val_every, val_start_epoch)
    total_steps = args.epochs * train_steps + val_epochs * val_steps
    if val_start_epoch > 1:
        print(
            f"Validation starts at epoch {val_start_epoch} "
            f"(epochs 1–{val_start_epoch - 1}: train only; val columns logged as nan)."
        )

    overall_pbar: tqdm | None = None
    if show_p:
        overall_pbar = tqdm(
            total=total_steps,
            desc="total",
            unit="step",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

    epoch_loop = range(1, args.epochs + 1)
    if show_p:
        epoch_loop = tqdm(
            epoch_loop,
            desc="epoch",
            unit="ep",
            total=args.epochs,
            position=1,
            leave=True,
            dynamic_ncols=True,
        )

    for epoch in epoch_loop:
        tr_reg = train_epoch(
            model,
            train_loader,
            opt,
            device,
            use_amp=use_amp,
            scaler=scaler if use_amp else None,
            loss=args.loss,
            smooth_weight=args.smooth_weight,
            huber_delta=args.huber_delta,
            show_progress=show_p,
            epoch=epoch,
            total_epochs=args.epochs,
            overall_pbar=overall_pbar,
        )
        want_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        run_val = want_val and (epoch >= val_start_epoch)
        if run_val:
            val_metric = evaluate_primary(
                model,
                val_loader,
                device,
                loss=args.loss,
                use_amp=use_amp,
                huber_delta=args.huber_delta,
                show_progress=show_p,
                desc=f"val {epoch}/{args.epochs}",
                overall_pbar=overall_pbar,
            )
            last_val = val_metric
        elif epoch < val_start_epoch:
            val_metric = float("nan")
            if show_p:
                print(f"  (warmup: no val before epoch {val_start_epoch})")
        else:
            val_metric = last_val
            if show_p and np.isfinite(val_metric):
                print(f"  (skipped val; last val_{args.loss}={val_metric:.6f})")
        if scheduler is not None and run_val and np.isfinite(val_metric):
            scheduler.step(val_metric)
        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_{args.loss}={tr_reg:.6f}  val_{args.loss}={val_metric:.6f}  "
            f"lr={lr_now:.2e}  elapsed={dt:.1f}s"
        )
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, args.loss, tr_reg, val_metric, dt])
        best_so_far = best_val if np.isnan(val_metric) else min(best_val, val_metric)
        log_wandb_epoch(
            wandb_run,
            epoch=epoch,
            loss_name=args.loss,
            train=tr_reg,
            val=val_metric,
            lr=lr_now,
            elapsed_s=dt,
            best_val_metric=best_so_far,
        )
        if run_val and np.isfinite(val_metric):
            if val_metric < best_val - args.early_stop_min_delta:
                best_val = val_metric
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(
                    {
                        "model_state": checkpoint_state_dict(model),
                        "epoch": epoch,
                        "val_metric": val_metric,
                        "training_loss": args.loss,
                        "config": meta,
                    },
                    best_path,
                )
                print(f"  saved best to {best_path}")
            else:
                epochs_without_improve += 1

        if (
            run_val
            and args.early_stop_patience > 0
            and epochs_without_improve >= args.early_stop_patience
        ):
            print(
                f"Early stopping: no val {args.loss} improvement > {args.early_stop_min_delta} "
                f"for {args.early_stop_patience} val check(s) "
                f"(best epoch {best_epoch}, val_{args.loss}={best_val:.6f})."
            )
            break

    if overall_pbar is not None:
        overall_pbar.close()

    finish_wandb(
        wandb_run,
        best_val_metric=best_val,
        best_epoch=best_epoch,
        best_checkpoint=best_path,
        training_loss=args.loss,
    )
    print(
        f"Done. Best val {args.loss}={best_val:.6f} at epoch {best_epoch} -> {best_path}"
    )
    print(f"Metrics log: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
