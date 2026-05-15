#!/usr/bin/env python3
"""
Sweep UniGradICON IO iteration counts to pick a non-overfitting IO budget for
the downstream ``error_map = phi_predio - phi_pred`` regression target.

How it works
------------
Run a single IO trajectory per subject (Adam, LR=2e-5, LNCC -- matching
``icon_registration.itk_wrapper.finetune_execute``) and snapshot ``phi`` at
each iteration in ``--checkpoints``. For each snapshot we track:

  - ``io_loss``, ``LNCC(warped, target)``   -- IO progress (should improve)
  - ``mean(error_map) = mean ||phi@N - phi@0||``  -- U-Net regression signal
  - ``neg_jac_pct``  ( = %|J|<0 of T = id + phi ) -- folded pixels (stop before
                                                    this starts climbing)

See ``compute_sweep_metrics`` for the formal definitions.

Outputs per subject (saved next to ``--save-path``)
---------------------------------------------------
  - ``<stem>_<subject>_images.png``  4-row grid (warped+grid, residual,
                                     ||phi||, error_map), one column / ckpt.
  - ``<stem>_<subject>_curves.png``  2-panel curves: quality + field health.
  - ``<stem>_<subject>_metrics.csv`` per-iter metrics.

Example
-------
  python sweep_io_iterations.py \\
      --split Train --num-subjects 3 \\
      --save-path ./assets/images/unigrad-io/sweep_io.png --no-show

  # (uses default --checkpoints 0,50,100,150,200,250 and --seed 42)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unigradicon import get_unigradicon, make_sim

_DH = Path(__file__).resolve().parent
if str(_DH) not in sys.path:
    sys.path.insert(0, str(_DH))

from create_unigrad_io_data import (  # noqa: E402
    ATLAS_SLICE_INDEX,
    apply_displacement_2d,
    load_atlas_slice,
    phi_vectorfield_to_slice_pixels,
    preprocess_for_unigrad,
)
from visualize_unigrad_io_data import overlay_deformation_grid  # noqa: E402


def lncc_2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sigma: int = 5,
    device: torch.device | None = None,
) -> float:
    """Mean local normalized cross correlation between two 2D images.

    Uses a uniform window of size ``2*sigma+1`` (matching the ``sigma`` used by
    ``icon_registration.LNCC``). Returns the mean LNCC over all pixels; range
    is roughly ``[-1, 1]`` (higher is better).
    """
    dev = device if device is not None else torch.device("cpu")
    k = 2 * sigma + 1
    pad = sigma
    A = torch.from_numpy(a).to(dev, dtype=torch.float32)[None, None]
    B = torch.from_numpy(b).to(dev, dtype=torch.float32)[None, None]

    mu_a = F.avg_pool2d(A, kernel_size=k, stride=1, padding=pad)
    mu_b = F.avg_pool2d(B, kernel_size=k, stride=1, padding=pad)
    var_a = F.avg_pool2d(A * A, kernel_size=k, stride=1, padding=pad) - mu_a * mu_a
    var_b = F.avg_pool2d(B * B, kernel_size=k, stride=1, padding=pad) - mu_b * mu_b
    cov_ab = F.avg_pool2d(A * B, kernel_size=k, stride=1, padding=pad) - mu_a * mu_b

    denom = torch.sqrt(torch.clamp(var_a * var_b, min=1e-10))
    lncc_map = cov_ab / denom
    return float(lncc_map.mean().item())


def pct_negative_jacobian_2d(phi_px: np.ndarray) -> float:
    """Percentage of pixels with negative Jacobian determinant for a 2D transform.

    The 2D transform is ``T(y, x) = (y + phi[1, y, x], x + phi[0, y, x])``.
    ``det(J) = (1 + d phi0 / dx)(1 + d phi1 / dy) - (d phi0 / dy)(d phi1 / dx)``,
    computed with forward finite differences on a ``(H-1, W-1)`` interior.
    Negative determinants indicate folded (unphysical) deformations.
    """
    dphi0 = phi_px[0].astype(np.float64)  # column displacement
    dphi1 = phi_px[1].astype(np.float64)  # row displacement
    dphi0_dx = np.diff(dphi0, axis=1)[:-1, :]
    dphi0_dy = np.diff(dphi0, axis=0)[:, :-1]
    dphi1_dx = np.diff(dphi1, axis=1)[:-1, :]
    dphi1_dy = np.diff(dphi1, axis=0)[:, :-1]
    det = (1.0 + dphi0_dx) * (1.0 + dphi1_dy) - dphi0_dy * dphi1_dx
    return 100.0 * float(np.mean(det < 0))


def parse_checkpoints(spec: str) -> list[int]:
    """Parse comma-separated checkpoint iterations; ensures sorted unique non-negative ints."""
    vals = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    bad = [v for v in vals if v < 0]
    if bad:
        raise ValueError(f"checkpoints must be >= 0; got {bad}")
    return vals


def resolve_subject_paths(
    ixi_root: Path,
    *,
    split: str,
    subject_path: Path | None,
    subject_indices: list[int] | None,
    num_subjects: int | None,
    seed: int,
) -> list[Path]:
    """Resolve which subject .npy files to sweep over.

    Priority of CLI flags (first that is set wins):
      1. ``--subject-path P``       -> exactly that one file.
      2. ``--subject-indices i,j,k`` -> those sorted indices into the split.
      3. ``--num-subjects N`` (+ ``--seed S``) -> N random indices from the split.
      4. Default (nothing set)      -> single subject at index 0 (back-compat).
    """
    if subject_path is not None:
        if not subject_path.is_file():
            raise FileNotFoundError(f"--subject-path not found: {subject_path}")
        return [subject_path]
    in_dir = ixi_root / split
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Split directory missing: {in_dir}")
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy in {in_dir}")
    n = len(files)

    if subject_indices:
        chosen = list(subject_indices)
    elif num_subjects is not None:
        if num_subjects < 1 or num_subjects > n:
            raise ValueError(
                f"--num-subjects {num_subjects} out of valid range [1, {n}]"
            )
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(n), k=num_subjects))
    else:
        chosen = [0]

    for idx in chosen:
        if idx < 0 or idx >= n:
            raise IndexError(f"Subject index {idx} out of range [0, {n})")
    return [in_dir / files[idx] for idx in chosen]


def write_metrics_csv(
    csv_path: Path,
    metrics: dict[str, list[tuple[int, float]]],
    io_loss_at_iter: dict[int, float],
) -> None:
    """Write the per-iteration metrics for one subject to a CSV file.

    Columns: iter, io_loss, lncc, mean_phi_px, mean_error_map_px,
             err_map_p50_px, err_map_p95_px, err_map_max_px, neg_jac_pct.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    iters = [it for it, _ in metrics["lncc"]]
    fieldnames = [
        "iter",
        "io_loss",
        "lncc",
        "mean_phi_px",
        "mean_error_map_px",
        "err_map_p50_px",
        "err_map_p95_px",
        "err_map_max_px",
        "neg_jac_pct",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, it in enumerate(iters):
            writer.writerow(
                {
                    "iter": it,
                    "io_loss": io_loss_at_iter.get(it, float("nan")),
                    "lncc": metrics["lncc"][i][1],
                    "mean_phi_px": metrics["phi_mag"][i][1],
                    "mean_error_map_px": metrics["err_mag"][i][1],
                    "err_map_p50_px": metrics["err_p50"][i][1],
                    "err_map_p95_px": metrics["err_p95"][i][1],
                    "err_map_max_px": metrics["err_max"][i][1],
                    "neg_jac_pct": metrics["negjac_pct"][i][1],
                }
            )
    print(f"Saved metrics CSV: {csv_path}")


def run_io_with_snapshots(
    net: torch.nn.Module,
    source_175: torch.Tensor,
    target_175: torch.Tensor,
    *,
    checkpoints: list[int],
    lr: float,
    optimizer_name: str,
    orig_h: int,
    orig_w: int,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Run a single IO trajectory and snapshot ``phi`` (pixel space) at each checkpoint.

    Returns ``(snapshots, io_loss_at_iter)``:
      - ``snapshots[N]``       : ``(2, H, W)`` displacement field at iter ``N``.
      - ``io_loss_at_iter[N]`` : value of the IO loss (e.g. -LNCC + reg) at iter
                                 ``N``, evaluated in ``no_grad`` after step ``N``.
                                 Should decrease monotonically while IO descends.

    The model's original weights are restored before returning.
    """
    device = source_175.device
    state0_cpu = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    snapshots: dict[int, np.ndarray] = {}
    io_loss_at_iter: dict[int, float] = {}

    if 0 in checkpoints:
        with torch.no_grad():
            loss_tuple = net(source_175, target_175)
            io_loss_at_iter[0] = float(loss_tuple[0].detach().item())
            snapshots[0] = phi_vectorfield_to_slice_pixels(net, orig_h, orig_w)
            del loss_tuple

    pending = [c for c in checkpoints if c > 0]
    if not pending:
        net.load_state_dict(state0_cpu)
        net.eval()
        torch.cuda.empty_cache()
        return snapshots, io_loss_at_iter

    if optimizer_name == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    was_training = net.training
    net.train()
    cur = 0
    total_iters = pending[-1]
    pbar = tqdm(total=total_iters, desc="IO iters", dynamic_ncols=True)
    last_loss = float("nan")
    try:
        for target_iter in pending:
            while cur < target_iter:
                opt.zero_grad(set_to_none=True)
                loss_tuple = net(source_175, target_175)
                loss_tuple[0].backward()
                opt.step()
                last_loss = float(loss_tuple[0].detach().item())
                del loss_tuple
                cur += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{last_loss:.4f}", ckpt=f"{target_iter}")
            with torch.no_grad():
                loss_tuple = net(source_175, target_175)
                io_loss_at_iter[target_iter] = float(loss_tuple[0].detach().item())
                snapshots[target_iter] = phi_vectorfield_to_slice_pixels(net, orig_h, orig_w)
                del loss_tuple
            tqdm.write(
                f"  [snapshot] iter={target_iter:>5d}  "
                f"io_loss={io_loss_at_iter[target_iter]:.4f}  "
                f"(phi shape={snapshots[target_iter].shape})"
            )
    finally:
        pbar.close()
        if not was_training:
            net.eval()
        del opt
        torch.cuda.empty_cache()
        net.load_state_dict(state0_cpu)
        net.eval()
        torch.cuda.empty_cache()

    _ = device  # keep linter happy; device determined by source_175
    return snapshots, io_loss_at_iter


def compute_sweep_metrics(
    *,
    target_img: np.ndarray,
    snapshots: dict[int, np.ndarray],
    warped_by_iter: dict[int, np.ndarray],
    lncc_sigma: int,
) -> dict[str, list[tuple[int, float]]]:
    """Compute per-checkpoint metrics for the sweep.

    Returns a dict where each entry is a list of ``(iteration, value)``:
      - ``"lncc"``       : LNCC(warped@N, target)            -- quality, higher better
      - ``"phi_mag"``    : mean ||phi@N|| in pixels          -- field magnitude
      - ``"err_mag"``    : mean ||phi@N - phi@0||_2          -- mean magnitude of
                              the downstream U-Net regression target if
                              ``--io-iterations=N`` were chosen. Starts at 0,
                              grows monotonically; the "signal strength" of
                              the error_map fed to the U-Net.
      - ``"negjac_pct"`` : neg_jac_pct = %|J|<0 of T = id + phi -- folded pixels

    NOTE: MAE(warped, target) is intentionally not computed -- IO descends LNCC
    (intensity-invariant), so MAE is dominated by the cross-subject intensity
    floor and is flat regardless of how well IO is doing.
    """
    iters_sorted = sorted(snapshots.keys())
    if 0 not in iters_sorted:
        raise ValueError("Snapshots must include iteration 0 for the error_map baseline.")
    phi0 = snapshots[0]
    metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lncc: list[tuple[int, float]] = []
    phi_mag: list[tuple[int, float]] = []
    err_mag: list[tuple[int, float]] = []
    err_p50: list[tuple[int, float]] = []
    err_p95: list[tuple[int, float]] = []
    err_max: list[tuple[int, float]] = []
    negjac: list[tuple[int, float]] = []
    for it in iters_sorted:
        warped = warped_by_iter[it]
        phi = snapshots[it]
        emap = np.sqrt(np.sum((phi - phi0) ** 2, axis=0))
        lncc.append((it, lncc_2d(warped, target_img, sigma=lncc_sigma, device=metric_device)))
        phi_mag.append((it, float(np.mean(np.sqrt(np.sum(phi * phi, axis=0))))))
        err_mag.append((it, float(np.mean(emap))))
        err_p50.append((it, float(np.percentile(emap, 50.0))))
        err_p95.append((it, float(np.percentile(emap, 95.0))))
        err_max.append((it, float(np.max(emap))))
        negjac.append((it, pct_negative_jacobian_2d(phi)))
    return {
        "lncc": lncc,
        "phi_mag": phi_mag,
        "err_mag": err_mag,
        "err_p50": err_p50,
        "err_p95": err_p95,
        "err_max": err_max,
        "negjac_pct": negjac,
    }


def print_sweep_metrics(
    metrics: dict[str, list[tuple[int, float]]],
    *,
    io_loss_at_iter: dict[int, float],
    lncc_sigma: int,
) -> None:
    """Pretty-print the sweep metrics table to stdout.

    ``io_loss`` is the actual quantity the IO optimizer minimizes (e.g. -LNCC +
    regulariser inside the network). It should decrease monotonically when IO
    is descending; this is the most direct sanity check that the optimiser is
    doing real work, independent of any external evaluation metric.
    """
    print(
        f"\nSweep metrics (atlas slice {ATLAS_SLICE_INDEX}, LNCC sigma={lncc_sigma}, "
        "io_loss = quantity Adam descends; mean(error_map) = signal magnitude "
        "of the U-Net regression target if --io-iterations=N is picked):"
    )
    print(
        f"  {'iter':>6s} | {'io_loss':>9s} | {'LNCC':>7s} | "
        f"{'mean||phi|| (px)':>17s} | {'mean(error_map)':>17s} | {'neg_jac_pct':>11s}"
    )
    print(
        f"  {'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*17}-+-{'-'*17}-+-{'-'*11}"
    )
    for (it, lc), (_, pm), (_, em), (_, nj) in zip(
        metrics["lncc"], metrics["phi_mag"], metrics["err_mag"], metrics["negjac_pct"]
    ):
        il = io_loss_at_iter.get(it, float("nan"))
        print(
            f"  {it:>6d} | {il:>9.4f} | {lc:>7.4f} | "
            f"{pm:>17.4f} | {em:>17.4f} | {nj:>11.4f}"
        )

    print(
        "\nError-map shape (px). p50 < mean < p95 ~ structured signal; "
        "max >> p95 ~ outliers / folds:"
    )
    print(
        f"  {'iter':>6s} | {'p50':>8s} | {'mean':>8s} | {'p95':>8s} | {'max':>8s}"
    )
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for (it, em), (_, p50), (_, p95), (_, emx) in zip(
        metrics["err_mag"], metrics["err_p50"], metrics["err_p95"], metrics["err_max"]
    ):
        print(
            f"  {it:>6d} | {p50:>8.4f} | {em:>8.4f} | {p95:>8.4f} | {emx:>8.4f}"
        )


def _suptitle_str(subject_path: Path, io_optimizer: str, io_lr: float, io_sim: str) -> str:
    return (
        f"IO iteration sweep — {subject_path.name}  |  "
        f"atlas slice {ATLAS_SLICE_INDEX}, opt={io_optimizer}, lr={io_lr:g}, sim={io_sim}"
    )


def render_sweep_images(
    *,
    target_img: np.ndarray,
    snapshots: dict[int, np.ndarray],
    warped_by_iter: dict[int, np.ndarray],
    grid_stride: int,
    save_path: Path | None,
    no_show: bool,
    subject_path: Path,
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
) -> None:
    """Render the 4-row image grid: warped+grid / residual / ||phi|| / error_map."""
    iters_sorted = sorted(snapshots.keys())
    n_iters = len(iters_sorted)
    fig = plt.figure(figsize=(3.0 * max(n_iters, 4), 12.5))
    gs = fig.add_gridspec(
        4, n_iters, height_ratios=[3.0, 3.0, 3.0, 3.0], hspace=0.25, wspace=0.05
    )

    phi0 = snapshots[iters_sorted[0]]

    residuals_concat = np.concatenate(
        [(warped_by_iter[it] - target_img).ravel() for it in iters_sorted]
    )
    res_v = max(float(np.percentile(np.abs(residuals_concat), 99.0)), 1e-6)

    phi_mag_by_iter: dict[int, np.ndarray] = {
        it: np.sqrt(np.sum(snapshots[it] ** 2, axis=0)) for it in iters_sorted
    }
    err_map_by_iter: dict[int, np.ndarray] = {
        it: np.sqrt(np.sum((snapshots[it] - phi0) ** 2, axis=0)) for it in iters_sorted
    }
    phi_mag_concat = np.concatenate([m.ravel() for m in phi_mag_by_iter.values()])
    err_map_concat = np.concatenate([m.ravel() for m in err_map_by_iter.values()])
    phi_v = max(float(np.percentile(phi_mag_concat, 99.0)), 1e-6)
    err_v = max(float(np.percentile(err_map_concat, 99.0)), 1e-6)

    row_labels = ["warped", "residual", "||phi|| (px)", "error_map (px)"]

    for col, it in enumerate(iters_sorted):
        ax_w = fig.add_subplot(gs[0, col])
        ax_w.imshow(warped_by_iter[it], cmap="gray")
        ax_w.set_title(f"iter {it}", fontsize=10)
        ax_w.axis("off")
        overlay_deformation_grid(ax_w, snapshots[it], stride=grid_stride)
        if col == 0:
            ax_w.text(-0.04, 0.5, row_labels[0], rotation=90, va="center",
                      ha="right", transform=ax_w.transAxes, fontsize=10, fontweight="bold")

        ax_d = fig.add_subplot(gs[1, col])
        residual = warped_by_iter[it] - target_img
        im_d = ax_d.imshow(residual, cmap="coolwarm", vmin=-res_v, vmax=res_v)
        ax_d.axis("off")
        if col == 0:
            ax_d.text(-0.04, 0.5, row_labels[1], rotation=90, va="center",
                      ha="right", transform=ax_d.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.02)

        ax_p = fig.add_subplot(gs[2, col])
        im_p = ax_p.imshow(phi_mag_by_iter[it], cmap="viridis", vmin=0.0, vmax=phi_v)
        ax_p.axis("off")
        if col == 0:
            ax_p.text(-0.04, 0.5, row_labels[2], rotation=90, va="center",
                      ha="right", transform=ax_p.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02)

        ax_e = fig.add_subplot(gs[3, col])
        im_e = ax_e.imshow(err_map_by_iter[it], cmap="magma", vmin=0.0, vmax=err_v)
        ax_e.axis("off")
        if col == 0:
            ax_e.text(-0.04, 0.5, row_labels[3], rotation=90, va="center",
                      ha="right", transform=ax_e.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02)

    fig.suptitle(_suptitle_str(subject_path, io_optimizer, io_lr, io_sim), fontsize=11)
    fig.text(
        0.5, 0.945,
        "rows: warped (+ grid) | warped - target | ||phi||_2 | error_map = ||phi - phi@0||_2",
        ha="center", va="top", fontsize=9, color="#444",
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved images figure: {save_path}")

    if no_show:
        plt.close(fig)


def render_sweep_curves(
    *,
    metrics: dict[str, list[tuple[int, float]]],
    io_loss_at_iter: dict[int, float],
    lncc_sigma: int,
    save_path: Path | None,
    no_show: bool,
    subject_path: Path,
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
) -> None:
    """Render two panels: (left) quality vs target, (right) downstream-target health."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(15.0, 4.8), gridspec_kw={"wspace": 0.45}
    )

    iters_x = [it for it, _ in metrics["lncc"]]
    lncc_y = [v for _, v in metrics["lncc"]]
    err_y = [v for _, v in metrics["err_mag"]]
    negjac_y = [v for _, v in metrics["negjac_pct"]]
    io_loss_y = [io_loss_at_iter.get(it, float("nan")) for it in iters_x]

    # Left panel: registration quality vs target -- LNCC + the actual IO loss.
    ax_left.plot(iters_x, lncc_y, marker="o", color="C2", label="LNCC  (higher better)")
    ax_left.set_xlabel("IO iterations")
    ax_left.set_ylabel("LNCC", color="C2")
    ax_left.tick_params(axis="y", labelcolor="C2")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_title("Registration quality", fontsize=11)

    ax_left2 = ax_left.twinx()
    ax_left2.plot(iters_x, io_loss_y, marker="s", color="tab:purple",
                  label="io_loss  (lower better)")
    ax_left2.set_ylabel("io_loss", color="tab:purple")
    ax_left2.tick_params(axis="y", labelcolor="tab:purple")
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_l2, labels_l2 = ax_left2.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_l2, labels_l + labels_l2, loc="center right", fontsize=9)

    # Right panel: downstream-target health -- mean(error_map) signal vs neg_jac_pct folds.
    ax_right.plot(iters_x, err_y, marker="o", color="C1",
                  label="mean(error_map)  (signal)")
    ax_right.set_xlabel("IO iterations")
    ax_right.set_ylabel("mean(error_map)  [px]", color="C1")
    ax_right.tick_params(axis="y", labelcolor="C1")
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title("Error-map signal vs folds", fontsize=11)

    ax_right2 = ax_right.twinx()
    ax_right2.plot(iters_x, negjac_y, marker="s", color="C3",
                   label="neg_jac_pct  (folds, lower better)")
    ax_right2.set_ylabel("neg_jac_pct", color="C3")
    ax_right2.tick_params(axis="y", labelcolor="C3")
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    lines_r2, labels_r2 = ax_right2.get_legend_handles_labels()
    ax_right.legend(lines_r + lines_r2, labels_r + labels_r2, loc="lower right", fontsize=9)

    fig.suptitle(_suptitle_str(subject_path, io_optimizer, io_lr, io_sim), fontsize=11)
    # Explicit margins so twinx() right-axis labels never collide with the
    # next subplot's left-axis labels (tight_layout doesn't account for them).
    fig.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.14, wspace=0.45)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved curves figure: {save_path}")

    if no_show:
        plt.close(fig)


def derive_save_paths(
    save_path: Path | None,
    subject_path: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Derive per-subject (images, curves, metrics_csv) paths from a base path.

    Example: with ``save_path=./out/sweep_io.png`` and a subject file
    ``IXI002_T1_slice_111.npy``, returns::

      (./out/sweep_io_IXI002_T1_slice_111_images.png,
       ./out/sweep_io_IXI002_T1_slice_111_curves.png,
       ./out/sweep_io_IXI002_T1_slice_111_metrics.csv)

    Returns ``(None, None, None)`` if ``save_path`` is ``None``.
    """
    if save_path is None:
        return None, None, None
    stem = save_path.stem
    suffix = save_path.suffix or ".png"
    subj_stem = subject_path.stem
    images_path = save_path.with_name(f"{stem}_{subj_stem}_images{suffix}")
    curves_path = save_path.with_name(f"{stem}_{subj_stem}_curves{suffix}")
    csv_path = save_path.with_name(f"{stem}_{subj_stem}_metrics.csv")
    return images_path, curves_path, csv_path


def _sweep_one_subject(
    *,
    net: torch.nn.Module,
    device: torch.device,
    target_img: np.ndarray,
    target_175: torch.Tensor,
    subject_path: Path,
    checkpoints: list[int],
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
    grid_stride: int,
    lncc_sigma: int,
    save_path: Path | None,
    no_show: bool,
) -> None:
    """Run the full sweep + render + CSV save for a single subject."""
    source_img = np.load(subject_path).astype(np.float32)
    sh, sw = int(source_img.shape[0]), int(source_img.shape[1])
    th, tw = int(target_img.shape[0]), int(target_img.shape[1])
    if (sh, sw) != (th, tw):
        raise ValueError(
            f"Shape mismatch {subject_path} ({sh},{sw}) vs atlas ({th},{tw})"
        )

    I_source = torch.from_numpy(source_img).float().unsqueeze(0).unsqueeze(0)
    source_175 = preprocess_for_unigrad(I_source).to(device)

    snapshots, io_loss_at_iter = run_io_with_snapshots(
        net,
        source_175,
        target_175,
        checkpoints=checkpoints,
        lr=io_lr,
        optimizer_name=io_optimizer,
        orig_h=sh,
        orig_w=sw,
    )
    del source_175
    torch.cuda.empty_cache()

    warped_by_iter: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for it, phi_px in snapshots.items():
            warped_by_iter[it] = apply_displacement_2d(source_img, phi_px, device)

    metrics = compute_sweep_metrics(
        target_img=target_img,
        snapshots=snapshots,
        warped_by_iter=warped_by_iter,
        lncc_sigma=lncc_sigma,
    )
    print_sweep_metrics(metrics, io_loss_at_iter=io_loss_at_iter, lncc_sigma=lncc_sigma)

    images_path, curves_path, csv_path = derive_save_paths(save_path, subject_path)
    if csv_path is not None:
        write_metrics_csv(csv_path, metrics, io_loss_at_iter)

    render_sweep_images(
        target_img=target_img,
        snapshots=snapshots,
        warped_by_iter=warped_by_iter,
        grid_stride=grid_stride,
        save_path=images_path,
        no_show=no_show,
        subject_path=subject_path,
        io_lr=io_lr,
        io_sim=io_sim,
        io_optimizer=io_optimizer,
    )
    render_sweep_curves(
        metrics=metrics,
        io_loss_at_iter=io_loss_at_iter,
        lncc_sigma=lncc_sigma,
        save_path=curves_path,
        no_show=no_show,
        subject_path=subject_path,
        io_lr=io_lr,
        io_sim=io_sim,
        io_optimizer=io_optimizer,
    )


def run_sweep(
    ixi_root: Path,
    *,
    split: str,
    subject_path: Path | None,
    subject_indices: list[int] | None,
    num_subjects: int | None,
    seed: int,
    checkpoints: list[int],
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
    grid_stride: int,
    lncc_sigma: int,
    save_path: Path | None,
    no_show: bool,
) -> None:
    """Load atlas + UniGradICON once, then run the sweep for each chosen subject."""
    device = torch.device("cuda")

    subj_paths = resolve_subject_paths(
        ixi_root,
        split=split,
        subject_path=subject_path,
        subject_indices=subject_indices,
        num_subjects=num_subjects,
        seed=seed,
    )
    print(f"Running sweep over {len(subj_paths)} subject(s):")
    for sp in subj_paths:
        print(f"  - {sp}")

    target_img, atlas_i, atlas_path = load_atlas_slice(ixi_root / "Atlas")
    th, tw = int(target_img.shape[0]), int(target_img.shape[1])
    print(f"Target: atlas slice {atlas_i} ({atlas_path}) shape=({th}, {tw})")
    I_target = torch.from_numpy(target_img).float().unsqueeze(0).unsqueeze(0)
    target_175 = preprocess_for_unigrad(I_target).to(device)

    print(f"Loading UniGradICON (IO similarity={io_sim}) on {device}...")
    net = get_unigradicon(loss_fn=make_sim(io_sim)).to(device)
    net.eval()

    print(f"Sweeping IO iterations: {checkpoints}")
    for i, subj_path in enumerate(subj_paths, start=1):
        print(f"\n=== [{i}/{len(subj_paths)}] Subject: {subj_path.name} ===")
        _sweep_one_subject(
            net=net,
            device=device,
            target_img=target_img,
            target_175=target_175,
            subject_path=subj_path,
            checkpoints=checkpoints,
            io_lr=io_lr,
            io_sim=io_sim,
            io_optimizer=io_optimizer,
            grid_stride=grid_stride,
            lncc_sigma=lncc_sigma,
            save_path=save_path,
            no_show=no_show,
        )

    del target_175
    torch.cuda.empty_cache()
    if not no_show:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the IO iteration sweep."""
    p = argparse.ArgumentParser(
        description="Sweep IO iteration counts on a single (source, target) pair and visualize convergence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--ixi-root",
        type=Path,
        default=Path("./data/IXI_2D/"),
        help="Folder with Train/Val/Test/Atlas; atlas slice 111 is the fixed target.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="Train",
        choices=["Train", "Val", "Test"],
        help="Split to pick subject slices from (ignored if --subject-path is set).",
    )
    p.add_argument(
        "--subject-path",
        type=Path,
        default=None,
        help="Explicit subject .npy path (single subject). Overrides everything else.",
    )
    p.add_argument(
        "--subject-indices",
        type=str,
        default=None,
        help="Comma-separated sorted-index list into the split, e.g. '0,7,15'. "
        "Overrides --num-subjects/--seed when given.",
    )
    p.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="How many subjects to randomly sample from the split for the sweep. "
        "If unset (default), falls back to subject 0 unless --subject-indices or "
        "--subject-path is provided.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --num-subjects sampling (default 42 for reproducibility).",
    )
    p.add_argument(
        "--checkpoints",
        type=str,
        default="0,50,100,150,200,250",
        help="Comma-separated IO iteration counts to snapshot. Default 0,50,100,150,200,250 "
        "centres the sweep on the official UniGradICON IO default of 50 iterations.",
    )
    p.add_argument(
        "--io-lr",
        type=float,
        default=2e-5,
        help="Adam LR for IO. Default 2e-5 matches upstream icon_registration.",
    )
    p.add_argument(
        "--io-sim",
        type=str,
        default="lncc",
        choices=["lncc", "lncc2", "mind"],
        help="Similarity loss for IO (matches unigradicon-register --io_sim).",
    )
    p.add_argument(
        "--io-optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer for IO. 'adam' matches the official protocol.",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=12,
        help="Grid line spacing in pixels for the deformation overlay (default: 12).",
    )
    p.add_argument(
        "--lncc-sigma",
        type=int,
        default=5,
        help="Window half-size for LNCC(warped, target). Matches icon_registration "
        "default sigma=5 (window 2*sigma+1 = 11 pixels).",
    )
    p.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Base path for outputs (parent dirs created). For each swept subject "
        "S, three files are written: <stem>_<S>_images<ext>, <stem>_<S>_curves<ext>, "
        "and <stem>_<S>_metrics.csv. E.g. '--save-path ./out/sweep_io.png' with "
        "subject 'IXI002_T1_slice_111.npy' -> sweep_io_IXI002_T1_slice_111_images.png "
        "(and the matching _curves.png / _metrics.csv).",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    checkpoints = parse_checkpoints(args.checkpoints)
    subject_indices: list[int] | None = None
    if args.subject_indices is not None:
        subject_indices = sorted({int(x.strip()) for x in args.subject_indices.split(",") if x.strip()})
    run_sweep(
        args.ixi_root.resolve(),
        split=args.split,
        subject_path=args.subject_path.resolve() if args.subject_path else None,
        subject_indices=subject_indices,
        num_subjects=args.num_subjects,
        seed=args.seed,
        checkpoints=checkpoints,
        io_lr=args.io_lr,
        io_sim=args.io_sim,
        io_optimizer=args.io_optimizer,
        grid_stride=args.grid_stride,
        lncc_sigma=args.lncc_sigma,
        save_path=args.save_path,
        no_show=args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
