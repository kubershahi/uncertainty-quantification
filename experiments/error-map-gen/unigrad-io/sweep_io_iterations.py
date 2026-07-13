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
  - ``neg_jac_pct``  ( = %|J|<0 of T = id + phi ) -- folded voxels (stop before
                                                    this starts climbing)

See ``compute_sweep_metrics`` for the formal definitions.

Outputs per subject (saved next to ``--save-path``)
---------------------------------------------------
  - ``<stem>_<subject>_images.png``  4-row grid (warped+grid, residual,
                                     ||phi||, error_map), one column / ckpt.
  - ``<stem>_<subject>_curves.png``  2-panel curves: quality + field health.
  - ``<stem>_<subject>_metrics.csv`` per-iter metrics.

Example (Nautilus PVC, ``/files`` mounted)::

python experiments/error-map-gen/unigrad-io/sweep_io_iterations.py --ixi-root datasets/IXI --atlas-pkl datasets/IXI/atlas.pkl --split Train --num-subjects 5 --save-path assets/images/error-map/unigrad-io/unigradio_sweep_io.png --no-show

Omit ``--ixi-root`` / ``--save-path`` to use defaults (see argparse help): on NRP,
inputs under ``<repo>/datasets/IXI``; figures under ``<repo>/assets/images/error-map/unigrad-io/3d/``.

Default ``--checkpoints``: ``0,50,100,150,200,250,300``. ``--seed 42``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Matplotlib builds a font cache on first import; the default config dir on NRP PVCs
# (/files, $HOME) is slow — use local tmpfs so the job starts quickly.
os.environ.setdefault("MPLBACKEND", "Agg")
_mpl_cfg = os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib-user")
os.makedirs(_mpl_cfg, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)

import matplotlib

matplotlib.use("Agg")
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
    ATLAS_MASK_FILENAME,
    apply_displacement_3d,
    load_ixi_image_volume_from_pkl,
    numpy_volume_hw_d_to_torch5d,
    phi_vectorfield_to_volume_voxels,
    preprocess_volume_for_unigrad,
)
from visualize_unigrad_io_atlas import load_atlas_bundle  # noqa: E402
from visualize_unigrad_io_data import DISPLACEMENT_UNIT, default_slice_index  # noqa: E402


def overlay_deformation_grid(
    ax,
    phi_px: np.ndarray,
    *,
    stride: int = 12,
    color: str = "cyan",
    linewidth: float = 0.5,
    alpha: float = 0.7,
) -> None:
    """Overlay in-plane grid for ``phi_px`` ``(2, H, W)`` (col, row displacements)."""
    if phi_px.ndim != 3 or phi_px.shape[0] != 2:
        raise ValueError(f"Expected phi_px (2, H, W), got {phi_px.shape}")
    _, h, w = phi_px.shape
    rows = np.arange(h)
    cols = np.arange(w)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    pos_col = grid_col + phi_px[0]
    pos_row = grid_row + phi_px[1]
    levels_col = np.arange(0, w + stride, stride)
    levels_row = np.arange(0, h + stride, stride)
    ax.contour(pos_col, levels=levels_col, colors=color, linewidths=linewidth, alpha=alpha)
    ax.contour(pos_row, levels=levels_row, colors=color, linewidths=linewidth, alpha=alpha)


# Nautilus ``/files`` PVC layout (see ``deploy/nautilus/scripts/env.sh``).
_FILES_ROOT = Path("/files")
NRP_REPO = _FILES_ROOT / "repo" / "uncertainty-quantification"
NRP_DATASETS = NRP_REPO / "datasets"
NRP_IXI_PKL_ROOT = NRP_DATASETS / "IXI"
NRP_SWEEP_SAVE = NRP_REPO / "assets" / "images" / "error-map" / "unigrad-io" / "3d" / "sweep_io.png"
LOCAL_IXI_PKL = Path("./datasets/IXI/")
LOCAL_SWEEP_SAVE = Path("./assets/images/error-map/unigrad-io/3d/sweep_io.png")

CHECKPOINTS_DEFAULT = "0,50,100,150,200,250,300"


def _tqdm_common_kwargs() -> dict:
    """Shared tqdm settings: work in non-TTY (e.g. ``kubectl logs``); set ``TQDM_DISABLE=1`` to hide."""
    return {
        "file": sys.stderr,
        "dynamic_ncols": True,
        "disable": os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes"),
    }


def default_ixi_root() -> Path:
    """Prefer PVC paths when mounted; else repo-relative ``datasets/IXI``."""
    return NRP_IXI_PKL_ROOT if NRP_IXI_PKL_ROOT.is_dir() else LOCAL_IXI_PKL


def default_save_path() -> Path:
    """Sweep figure prefix: ``<repo>/assets/images/error-map/unigrad-io/`` on NRP when repo is on PVC."""
    if NRP_REPO.is_dir():
        return NRP_SWEEP_SAVE
    return LOCAL_SWEEP_SAVE


def lncc_3d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sigma: int = 5,
    device: torch.device | None = None,
) -> float:
    """Mean LNCC between two ``(H, W, D)`` volumes (uniform window ``2*sigma+1`` per axis)."""
    dev = device if device is not None else torch.device("cpu")
    k = 2 * sigma + 1
    pad = sigma
    H, W, Dd = a.shape
    A = torch.from_numpy(a).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    B = torch.from_numpy(b).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    mu_a = F.avg_pool3d(A, kernel_size=k, stride=1, padding=pad)
    mu_b = F.avg_pool3d(B, kernel_size=k, stride=1, padding=pad)
    var_a = F.avg_pool3d(A * A, kernel_size=k, stride=1, padding=pad) - mu_a * mu_a
    var_b = F.avg_pool3d(B * B, kernel_size=k, stride=1, padding=pad) - mu_b * mu_b
    cov_ab = F.avg_pool3d(A * B, kernel_size=k, stride=1, padding=pad) - mu_a * mu_b

    denom = torch.sqrt(torch.clamp(var_a * var_b, min=1e-10))
    lncc_map = cov_ab / denom
    return float(lncc_map.mean().item())


def pct_negative_jacobian_3d(phi_dhw: np.ndarray, *, stride: int = 4) -> float:
    """Approximate % of voxels with negative det(J) for ``T = id + phi`` on a strided interior."""
    phi = np.asarray(phi_dhw, dtype=np.float64)
    _, D, H, W = phi.shape
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    T = np.stack([zz + phi[0], yy + phi[1], xx + phi[2]], axis=0)

    sl = (slice(1, -1, stride), slice(1, -1, stride), slice(1, -1, stride))
    T_sub = [T[i][sl] for i in range(3)]

    grads = [np.gradient(T_sub[i]) for i in range(3)]
    # J[..., i, j] = d T_i / d x_j ; x = (d,h,w) axes 0,1,2
    J = np.stack(
        [
            np.stack([grads[0][0], grads[0][1], grads[0][2]], axis=-1),
            np.stack([grads[1][0], grads[1][1], grads[1][2]], axis=-1),
            np.stack([grads[2][0], grads[2][1], grads[2][2]], axis=-1),
        ],
        axis=-2,
    )
    det = np.linalg.det(J.astype(np.float64))
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
    """Same as ``resolve_subject_paths`` but for ``Train/*.pkl`` (IXI volume pickles)."""
    if subject_path is not None:
        if not subject_path.is_file():
            raise FileNotFoundError(f"--subject-path not found: {subject_path}")
        return [subject_path]
    in_dir = ixi_root / split
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Split directory missing: {in_dir}")
    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl in {in_dir}")
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


def resolve_atlas_hw_d(
    atlas_pkl: Path,
    unigrad_io_dir: Path | None,
) -> tuple[np.ndarray, str]:
    """
    Load fixed atlas ``(H, W, D)``.

    Prefer ``<unigrad_io_dir>/atlas_valid_mask.npz`` (from ``create_unigrad_io_data.py``);
    fall back to ``atlas.pkl``.
    """
    if unigrad_io_dir is not None:
        npz_path = Path(unigrad_io_dir) / ATLAS_MASK_FILENAME
        if npz_path.is_file():
            atlas, _, threshold, fg_pct = load_atlas_bundle(unigrad_io_dir)
            detail = (
                f"{npz_path.name}  threshold={threshold:.6g}  "
                f"fg_percentile={fg_pct:g}"
            )
            return atlas, detail
    atlas = load_ixi_image_volume_from_pkl(atlas_pkl)
    return atlas, str(atlas_pkl)


def phi_axial_overlay_slice(phi_dhw: np.ndarray, d_index: int) -> np.ndarray:
    """Take axial slice ``phi[:, d_index]`` → ``(2, H, W)`` for ``overlay_deformation_grid``."""
    # Channels 2,1 → col, row (in-plane W,H at axial depth).
    return np.stack([phi_dhw[2, d_index], phi_dhw[1, d_index]], axis=0).astype(np.float32)


def write_metrics_csv(
    csv_path: Path,
    metrics: dict[str, list[tuple[int, float]]],
    io_loss_at_iter: dict[int, float],
) -> None:
    """Write the per-iteration metrics for one subject to a CSV file.

    Columns: iter, io_loss, lncc, mean_phi_vox, mean_error_map_vox,
             err_map_p50_vox, err_map_p95_vox, err_map_max_vox, neg_jac_pct.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    iters = [it for it, _ in metrics["lncc"]]
    fieldnames = [
        "iter",
        "io_loss",
        "lncc",
        "mean_phi_vox",
        "mean_error_map_vox",
        "err_map_p50_vox",
        "err_map_p95_vox",
        "err_map_max_vox",
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
                    "mean_phi_vox": metrics["phi_mag"][i][1],
                    "mean_error_map_vox": metrics["err_mag"][i][1],
                    "err_map_p50_vox": metrics["err_p50"][i][1],
                    "err_map_p95_vox": metrics["err_p95"][i][1],
                    "err_map_max_vox": metrics["err_max"][i][1],
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
    phi_extractor: Callable[[torch.nn.Module], np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Run a single IO trajectory and snapshot ``phi`` at each checkpoint.

    ``phi_extractor(net)`` returns voxel displacement ``(3, D, H, W)`` for ``apply_displacement_3d``.
    """
    device = source_175.device
    state0_cpu = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    snapshots: dict[int, np.ndarray] = {}
    io_loss_at_iter: dict[int, float] = {}

    if 0 in checkpoints:
        with torch.no_grad():
            loss_tuple = net(source_175, target_175)
            io_loss_at_iter[0] = float(loss_tuple[0].detach().item())
            snapshots[0] = phi_extractor(net)
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
    pbar = tqdm(total=total_iters, desc="IO iters", **_tqdm_common_kwargs())
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
                snapshots[target_iter] = phi_extractor(net)
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
      - ``"phi_mag"``    : mean ||phi@N|| in voxels
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
    if target_img.ndim != 3:
        raise ValueError(f"Expected target volume (H, W, D), got {target_img.shape}")
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
        lncc.append((it, lncc_3d(warped, target_img, sigma=lncc_sigma, device=metric_device)))
        negjac.append((it, pct_negative_jacobian_3d(phi)))
        phi_mag.append((it, float(np.mean(np.sqrt(np.sum(phi * phi, axis=0))))))
        err_mag.append((it, float(np.mean(emap))))
        err_p50.append((it, float(np.percentile(emap, 50.0))))
        err_p95.append((it, float(np.percentile(emap, 95.0))))
        err_max.append((it, float(np.max(emap))))
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
    displacement_unit: str,
    header_note: str = "",
) -> None:
    """Pretty-print the sweep metrics table to stdout.

    ``io_loss`` is the actual quantity the IO optimizer minimizes (e.g. -LNCC +
    regulariser inside the network). It should decrease monotonically when IO
    is descending; this is the most direct sanity check that the optimiser is
    doing real work, independent of any external evaluation metric.
    """
    atlas_summary = header_note.strip() if header_note else "3D atlas vs subject volume"
    print(
        f"\nSweep metrics ({atlas_summary}, LNCC sigma={lncc_sigma}, "
        "io_loss = quantity Adam descends; mean(error_map) = signal magnitude "
        "of the U-Net regression target if --io-iterations=N is picked):"
    )
    du = displacement_unit
    print(
        f"  {'iter':>6s} | {'io_loss':>9s} | {'LNCC':>7s} | "
        f"{f'mean||phi|| ({du})':>17s} | {f'mean(error_map) ({du})':>17s} | {'neg_jac_pct':>11s}"
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
        f"\nError-map shape ({du}). p50 < mean < p95 ~ structured signal; "
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


def figure_heading(
    subject_path: Path,
    *,
    io_optimizer: str,
    io_lr: float,
    io_sim: str,
    mode: str,
    detail: str | None = None,
) -> tuple[str, str]:
    """Main title and subtitle for sweep figures."""
    main = f"UniGrad ICON IO sweep · {subject_path.stem}"
    bits = [mode]
    if detail:
        bits.append(detail)
    bits.append(f"{io_optimizer} lr={io_lr:g}")
    bits.append(io_sim)
    return main, " · ".join(bits)


def apply_figure_heading(
    fig: plt.Figure,
    subject_path: Path,
    *,
    io_optimizer: str,
    io_lr: float,
    io_sim: str,
    mode: str,
    detail: str | None = None,
    y_main: float = 0.98,
) -> None:
    main, sub = figure_heading(
        subject_path,
        io_optimizer=io_optimizer,
        io_lr=io_lr,
        io_sim=io_sim,
        mode=mode,
        detail=detail,
    )
    fig.suptitle(main, fontsize=12, y=y_main, fontweight="medium")
    fig.text(0.5, y_main - 0.028, sub, ha="center", va="top", fontsize=9, color="#555555")


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
    axial_slice_idx: int | None = None,
) -> None:
    """Render the 4-row image grid: warped+grid / residual / ||phi|| / error_map."""
    iters_sorted = sorted(snapshots.keys())
    n_iters = len(iters_sorted)
    fig = plt.figure(figsize=(3.0 * max(n_iters, 4), 12.5))
    gs = fig.add_gridspec(
        4, n_iters, height_ratios=[3.0, 3.0, 3.0, 3.0], hspace=0.25, wspace=0.05
    )

    phi0 = snapshots[iters_sorted[0]]

    if target_img.ndim != 3:
        raise ValueError(f"Expected target volume (H, W, D), got {target_img.shape}")
    hwd_d = int(target_img.shape[2])
    mid = (
        int(axial_slice_idx)
        if axial_slice_idx is not None
        else default_slice_index(hwd_d)
    )
    mid = max(0, min(mid, hwd_d - 1))
    slice_detail = f"axial z={mid}"

    def axial_tile(vol_hw_d: np.ndarray) -> np.ndarray:
        return vol_hw_d[:, :, mid]

    def axial_mag(phi_dhw: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(phi_dhw**2, axis=0))[mid, :, :]

    def axial_err(phi_dhw: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((phi_dhw - phi0) ** 2, axis=0))[mid, :, :]

    target_tile = axial_tile(target_img)

    residuals_concat = np.concatenate(
        [(axial_tile(warped_by_iter[it]) - target_tile).ravel() for it in iters_sorted]
    )
    res_v = max(float(np.percentile(np.abs(residuals_concat), 99.0)), 1e-6)

    phi_mag_by_iter: dict[int, np.ndarray] = {
        it: axial_mag(snapshots[it]) for it in iters_sorted
    }
    err_map_by_iter: dict[int, np.ndarray] = {
        it: axial_err(snapshots[it]) for it in iters_sorted
    }
    phi_mag_concat = np.concatenate([m.ravel() for m in phi_mag_by_iter.values()])
    err_map_concat = np.concatenate([m.ravel() for m in err_map_by_iter.values()])
    phi_v = max(float(np.percentile(phi_mag_concat, 99.0)), 1e-6)
    err_v = max(float(np.percentile(err_map_concat, 99.0)), 1e-6)

    row_labels = ["warped", "residual", r"$\|\phi\|$", "error_map"]

    for col, it in enumerate(iters_sorted):
        ax_w = fig.add_subplot(gs[0, col])
        warped_tile = axial_tile(warped_by_iter[it])
        ax_w.imshow(warped_tile, cmap="gray")
        ax_w.set_title(f"iter@{it}", fontsize=10)
        ax_w.axis("off")
        overlay_deformation_grid(
            ax_w, phi_axial_overlay_slice(snapshots[it], mid), stride=grid_stride
        )
        if col == 0:
            ax_w.text(-0.04, 0.5, row_labels[0], rotation=90, va="center",
                      ha="right", transform=ax_w.transAxes, fontsize=10, fontweight="bold")

        ax_d = fig.add_subplot(gs[1, col])
        residual = warped_tile - target_tile
        im_d = ax_d.imshow(residual, cmap="coolwarm", vmin=-res_v, vmax=res_v)
        ax_d.axis("off")
        if col == 0:
            ax_d.text(-0.04, 0.5, row_labels[1], rotation=90, va="center",
                      ha="right", transform=ax_d.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.02)
            cb_d.set_label("intensity difference", fontsize=8)

        ax_p = fig.add_subplot(gs[2, col])
        im_p = ax_p.imshow(phi_mag_by_iter[it], cmap="viridis", vmin=0.0, vmax=phi_v)
        ax_p.axis("off")
        if col == 0:
            ax_p.text(-0.04, 0.5, row_labels[2], rotation=90, va="center",
                      ha="right", transform=ax_p.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            cb_p = fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.02)
            cb_p.set_label(f"displacement ({DISPLACEMENT_UNIT})", fontsize=8)

        ax_e = fig.add_subplot(gs[3, col])
        im_e = ax_e.imshow(err_map_by_iter[it], cmap="magma", vmin=0.0, vmax=err_v)
        ax_e.axis("off")
        if col == 0:
            ax_e.text(-0.04, 0.5, row_labels[3], rotation=90, va="center",
                      ha="right", transform=ax_e.transAxes, fontsize=10, fontweight="bold")
        if col == n_iters - 1:
            cb_e = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02)
            cb_e.set_label(f"‖Δφ‖ ({DISPLACEMENT_UNIT})", fontsize=8)

    apply_figure_heading(
        fig,
        subject_path,
        io_optimizer=io_optimizer,
        io_lr=io_lr,
        io_sim=io_sim,
        mode="3D volume",
        detail=slice_detail,
        y_main=0.99,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))

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
    displacement_unit: str,
    save_path: Path | None,
    no_show: bool,
    subject_path: Path,
    io_lr: float,
    io_sim: str,
    io_optimizer: str,
    mode: str,
    detail: str | None = None,
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
    ax_left.plot(iters_x, lncc_y, marker="o", color="C2", label="LNCC")
    ax_left.set_xlabel("IO iterations")
    ax_left.set_ylabel("LNCC", color="C2")
    ax_left.tick_params(axis="y", labelcolor="C2")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_title("LNCC and IO loss", fontsize=11)

    ax_left2 = ax_left.twinx()
    ax_left2.plot(iters_x, io_loss_y, marker="s", color="tab:purple", label="IO loss")
    ax_left2.set_ylabel("io_loss", color="tab:purple")
    ax_left2.tick_params(axis="y", labelcolor="tab:purple")
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_l2, labels_l2 = ax_left2.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_l2, labels_l + labels_l2, loc="center right", fontsize=9)

    # Right panel: downstream-target health -- mean(error_map) signal vs neg_jac_pct folds.
    ax_right.plot(iters_x, err_y, marker="o", color="C1", label="mean(error_map)")
    ax_right.set_xlabel("IO iterations")
    ax_right.set_ylabel(f"mean(error_map)  [{displacement_unit}]", color="C1")
    ax_right.tick_params(axis="y", labelcolor="C1")
    ax_right.grid(True, alpha=0.3)
    ax_right.set_title("Error map and Jacobian folds", fontsize=11)

    ax_right2 = ax_right.twinx()
    ax_right2.plot(iters_x, negjac_y, marker="s", color="C3", label="neg. Jacobian %")
    ax_right2.set_ylabel("neg_jac_pct", color="C3")
    ax_right2.tick_params(axis="y", labelcolor="C3")
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    lines_r2, labels_r2 = ax_right2.get_legend_handles_labels()
    ax_right.legend(lines_r + lines_r2, labels_r + labels_r2, loc="lower right", fontsize=9)

    apply_figure_heading(
        fig,
        subject_path,
        io_optimizer=io_optimizer,
        io_lr=io_lr,
        io_sim=io_sim,
        mode=mode,
        detail=detail,
        y_main=0.97,
    )
    # Explicit margins so twinx() right-axis labels never collide with the
    # next subplot's left-axis labels (tight_layout doesn't account for them).
    fig.subplots_adjust(left=0.07, right=0.94, top=0.82, bottom=0.14, wspace=0.45)

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

    Example: with ``save_path=./out/sweep_io.png`` and subject ``subject_042.pkl``::

(./out/sweep_io_subject_042_images.png,
./out/sweep_io_subject_042_curves.png,
./out/sweep_io_subject_042_metrics.csv)

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
    target_vol: np.ndarray,
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
    viz_axial_index: int | None,
) -> None:
    """Sweep one ``Train/*.pkl`` subject against the full atlas volume."""
    source_vol = load_ixi_image_volume_from_pkl(subject_path)
    if source_vol.shape != target_vol.shape:
        raise ValueError(
            f"Shape mismatch {subject_path} {source_vol.shape} vs atlas {target_vol.shape}"
        )
    oh, ow, od = int(source_vol.shape[0]), int(source_vol.shape[1]), int(source_vol.shape[2])

    vol_5d = numpy_volume_hw_d_to_torch5d(source_vol).to(device)
    source_175 = preprocess_volume_for_unigrad(vol_5d)
    del vol_5d

    snapshots, io_loss_at_iter = run_io_with_snapshots(
        net,
        source_175,
        target_175,
        checkpoints=checkpoints,
        lr=io_lr,
        optimizer_name=io_optimizer,
        phi_extractor=lambda m: phi_vectorfield_to_volume_voxels(m, od, oh, ow),
    )
    del source_175
    torch.cuda.empty_cache()

    warped_by_iter: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for it, phi_vox in snapshots.items():
            warped_by_iter[it] = apply_displacement_3d(source_vol, phi_vox, device)

    viz_mid = (
        int(viz_axial_index)
        if viz_axial_index is not None
        else default_slice_index(int(target_vol.shape[2]))
    )
    viz_mid = max(0, min(viz_mid, int(target_vol.shape[2]) - 1))
    slice_detail = f"axial z={viz_mid}"
    metrics_header = (
        "3D atlas vs subject — volume LNCC; neg_jac_pct = strided interior det(J)<0"
    )

    metrics = compute_sweep_metrics(
        target_img=target_vol,
        snapshots=snapshots,
        warped_by_iter=warped_by_iter,
        lncc_sigma=lncc_sigma,
    )
    print_sweep_metrics(
        metrics,
        io_loss_at_iter=io_loss_at_iter,
        lncc_sigma=lncc_sigma,
        displacement_unit=DISPLACEMENT_UNIT,
        header_note=metrics_header,
    )

    images_path, curves_path, csv_path = derive_save_paths(save_path, subject_path)
    if csv_path is not None:
        write_metrics_csv(csv_path, metrics, io_loss_at_iter)

    render_sweep_images(
        target_img=target_vol,
        snapshots=snapshots,
        warped_by_iter=warped_by_iter,
        grid_stride=grid_stride,
        save_path=images_path,
        no_show=no_show,
        subject_path=subject_path,
        io_lr=io_lr,
        io_sim=io_sim,
        io_optimizer=io_optimizer,
        axial_slice_idx=viz_axial_index,
    )
    render_sweep_curves(
        metrics=metrics,
        io_loss_at_iter=io_loss_at_iter,
        lncc_sigma=lncc_sigma,
        displacement_unit=DISPLACEMENT_UNIT,
        save_path=curves_path,
        no_show=no_show,
        subject_path=subject_path,
        io_lr=io_lr,
        io_sim=io_sim,
        io_optimizer=io_optimizer,
        mode="3D volume",
        detail=slice_detail,
    )


def run_sweep(
    ixi_root: Path,
    *,
    atlas_pkl: Path,
    unigrad_io_dir: Path | None,
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
    viz_axial_index: int | None,
) -> None:
    """Load full atlas + UniGradICON once; sweep IO checkpoints per ``*.pkl`` volume."""
    device = torch.device("cuda")

    subj_paths = resolve_subject_paths(
        ixi_root,
        split=split,
        subject_path=subject_path,
        subject_indices=subject_indices,
        num_subjects=num_subjects,
        seed=seed,
    )
    print(f"Running sweep over {len(subj_paths)} subject volume(s):")
    for sp in subj_paths:
        print(f"  - {sp}")

    if unigrad_io_dir is None and not atlas_pkl.is_file():
        raise FileNotFoundError(f"--atlas-pkl not found: {atlas_pkl}")

    target_vol, atlas_src = resolve_atlas_hw_d(atlas_pkl, unigrad_io_dir)
    oh, ow, od = int(target_vol.shape[0]), int(target_vol.shape[1]), int(target_vol.shape[2])
    print(f"Atlas (H,W,D)=({oh},{ow},{od})  source: {atlas_src}")

    atlas_5d = numpy_volume_hw_d_to_torch5d(target_vol).to(device)
    target_175 = preprocess_volume_for_unigrad(atlas_5d)
    del atlas_5d

    print(f"Loading UniGradICON (IO similarity={io_sim}) on {device}...")
    net = get_unigradicon(loss_fn=make_sim(io_sim)).to(device)
    net.eval()

    print(f"Sweeping IO iterations: {checkpoints}")
    for subj_path in tqdm(
        subj_paths,
        desc="subjects",
        unit="vol",
        **_tqdm_common_kwargs(),
    ):
        tqdm.write(f"\n=== Subject volume: {subj_path.name} ===")
        _sweep_one_subject(
            net=net,
            device=device,
            target_vol=target_vol,
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
            viz_axial_index=viz_axial_index,
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
        default=None,
        metavar="DIR",
        help="IXI root with Train|Val|Test/*.pkl (default: <repo>/datasets/IXI).",
    )
    p.add_argument(
        "--atlas-pkl",
        type=Path,
        default=None,
        help="Atlas volume atlas.pkl (default: <ixi-root>/atlas.pkl).",
    )
    p.add_argument(
        "--unigrad-io-dir",
        type=Path,
        default=None,
        help="If set and atlas_valid_mask.npz exists, load atlas from there "
        "(e.g. datasets/error-map/unigrad-io/ixi).",
    )
    p.add_argument(
        "--viz-axial-index",
        type=int,
        default=None,
        metavar="I",
        help="Axial slice index along volume axis D (H,W,D layout); default: middle.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="Train",
        choices=["Train", "Val", "Test"],
        help="Split to pick subject volumes from (ignored if --subject-path is set).",
    )
    p.add_argument(
        "--subject-path",
        type=Path,
        default=None,
        help="Explicit subject .pkl path. Overrides sampling.",
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
        default=None,
        metavar="I,I,...",
        help=f"Comma-separated IO iteration counts to snapshot (default: {CHECKPOINTS_DEFAULT}).",
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
        help="Grid line spacing (voxels) for the deformation overlay (default: 12).",
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
        "and <stem>_<S>_metrics.csv. Default on NRP: "
        "/files/repo/uncertainty-quantification/assets/images/error-map/unigrad-io/3d/sweep_io.png.",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if args.ixi_root is None:
        args.ixi_root = default_ixi_root()
    if args.save_path is None:
        args.save_path = default_save_path()
    ck_spec = args.checkpoints if args.checkpoints is not None else CHECKPOINTS_DEFAULT
    checkpoints = parse_checkpoints(ck_spec)
    subject_indices: list[int] | None = None
    if args.subject_indices is not None:
        subject_indices = sorted({int(x.strip()) for x in args.subject_indices.split(",") if x.strip()})
    subj = args.subject_path.resolve() if args.subject_path else None
    atlas_pkl = (
        args.atlas_pkl.resolve()
        if args.atlas_pkl is not None
        else (args.ixi_root / "atlas.pkl").resolve()
    )
    unigrad_io_dir = args.unigrad_io_dir.resolve() if args.unigrad_io_dir else None
    run_sweep(
        args.ixi_root.resolve(),
        atlas_pkl=atlas_pkl,
        unigrad_io_dir=unigrad_io_dir,
        split=args.split,
        subject_path=subj,
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
        viz_axial_index=args.viz_axial_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
