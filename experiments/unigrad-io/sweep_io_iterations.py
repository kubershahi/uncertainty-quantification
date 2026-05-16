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

Example (Nautilus PVC, ``/files`` mounted):
  python experiments/unigrad-io/sweep_io_iterations.py --split Train --num-subjects 3 \\
      --save-path /files/repo/uncertainty-quantification/assets/images/unigrad-io/sweep_io.png --no-show
  python experiments/unigrad-io/sweep_io_iterations.py --mode 3d-pkl \\
      --ixi-root /files/repo/uncertainty-quantification/datasets/IXI \\
      --atlas-pkl /files/repo/uncertainty-quantification/datasets/IXI/atlas.pkl \\
      --split Train --num-subjects 3 \\
      --save-path /files/repo/uncertainty-quantification/assets/images/unigrad-io/3d/sweep_io.png --no-show

Omit ``--ixi-root`` / ``--save-path`` to use defaults (see argparse help): on NRP,
inputs under ``<repo>/datasets/``; sweep figures under ``<repo>/assets/images/unigrad-io/``.

Defaults: ``--checkpoints`` is ``0,50,100,150,200,250`` for 2d and ``0,50,100,200`` for
``3d-pkl`` (four IO snapshots to limit 3d cost). ``--seed 42``.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
import warnings
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
    ATLAS_SLICE_INDEX,
    apply_displacement_2d,
    load_atlas_slice,
    phi_vectorfield_to_slice_pixels,
    preprocess_for_unigrad,
)
from visualize_unigrad_io_data import overlay_deformation_grid  # noqa: E402

# Nautilus ``/files`` PVC layout (see ``deploy/nautilus/scripts/env.sh``).
_FILES_ROOT = Path("/files")
NRP_REPO = _FILES_ROOT / "repo" / "uncertainty-quantification"
NRP_DATASETS = NRP_REPO / "datasets"
NRP_IXI_PKL_ROOT = NRP_DATASETS / "IXI"
NRP_IXI_2D_ROOT = NRP_DATASETS / "IXI_2D"
NRP_SWEEP_SAVE = NRP_REPO / "assets" / "images" / "unigrad-io" / "sweep_io.png"
LOCAL_IXI_PKL = Path("./data/IXI/")
LOCAL_IXI_2D = Path("./data/IXI_2D/")
LOCAL_SWEEP_SAVE = Path("./assets/images/unigrad-io/sweep_io.png")

# IO iteration snapshots (UniGradICON default IO length is 50; full 2d sweep adds dense samples).
CHECKPOINTS_DEFAULT_2D = "0,50,100,150,200,250"
CHECKPOINTS_DEFAULT_3D = "0,50,100,200"


def _tqdm_common_kwargs() -> dict:
    """Shared tqdm settings: work in non-TTY (e.g. ``kubectl logs``); set ``TQDM_DISABLE=1`` to hide."""
    return {
        "file": sys.stderr,
        "dynamic_ncols": True,
        "disable": os.environ.get("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes"),
    }


def default_ixi_root(mode: str) -> Path:
    """Prefer PVC paths when mounted; else small repo-relative roots."""
    if mode == "3d-pkl":
        return NRP_IXI_PKL_ROOT if NRP_IXI_PKL_ROOT.is_dir() else LOCAL_IXI_PKL
    return NRP_IXI_2D_ROOT if NRP_IXI_2D_ROOT.is_dir() else LOCAL_IXI_2D


def default_save_path() -> Path:
    """Sweep figure prefix: ``<repo>/assets/images/unigrad-io/`` on NRP when repo is on PVC."""
    if NRP_REPO.is_dir():
        return NRP_SWEEP_SAVE
    return LOCAL_SWEEP_SAVE


def pkload(path: Path):
    """Load a pickle file (IXI volumes as ``(image, label)`` tuples)."""
    # Pickled ndarrays from older NumPy can reconstruct dtypes with ``align=0``;
    # NumPy 2.4+ deprecates that and emits a warning during unpickling.
    with path.open("rb") as f, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*align should be passed as Python or NumPy boolean.*",
        )
        return pickle.load(f)


def numpy_volume_hw_d_to_torch5d(vol_hw_d: np.ndarray) -> torch.Tensor:
    """``(H, W, D)`` float volume → ``(1, 1, D, H, W)`` for UniGradICON / conv3d."""
    t = torch.from_numpy(vol_hw_d.astype(np.float32))
    return t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


def preprocess_volume_for_unigrad(vol_5d: torch.Tensor) -> torch.Tensor:
    """Normalize full 3D volume and resize to ``175³`` (real anatomy, not pseudo-slices)."""
    im_min = torch.min(vol_5d)
    im_max = torch.quantile(vol_5d.reshape(-1), 0.99)
    denom = torch.clamp(im_max - im_min, min=1e-5)
    img = torch.clip(vol_5d, im_min, im_max)
    img = (img - im_min) / denom
    return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)


def phi_vectorfield_to_volume_voxels(
    net: torch.nn.Module, orig_d: int, orig_h: int, orig_w: int
) -> np.ndarray:
    """ICON displacement in voxel units, shape ``(3, D, H, W)`` matching torch5d layout."""
    identity = net.identity_map
    phi_disp_175 = net.phi_AB_vectorfield - identity
    phi_rescaled = F.interpolate(
        phi_disp_175,
        [orig_d, orig_h, orig_w],
        mode="trilinear",
        align_corners=True,
    )
    p = phi_rescaled[0].cpu().numpy()
    out = np.zeros((3, orig_d, orig_h, orig_w), dtype=np.float32)
    out[0] = p[0] * (orig_d - 1)
    out[1] = p[1] * (orig_h - 1)
    out[2] = p[2] * (orig_w - 1)
    return out


def apply_displacement_3d(
    moving_hw_d: np.ndarray, phi_dhw: np.ndarray, device: torch.device
) -> np.ndarray:
    """Warp ``(H, W, D)`` volume with ``phi_dhw`` ``(3, D, H, W)`` (voxel shifts along D/H/W)."""
    h, w, d = int(moving_hw_d.shape[0]), int(moving_hw_d.shape[1]), int(moving_hw_d.shape[2])
    od, oh, ow = phi_dhw.shape[1], phi_dhw.shape[2], phi_dhw.shape[3]
    if (od, oh, ow) != (d, h, w):
        raise ValueError(f"phi spatial shape {(od, oh, ow)} vs volume {(d, h, w)} (D,H,W)")

    vol = torch.from_numpy(moving_hw_d).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    phi_t = torch.from_numpy(phi_dhw).to(device, dtype=torch.float32)

    zs = torch.arange(d, device=device, dtype=torch.float32)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")

    src_x = grid_x + phi_t[2]
    src_y = grid_y + phi_t[1]
    src_z = grid_z + phi_t[0]
    src_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    src_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    src_z = 2.0 * src_z / max(d - 1, 1) - 1.0
    grid = torch.stack([src_x, src_y, src_z], dim=-1).unsqueeze(0)

    warped = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


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


def resolve_subject_paths_pkl(
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


def load_ixi_image_volume_from_pkl(pkl_path: Path) -> np.ndarray:
    """Load ``image`` array from an IXI-style pickle ``(image, label)`` tuple."""
    payload = pkload(pkl_path)
    if isinstance(payload, tuple) and len(payload) >= 1:
        raw: object = payload[0]
    elif isinstance(payload, np.ndarray):
        raw = payload
    else:
        raise ValueError(f"Unexpected pickle structure in {pkl_path}: {type(payload)}")
    # New float32 allocation so dtype metadata from the pickle cannot leak downstream.
    img = np.array(raw, dtype=np.float32, copy=True)
    if img.ndim != 3:
        raise ValueError(f"Expected 3D volume in {pkl_path}, got shape {img.shape}")
    return img


def load_atlas_volume_from_pkl(atlas_pkl: Path) -> np.ndarray:
    """Alias for atlas.pkl loading (same schema as subject pickles)."""
    return load_ixi_image_volume_from_pkl(atlas_pkl)


def phi_axial_overlay_slice(phi_dhw: np.ndarray, d_index: int) -> np.ndarray:
    """Take axial slice ``phi[:, d_index]`` → ``(2, H, W)`` for ``overlay_deformation_grid``."""
    # Channels 2,1 → col, row (match 2D convention used with ICON channels W,H).
    return np.stack([phi_dhw[2, d_index], phi_dhw[1, d_index]], axis=0).astype(np.float32)


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
    phi_extractor: Callable[[torch.nn.Module], np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Run a single IO trajectory and snapshot ``phi`` at each checkpoint.

    ``phi_extractor(net)`` returns displacement in physical indexing space used by
    ``apply_displacement_*`` — either ``(2, H, W)`` (2D slices) or ``(3, D, H, W)`` (volumes).
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
    is_vol = target_img.ndim == 3
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
        if is_vol:
            lncc.append((it, lncc_3d(warped, target_img, sigma=lncc_sigma, device=metric_device)))
            negjac.append((it, pct_negative_jacobian_3d(phi)))
        else:
            lncc.append((it, lncc_2d(warped, target_img, sigma=lncc_sigma, device=metric_device)))
            negjac.append((it, pct_negative_jacobian_2d(phi)))
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
    header_note: str = "",
) -> None:
    """Pretty-print the sweep metrics table to stdout.

    ``io_loss`` is the actual quantity the IO optimizer minimizes (e.g. -LNCC +
    regulariser inside the network). It should decrease monotonically when IO
    is descending; this is the most direct sanity check that the optimiser is
    doing real work, independent of any external evaluation metric.
    """
    atlas_summary = header_note.strip() if header_note else f"atlas slice {ATLAS_SLICE_INDEX}"
    print(
        f"\nSweep metrics ({atlas_summary}, LNCC sigma={lncc_sigma}, "
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


def _suptitle_str(
    subject_path: Path,
    io_optimizer: str,
    io_lr: float,
    io_sim: str,
    *,
    atlas_note: str | None = None,
) -> str:
    atlas_part = atlas_note if atlas_note else f"atlas slice {ATLAS_SLICE_INDEX}"
    return (
        f"IO iteration sweep — {subject_path.name}  |  "
        f"{atlas_part}, opt={io_optimizer}, lr={io_lr:g}, sim={io_sim}"
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
    axial_slice_idx: int | None = None,
    atlas_note: str | None = None,
) -> None:
    """Render the 4-row image grid: warped+grid / residual / ||phi|| / error_map."""
    iters_sorted = sorted(snapshots.keys())
    n_iters = len(iters_sorted)
    fig = plt.figure(figsize=(3.0 * max(n_iters, 4), 12.5))
    gs = fig.add_gridspec(
        4, n_iters, height_ratios=[3.0, 3.0, 3.0, 3.0], hspace=0.25, wspace=0.05
    )

    phi0 = snapshots[iters_sorted[0]]

    if target_img.ndim == 3:
        hwd_h, hwd_w, hwd_d = target_img.shape
        mid = int(axial_slice_idx) if axial_slice_idx is not None else hwd_d // 2
        mid = max(0, min(mid, hwd_d - 1))

        def axial_tile(vol_hw_d: np.ndarray) -> np.ndarray:
            return vol_hw_d[:, :, mid]

        def axial_mag(phi_dhw: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum(phi_dhw**2, axis=0))[mid, :, :]

        def axial_err(phi_dhw: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum((phi_dhw - phi0) ** 2, axis=0))[mid, :, :]

        row_notes = (
            f"axial slice index {mid} of volume (H,W,D)=({hwd_h},{hwd_w},{hwd_d}); "
            "grid overlay uses in-plane displacements at this depth"
        )
    else:
        mid = None

        def axial_tile(vol_hw_d: np.ndarray) -> np.ndarray:
            return vol_hw_d

        def axial_mag(phi_dhw: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum(phi_dhw**2, axis=0))

        def axial_err(phi_dhw: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum((phi_dhw - phi0) ** 2, axis=0))

        row_notes = ""

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

    row_labels = ["warped", "residual", "||phi|| (px)", "error_map (px)"]

    for col, it in enumerate(iters_sorted):
        ax_w = fig.add_subplot(gs[0, col])
        warped_tile = axial_tile(warped_by_iter[it])
        ax_w.imshow(warped_tile, cmap="gray")
        ax_w.set_title(f"iter {it}", fontsize=10)
        ax_w.axis("off")
        if mid is not None:
            overlay_deformation_grid(ax_w, phi_axial_overlay_slice(snapshots[it], mid), stride=grid_stride)
        else:
            overlay_deformation_grid(ax_w, snapshots[it], stride=grid_stride)
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

    fig.suptitle(_suptitle_str(subject_path, io_optimizer, io_lr, io_sim, atlas_note=atlas_note), fontsize=11)
    subtitle_rows = (
        "rows: warped (+ grid) | warped - target | ||phi||_2 | error_map = ||phi - phi@0||_2"
    )
    if row_notes:
        subtitle_rows += f"\n{row_notes}"
    fig.text(
        0.5, 0.945,
        subtitle_rows,
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
    atlas_note: str | None = None,
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

    fig.suptitle(_suptitle_str(subject_path, io_optimizer, io_lr, io_sim, atlas_note=atlas_note), fontsize=11)
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
        phi_extractor=lambda m: phi_vectorfield_to_slice_pixels(m, sh, sw),
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
    print_sweep_metrics(
        metrics,
        io_loss_at_iter=io_loss_at_iter,
        lncc_sigma=lncc_sigma,
        header_note="",
    )

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
        axial_slice_idx=None,
        atlas_note=None,
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
        atlas_note=None,
    )


def _sweep_one_subject_volume_pkl(
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

    atlas_note = "full atlas vs subject volumes (.pkl)"
    metrics_header = (
        "3D atlas vs moving volume — LNCC over volume; neg_jac_pct via strided "
        "interior det(J)<0 (approx)"
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
        atlas_note=atlas_note,
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
        atlas_note=atlas_note,
    )


def run_sweep_volume_pkl(
    ixi_root: Path,
    *,
    atlas_pkl: Path,
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

    subj_paths = resolve_subject_paths_pkl(
        ixi_root,
        split=split,
        subject_path=subject_path,
        subject_indices=subject_indices,
        num_subjects=num_subjects,
        seed=seed,
    )
    print(f"[3d-pkl] Running sweep over {len(subj_paths)} subject volume(s):")
    for sp in subj_paths:
        print(f"  - {sp}")

    if not atlas_pkl.is_file():
        raise FileNotFoundError(f"--atlas-pkl not found: {atlas_pkl}")

    target_vol = load_atlas_volume_from_pkl(atlas_pkl)
    oh, ow, od = int(target_vol.shape[0]), int(target_vol.shape[1]), int(target_vol.shape[2])
    print(f"Target: atlas volume {atlas_pkl} shape (H,W,D)=({oh},{ow},{od})")

    atlas_5d = numpy_volume_hw_d_to_torch5d(target_vol).to(device)
    target_175 = preprocess_volume_for_unigrad(atlas_5d)
    del atlas_5d

    print(f"Loading UniGradICON (IO similarity={io_sim}) on {device}...")
    net = get_unigradicon(loss_fn=make_sim(io_sim)).to(device)
    net.eval()

    print(f"Sweeping IO iterations: {checkpoints}")
    for subj_path in tqdm(
        subj_paths,
        desc="3d-pkl subjects",
        unit="vol",
        **_tqdm_common_kwargs(),
    ):
        tqdm.write(f"\n=== Subject volume: {subj_path.name} ===")
        _sweep_one_subject_volume_pkl(
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
        "--mode",
        type=str,
        default="2d",
        choices=["2d", "3d-pkl"],
        help="2d: IXI_2D .npy slices + atlas_slice_111 (default). "
        "3d-pkl: IXI Train/*.pkl volumes + atlas.pkl (default root on NRP: "
        "<repo>/datasets/IXI).",
    )
    p.add_argument(
        "--ixi-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Data root: 2d expects Train/Val/Test/Atlas .npy; 3d-pkl expects "
        "Train|Val|Test/*.pkl plus atlas.pkl (default: <repo>/datasets/IXI_2D or "
        "./data/IXI_2D for 2d; <repo>/datasets/IXI or ./data/IXI for 3d-pkl when "
        "--ixi-root is omitted).",
    )
    p.add_argument(
        "--atlas-pkl",
        type=Path,
        default=None,
        help="[3d-pkl] atlas.pkl path (default: <ixi-root>/atlas.pkl).",
    )
    p.add_argument(
        "--viz-axial-index",
        type=int,
        default=None,
        metavar="I",
        help="[3d-pkl] Axial slice index along numpy volume axis D (H,W,D layout); "
        "default: middle slice.",
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
        help="Explicit subject path — .npy (2d) or .pkl (3d-pkl). Overrides sampling.",
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
        help="Comma-separated IO iteration counts to snapshot. If omitted: 2d uses "
        f"{CHECKPOINTS_DEFAULT_2D}; 3d-pkl uses {CHECKPOINTS_DEFAULT_3D} (four snapshots).",
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
        "and <stem>_<S>_metrics.csv. Default on NRP: "
        "/files/repo/uncertainty-quantification/assets/images/unigrad-io/sweep_io.png. Example with "
        "'--save-path ./out/sweep_io.png' and subject 'IXI002_T1_slice_111.npy' -> "
        "sweep_io_IXI002_T1_slice_111_images.png (and matching _curves / _metrics).",
    )
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if args.ixi_root is None:
        args.ixi_root = default_ixi_root(args.mode)
    if args.save_path is None:
        args.save_path = default_save_path()
    ck_spec = args.checkpoints
    if ck_spec is None:
        ck_spec = (
            CHECKPOINTS_DEFAULT_3D
            if args.mode == "3d-pkl"
            else CHECKPOINTS_DEFAULT_2D
        )
    checkpoints = parse_checkpoints(ck_spec)
    subject_indices: list[int] | None = None
    if args.subject_indices is not None:
        subject_indices = sorted({int(x.strip()) for x in args.subject_indices.split(",") if x.strip()})
    subj = args.subject_path.resolve() if args.subject_path else None

    if args.mode == "3d-pkl":
        atlas_pkl = (
            args.atlas_pkl.resolve()
            if args.atlas_pkl is not None
            else (args.ixi_root / "atlas.pkl").resolve()
        )
        run_sweep_volume_pkl(
            args.ixi_root.resolve(),
            atlas_pkl=atlas_pkl,
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
    else:
        run_sweep(
            args.ixi_root.resolve(),
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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
