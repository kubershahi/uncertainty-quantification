#!/usr/bin/env python3
"""
Plot UniGradICON *_fiver.npz samples (fixed, warped, ‖φ_true‖, ‖φ_pred‖, error map).

Default: pick three files with **min / median / max** of a ranking scalar (default: **mean
error_map**). Use ``--selection random`` for random files.

Examples:
  python visualize_unigrad_data.py --split Train --save-path ./assets/images/ixi_unigrad_train_minmedmax.png --no-show --selection random --num-samples 3
  python visualize_unigrad_data.py --split Train --save-path ./assets/images/ixi_unigrad_train_minmedmax.png --no-show  --rank-by max_error
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_KEYS = frozenset(
    {"image", "warped", "phi_true", "phi_pred", "phi_diff", "error_map"}
)
SPLITS = ("Train", "Val", "Test", "Atlas")
FIVER_GLOB = "*_fiver.npz"


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def _unpack_qc_passed(raw) -> tuple[bool | None, str | None]:
    a = np.asarray(raw)
    if a.size != 1:
        return None, f"qc_passed must be a single value, got shape {a.shape}"
    v = a.reshape(-1)[0]
    try:
        return bool(v), None
    except (ValueError, TypeError) as e:
        return None, f"qc_passed not bool-convertible: {e}"


def collect_fivers(input_dir: Path, split: str, pattern: str) -> list[Path]:
    split_dir = input_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return sorted(split_dir.glob(pattern))


def load_fiver(npz_path: Path) -> dict:
    """Load fiver arrays; optional valid_mask / qc_passed in ``extra``."""
    extra: dict = {}
    with np.load(npz_path) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise KeyError(f"{npz_path.name} missing {sorted(missing)}")
        out = {
            "image": np.asarray(data["image"]),
            "warped": np.asarray(data["warped"]),
            "phi_true": np.asarray(data["phi_true"]),
            "phi_pred": np.asarray(data["phi_pred"]),
            "phi_diff": np.asarray(data["phi_diff"]),
            "error_map": np.asarray(data["error_map"]),
        }
        if "valid_mask" in data.files:
            extra["valid_mask"] = np.asarray(data["valid_mask"])
        if "qc_passed" in data.files:
            qc_val, qc_err = _unpack_qc_passed(data["qc_passed"])
            if qc_err:
                raise ValueError(f"{npz_path.name}: {qc_err}")
            extra["qc_passed"] = qc_val
    return {**out, **{"_extra": extra}}


def _rank_scalar(blob: dict, rank_by: str) -> float:
    err = blob["error_map"].astype(np.float64)
    pt = blob["phi_true"].astype(np.float64)
    pp = blob["phi_pred"].astype(np.float64)
    if rank_by == "mean_error":
        return float(np.mean(err))
    if rank_by == "max_error":
        return float(np.max(err))
    if rank_by == "mean_phi_true":
        return float(np.mean(phi_magnitude(pt)))
    if rank_by == "mean_phi_pred":
        return float(np.mean(phi_magnitude(pp)))
    if rank_by == "mean_phi_diff":
        return float(np.mean(phi_magnitude(blob["phi_diff"].astype(np.float64))))
    raise ValueError(f"unknown rank_by: {rank_by!r}")


def select_min_median_max(
    files: list[Path],
    rank_by: str,
) -> list[tuple[Path, str, float]]:
    if not files:
        return []
    scored: list[tuple[Path, float]] = []
    for fp in files:
        d = load_fiver(fp)
        d.pop("_extra", None)
        scored.append((fp, _rank_scalar(d, rank_by)))
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 1:
        return [(scored[0][0], "min", scored[0][1])]
    if n == 2:
        return [
            (scored[0][0], "min", scored[0][1]),
            (scored[1][0], "max", scored[1][1]),
        ]
    i_min, i_med, i_max = 0, n // 2, n - 1
    return [
        (scored[i_min][0], "min", scored[i_min][1]),
        (scored[i_med][0], "median", scored[i_med][1]),
        (scored[i_max][0], "max", scored[i_max][1]),
    ]


def visualize_fivers(
    input_dir: Path,
    split: str,
    pattern: str,
    save_path: Path | None,
    no_show: bool,
    *,
    selection: str = "min_median_max",
    rank_by: str = "mean_error",
    num_samples: int = 3,
    seed: int = 42,
    err_vmax: float | None = None,
    err_percentile: float = 99.0,
    phi_vmax: float | None = None,
    phi_percentile: float = 99.0,
) -> None:
    files = collect_fivers(input_dir, split, pattern)
    if not files:
        raise FileNotFoundError(
            f"No files in '{input_dir / split}' matching '{pattern}'."
        )

    if selection == "random":
        rng = random.Random(seed)
        n = min(num_samples, len(files))
        picked: list[tuple[Path, str, float]] = []
        for p in rng.sample(files, n):
            d = load_fiver(p)
            d.pop("_extra", None)
            picked.append((p, "", float("nan")))
        title_suffix = f"random {n} of {len(files)} (seed={seed})"
    elif selection == "min_median_max":
        picked = select_min_median_max(files, rank_by)
        title_suffix = f"min/median/max by {rank_by} ({len(picked)} of {len(files)} files)"
    else:
        raise ValueError(f"Unknown selection: {selection!r}")

    nrows = len(picked)
    # Shared color limits from picked samples only
    err_stack: list[np.ndarray] = []
    pt_stack: list[np.ndarray] = []
    pp_stack: list[np.ndarray] = []
    for fp, _, _ in picked:
        d = load_fiver(fp)
        d.pop("_extra", None)
        err_stack.append(d["error_map"].ravel())
        pt_stack.append(phi_magnitude(d["phi_true"]).ravel())
        pp_stack.append(phi_magnitude(d["phi_pred"]).ravel())
    err_v = err_vmax
    if err_v is None:
        err_v = float(np.percentile(np.concatenate(err_stack), err_percentile))
        if err_v <= 0:
            err_v = 1e-6
    phi_v = phi_vmax
    if phi_v is None:
        pt_p = float(np.percentile(np.concatenate(pt_stack), phi_percentile))
        pp_p = float(np.percentile(np.concatenate(pp_stack), phi_percentile))
        phi_v = max(pt_p, pp_p, 1e-6)

    fig, axes = plt.subplots(nrows, 5, figsize=(18, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for row, (file_path, rank_label, score) in enumerate(picked):
        d = load_fiver(file_path)
        extra = d.pop("_extra", {})
        image = d["image"]
        warped = d["warped"]
        phi_true = d["phi_true"]
        phi_pred = d["phi_pred"]
        error_map = d["error_map"]
        mag_t = phi_magnitude(phi_true.astype(np.float64))
        mag_p = phi_magnitude(phi_pred.astype(np.float64))

        qc_note = ""
        if "qc_passed" in extra:
            qc_note = f" qc_passed={extra['qc_passed']}"
        rank_note = ""
        if selection == "min_median_max" and rank_label and np.isfinite(score):
            rank_note = f" [{rank_label} {rank_by}={score:.4f}]"

        stem = file_path.stem
        stem_title = stem if len(stem) <= 44 else f"{stem[:41]}…"

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(
            f"Fixed: {stem_title}{qc_note}{rank_note}", fontsize=8
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(warped, cmap="gray")
        axes[row, 1].set_title("Warped", fontsize=9)
        axes[row, 1].axis("off")

        im_pt = axes[row, 2].imshow(mag_t, cmap="hot", vmin=0.0, vmax=phi_v)
        axes[row, 2].set_title("‖φ_true‖", fontsize=9)
        axes[row, 2].axis("off")
        fig.colorbar(im_pt, ax=axes[row, 2], fraction=0.046, pad=0.02)

        im_pp = axes[row, 3].imshow(mag_p, cmap="hot", vmin=0.0, vmax=phi_v)
        axes[row, 3].set_title("‖φ_pred‖", fontsize=9)
        axes[row, 3].axis("off")
        fig.colorbar(im_pp, ax=axes[row, 3], fraction=0.046, pad=0.02)

        im_e = axes[row, 4].imshow(error_map, cmap="hot", vmin=0.0, vmax=err_v)
        axes[row, 4].set_title("‖φ_true − φ_pred‖ (px)", fontsize=9)
        axes[row, 4].axis("off")
        fig.colorbar(im_e, ax=axes[row, 4], fraction=0.046, pad=0.02)

        axes[row, 0].set_ylabel(file_path.stem[:28], fontsize=7, rotation=90)

    fig.suptitle(f"UniGrad fivers — {split} ({title_suffix})", fontsize=12)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize UniGradICON *_fiver.npz (min/median/max error or random).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        "--input-dir",
        type=Path,
        default=Path("./data/IXI_2D_unigrad_fiver"),
        dest="data_dir",
        help="Root with Train/Val/Test/Atlas subfolders.",
    )
    p.add_argument("--split", type=str, default="Train", choices=list(SPLITS))
    p.add_argument(
        "--selection",
        type=str,
        default="min_median_max",
        choices=["min_median_max", "random"],
    )
    p.add_argument(
        "--rank-by",
        type=str,
        default="mean_error",
        choices=[
            "mean_error",
            "max_error",
            "mean_phi_true",
            "mean_phi_pred",
            "mean_phi_diff",
        ],
        help="Scalar per file for min/median/max selection only (ignored when --selection random).",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--err-vmax", type=float, default=None)
    p.add_argument("--err-percentile", type=float, default=99.0)
    p.add_argument("--phi-vmax", type=float, default=None)
    p.add_argument("--phi-percentile", type=float, default=99.0)
    p.add_argument("--save-path", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2
    visualize_fivers(
        input_dir=args.data_dir,
        split=args.split,
        pattern=FIVER_GLOB,
        save_path=args.save_path,
        no_show=args.no_show,
        selection=args.selection,
        rank_by=args.rank_by,
        num_samples=args.num_samples,
        seed=args.seed,
        err_vmax=args.err_vmax,
        err_percentile=args.err_percentile,
        phi_vmax=args.phi_vmax,
        phi_percentile=args.phi_percentile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
