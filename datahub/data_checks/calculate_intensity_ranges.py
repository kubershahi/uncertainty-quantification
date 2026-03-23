#!/usr/bin/env python3
"""
Print per-pixel min / max of ``image`` and ``warped`` on random files per split.

Datasets (default roots):
  - IXI_2D:        ``*.npy`` — one array per file (reported as ``image`` only; no warped).
  - synth trip:    ``*_triplet.npz`` with ``image``, ``warped``.
  - unigrad fiver: ``*_fiver.npz`` with ``image``, ``warped``.

Examples:
  python calculate_intensity_ranges.py --split Train --num-samples 5
  python calculate_intensity_ranges.py --split Val --num-samples 10 --seed 0 \\
    --ixi2d-dir ./data/IXI_2D --synth-dir ./data/IXI_2D_synth_trip \\
    --unigrad-dir ./data/IXI_2D_unigrad_fiver
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

SPLITS = ("Train", "Val", "Test", "Atlas")


def _pick_random(files: list[Path], n: int, rng: random.Random) -> list[Path]:
    if not files:
        return []
    k = min(n, len(files))
    return rng.sample(files, k)


def collect_ixi2d_npy(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.npy"))


def collect_synth_triplets(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*_triplet.npz"))


def collect_unigrad_fivers(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*_fiver.npz"))


def stats_min_max(arr: np.ndarray) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float64)
    return float(np.min(a)), float(np.max(a))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Min/max of image & warped on random samples per dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--split", type=str, default="Train", choices=list(SPLITS))
    p.add_argument("--num-samples", type=int, default=5, metavar="N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ixi2d-dir",
        type=Path,
        default=Path("./data/IXI_2D"),
        help="Root with Train/Val/... subfolders of *.npy slices.",
    )
    p.add_argument(
        "--synth-dir",
        type=Path,
        default=Path("./data/IXI_2D_synth_trip"),
        help="Root with *_triplet.npz per split.",
    )
    p.add_argument(
        "--unigrad-dir",
        type=Path,
        default=Path("./data/IXI_2D_unigrad_fiver"),
        help="Root with *_fiver.npz per split.",
    )
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    split = args.split
    n = max(0, args.num_samples)

    def block_title(name: str) -> None:
        print()
        print(f"=== {name} ({split}) — up to {n} random file(s) ===")

    # --- IXI_2D: .npy only (fixed slice; no warped in this format) ---
    block_title("IXI_2D (.npy)")
    ixi_dir = args.ixi2d_dir / split
    ixi_files = collect_ixi2d_npy(ixi_dir)
    if not ixi_files:
        print(f"  (no files or missing dir: {ixi_dir})")
    else:
        for fp in _pick_random(ixi_files, n, rng):
            img = np.load(fp)
            imin, imax = stats_min_max(img)
            print(
                f"  {fp.name}\n"
                f"    image:  min={imin:.8g}  max={imax:.8g}\n"
                f"    warped: (not stored — raw 2D is fixed image only)"
            )

    # --- Synth triplets ---
    block_title("IXI_2D_synth_trip (*_triplet.npz)")
    syn_dir = args.synth_dir / split
    syn_files = collect_synth_triplets(syn_dir)
    if not syn_files:
        print(f"  (no files or missing dir: {syn_dir})")
    else:
        for fp in _pick_random(syn_files, n, rng):
            with np.load(fp) as z:
                image = np.asarray(z["image"])
                warped = np.asarray(z["warped"])
            imin, imax = stats_min_max(image)
            wmin, wmax = stats_min_max(warped)
            print(
                f"  {fp.name}\n"
                f"    image:  min={imin:.8g}  max={imax:.8g}\n"
                f"    warped: min={wmin:.8g}  max={wmax:.8g}"
            )

    # --- UniGrad fivers ---
    block_title("IXI_2D_unigrad_fiver (*_fiver.npz)")
    uni_dir = args.unigrad_dir / split
    uni_files = collect_unigrad_fivers(uni_dir)
    if not uni_files:
        print(f"  (no files or missing dir: {uni_dir})")
    else:
        for fp in _pick_random(uni_files, n, rng):
            with np.load(fp) as z:
                image = np.asarray(z["image"])
                warped = np.asarray(z["warped"])
            imin, imax = stats_min_max(image)
            wmin, wmax = stats_min_max(warped)
            print(
                f"  {fp.name}\n"
                f"    image:  min={imin:.8g}  max={imax:.8g}\n"
                f"    warped: min={wmin:.8g}  max={wmax:.8g}"
            )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
