#!/usr/bin/env python3
"""
Quick QC checks for HCP synth NPZ outputs (e.g., 13x3 dry-run = 39 files).

Checks:
- expected samples per (deformation_class, magnitude_range)
- same subject reused across replicates per combination
- source/moving consistency stats (inside overlap mask)
- ventricle-proxy drift in central brain region

Example:
python experiments/unigrad-synth/check_synth_volume_qc.py --data-dir datasets/hcp_synth_dryrun --expected-replicates 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_scalar(npz: np.lib.npyio.NpzFile, key: str) -> str:
    v = np.asarray(npz[key]).reshape(-1)[0]
    return str(v)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ventricle_proxy(volume: np.ndarray, mask: np.ndarray) -> float:
    """
    Central dark-structure proxy:
    mean intensity in central box restricted to mask.
    """
    x, y, z = volume.shape
    xs = slice(int(x * 0.35), int(x * 0.65))
    ys = slice(int(y * 0.35), int(y * 0.65))
    zs = slice(int(z * 0.40), int(z * 0.60))
    box = volume[xs, ys, zs]
    box_mask = mask[xs, ys, zs]
    vals = box[box_mask]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def run_qc(data_dir: Path, expected_replicates: int) -> int:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    per_file_rows: list[dict] = []

    for fp in files:
        with np.load(fp) as z:
            cls = _load_scalar(z, "deformation_class")
            mag_range = _load_scalar(z, "magnitude_range")
            sid = _load_scalar(z, "subject_id")
            src = np.asarray(z["source"], dtype=np.float32)
            mov = np.asarray(z["moving"], dtype=np.float32)
            src_mask = np.asarray(z["source_mask"], dtype=bool)
            mov_mask = np.asarray(z["moving_mask"], dtype=bool)

            overlap = src_mask & mov_mask
            src_vals = src[overlap]
            mov_vals = mov[overlap]

            mae = float(np.mean(np.abs(src_vals - mov_vals))) if src_vals.size else float("nan")
            corr = _corr(src_vals, mov_vals)
            v_src = _ventricle_proxy(src, src_mask)
            v_mov = _ventricle_proxy(mov, mov_mask)

            groups[(cls, mag_range)].append(fp)
            per_file_rows.append(
                {
                    "file": fp.name,
                    "subject_id": sid,
                    "class": cls,
                    "range": mag_range,
                    "overlap_frac": float(np.mean(overlap)),
                    "mae_overlap": mae,
                    "corr_overlap": corr,
                    "ventricle_proxy_src": v_src,
                    "ventricle_proxy_mov": v_mov,
                    "ventricle_proxy_delta": abs(v_src - v_mov)
                    if np.isfinite(v_src) and np.isfinite(v_mov)
                    else float("nan"),
                }
            )

    print(f"Scanned files: {len(files)}")
    print("Per-combination counts / subject consistency:")
    bad = 0
    for key in sorted(groups.keys()):
        cls, mag_range = key
        fps = groups[key]
        sids = set()
        for fp in fps:
            with np.load(fp) as z:
                sids.add(_load_scalar(z, "subject_id"))
        ok_n = len(fps) == expected_replicates
        ok_sid = len(sids) == 1
        status = "OK" if (ok_n and ok_sid) else "WARN"
        print(
            f"- {cls}_{mag_range}: n={len(fps)} (expected {expected_replicates}), "
            f"subjects={len(sids)} [{status}]"
        )
        if not (ok_n and ok_sid):
            bad += 1

    arr_mae = np.asarray([r["mae_overlap"] for r in per_file_rows], dtype=np.float64)
    arr_corr = np.asarray([r["corr_overlap"] for r in per_file_rows], dtype=np.float64)
    arr_vd = np.asarray([r["ventricle_proxy_delta"] for r in per_file_rows], dtype=np.float64)

    print("\nGlobal stats (all files):")
    print(f"- overlap MAE p50/p95: {np.nanpercentile(arr_mae, 50):.4f} / {np.nanpercentile(arr_mae, 95):.4f}")
    print(f"- overlap corr p50/p05: {np.nanpercentile(arr_corr, 50):.4f} / {np.nanpercentile(arr_corr, 5):.4f}")
    print(
        f"- ventricle proxy |delta| p50/p95: "
        f"{np.nanpercentile(arr_vd, 50):.4f} / {np.nanpercentile(arr_vd, 95):.4f}"
    )

    worst = sorted(
        per_file_rows,
        key=lambda r: (np.nan_to_num(r["ventricle_proxy_delta"], nan=-1.0)),
        reverse=True,
    )[:5]
    print("\nTop 5 ventricle-proxy drifts:")
    for r in worst:
        print(
            f"- {r['file']}: {r['class']}_{r['range']} "
            f"delta={r['ventricle_proxy_delta']:.4f} corr={r['corr_overlap']:.4f}"
        )

    return bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC checks for synth volume NPZ outputs.")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing *.npz.")
    p.add_argument(
        "--expected-replicates",
        type=int,
        default=3,
        help="Expected files per class/range combination.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    issues = run_qc(args.data_dir, args.expected_replicates)
    if issues:
        raise SystemExit(1)
