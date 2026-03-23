#!/usr/bin/env python3
"""
Refresh existing synthetic triplets without rebuilding the whole dataset.

  1) Scan each *_triplet.npz (per split).
  2) QC using the same rules as create_synth_data.passes_checks (global max|phi| on full slice; interior on valid_mask).
  3) Near-identity: max|phi| on **full slice** < --near-zero-eps (same as check_synth_data).
  4) Regenerate from raw *.npy when:
       - QC fails, or
       - near-identity AND split already has more near-identity samples than the keep-quota
         (default: keep at least 10% of split count as near-identity; regenerate the excess).

Uses create_synth_data.py for QC limits and transform *geometry* (scales, degrees,
displacements). Regeneration uses higher RandomAffine / RandomElastic ``p`` so both
almost always apply—avoids re-drawing near-identity φ when replacing excess near-zero samples.

Examples:
  python modify_synth_data.py --dry-run
  python modify_synth_data.py --workers 64
  python modify_synth_data.py --near-zero-keep-frac 0.10 --near-zero-eps 1e-4
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_DATAHUB = Path(__file__).resolve().parent
if str(_DATAHUB) not in sys.path:
    sys.path.insert(0, str(_DATAHUB))

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

# Single source of truth for transforms + QC (same directory as this script)
import create_synth_data as csd

SPLITS = ("Train", "Val", "Test", "Atlas")
TRIPLET_SUFFIX = "_triplet.npz"

# Regen: same transform params as create_synth_data, but force both stages to run
# (create_synth_data uses AFFINE_P / ELASTIC_P < 1 → non-trivial chance of identity φ).
REGEN_AFFINE_P = 0.9
REGEN_ELASTIC_P = 0.9
# More QC draws than create_synth_data.MAX_TRANSFORM_ATTEMPTS (regen-only).
REGEN_MAX_TRANSFORM_ATTEMPTS = 40


def build_regen_transform() -> tio.Compose:
    """
    Same RandomAffine + RandomElastic *geometry* as create_synth_data, but with
    ``REGEN_*_P`` apply probabilities.

    Built here (not ``csd.build_transform(affine_p=...)``) so regen still works if
    an older ``create_synth_data.py`` without optional ``p`` overrides is on PYTHONPATH.
    """
    return tio.Compose(
        [
            tio.RandomAffine(
                scales=csd.AFFINE_SCALES,
                degrees=csd.AFFINE_DEGREES,
                translation=(csd.AFFINE_TRANSLATION_PX, csd.AFFINE_TRANSLATION_PX, 0),
                default_pad_value="minimum",
                p=REGEN_AFFINE_P,
            ),
            tio.RandomElasticDeformation(
                num_control_points=csd.ELASTIC_NUM_CONTROL_POINTS,
                max_displacement=(csd.ELASTIC_MAX_DISP_MM, csd.ELASTIC_MAX_DISP_MM, 0),
                locked_borders=2,
                p=REGEN_ELASTIC_P,
            ),
        ]
    )


def npz_stem_to_npy_name(npz_name: str) -> str | None:
    if not npz_name.endswith(TRIPLET_SUFFIX):
        return None
    return npz_name[: -len(TRIPLET_SUFFIX)] + ".npy"


def is_near_zero_phi(phi: np.ndarray, eps: float) -> bool:
    """True if max |φ| on the **full slice** is below eps (same rule as check_synth_data)."""
    mag = csd.phi_magnitude(phi.astype(np.float64))
    return float(np.max(mag)) < eps


def analyze_triplet(
    npz_path: Path,
    near_zero_eps: float,
    max_phi_interior: float | None,
    max_phi_global: float | None,
    min_warped_ratio: float | None,
) -> tuple[bool, bool, str | None]:
    """
    Returns (qc_ok, near_zero, error_message).
    """
    try:
        with np.load(npz_path) as z:
            if not {"image", "warped", "phi"}.issubset(z.files):
                return False, False, "missing keys"
            image = np.asarray(z["image"])
            warped = np.asarray(z["warped"])
            phi = np.asarray(z["phi"])
        qc_ok = csd.passes_checks(
            phi,
            image,
            warped,
            interior_margin=csd.INTERIOR_MARGIN,
            max_phi_interior_px=max_phi_interior,
            max_phi_global_px=max_phi_global,
            min_warped_mean_ratio=min_warped_ratio,
        )
        nz = is_near_zero_phi(phi, near_zero_eps)
        return qc_ok, nz, None
    except Exception as e:
        return False, False, str(e)


def _default_parallel_workers() -> int:
    """Use all logical CPUs the OS reports (matches typical HPC pool allocation)."""
    return max(1, os.cpu_count() or 4)


def _worker_analyze_split(
    args: tuple[str, str, float, float | None, float | None, float | None],
) -> tuple[str, str, bool, bool, str | None]:
    """Picklable: (split, npz_path, near_zero_eps, max_phi_interior, max_phi_global, min_warped_ratio)."""
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_k] = "1"

    split, path_s, near_zero_eps, max_phi_interior, max_phi_global, min_warped_ratio = args
    ok, nz, err = analyze_triplet(
        Path(path_s),
        near_zero_eps,
        max_phi_interior,
        max_phi_global,
        min_warped_ratio,
    )
    return split, path_s, ok, nz, err


def regenerate_one_triplet(
    raw_npy_path: Path,
    out_npz_path: Path,
    rng: random.Random,
) -> tuple[bool, str | None]:
    """
    Returns (qc_passed, error_or_none).
    """
    img_2d = np.load(raw_npy_path)
    w, h = img_2d.shape
    grid_x, grid_y = torch.meshgrid(
        torch.arange(w, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        indexing="ij",
    )
    identity_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(-1).float()

    transform = build_regen_transform()
    warped_img: np.ndarray | None = None
    phi_true: np.ndarray | None = None
    qc_passed = False

    for attempt in range(REGEN_MAX_TRANSFORM_ATTEMPTS):
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        img_tensor = torch.from_numpy(img_2d).unsqueeze(0).unsqueeze(-1).float()
        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=img_tensor),
            grid=tio.ScalarImage(tensor=identity_grid.clone()),
        )
        transformed = transform(subject)
        cand_warped = transformed.mri.data.squeeze().numpy()
        cand_phi = (
            transformed.grid.data.squeeze().numpy() - identity_grid.squeeze().numpy()
        ).astype(np.float32)
        warped_img, phi_true = cand_warped, cand_phi
        if csd.passes_checks(
            cand_phi,
            img_2d,
            cand_warped,
            interior_margin=csd.INTERIOR_MARGIN,
            max_phi_interior_px=csd.MAX_PHI_INTERIOR_PX,
            max_phi_global_px=csd.MAX_PHI_GLOBAL_PX,
            min_warped_mean_ratio=csd.MIN_WARPED_MEAN_RATIO,
        ):
            qc_passed = True
            break

    if warped_img is None or phi_true is None:
        return False, "no sample drawn"

    valid_mask = csd.interior_valid_mask(w, h, csd.INTERIOR_MARGIN)
    np.savez_compressed(
        out_npz_path,
        image=img_2d,
        warped=warped_img,
        phi=phi_true,
        valid_mask=valid_mask,
        qc_passed=qc_passed,
    )
    return qc_passed, None


def _worker_regenerate(args: tuple[str, str, int]) -> tuple[str, bool, str | None]:
    """Picklable worker: (raw_npy, out_npz, seed)."""
    # Each process would otherwise default to using all CPU cores (OpenMP/MKL),
    # so N workers × N threads oversubscribes and makes regeneration much slower.
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_k] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    raw_s, out_s, seed = args
    rng = random.Random(seed)
    ok, err = regenerate_one_triplet(Path(raw_s), Path(out_s), rng)
    return out_s, ok, err


def plan_split(
    split: str,
    triplet_root: Path,
    raw_root: Path,
    near_zero_eps: float,
    near_zero_keep_frac: float,
    seed: int,
    *,
    analyses: dict[str, tuple[bool, bool, str | None]] | None = None,
) -> tuple[list[Path], list[Path], dict]:
    """
    Returns (paths_to_regenerate, paths_kept_unchanged, stats).

    If ``analyses`` is provided, keys must be ``str(path.resolve())`` for each
    ``*_triplet.npz`` in this split; values are ``(qc_ok, near_zero, err)`` as
    from ``analyze_triplet``. Otherwise each file is analyzed in-process.
    """
    split_dir = triplet_root / split
    if not split_dir.is_dir():
        return [], [], {"skipped": True}

    npz_files = sorted(split_dir.glob(f"*{TRIPLET_SUFFIX}"))
    n = len(npz_files)
    if n == 0:
        return [], [], {"empty": True}

    k_keep = max(1, math.ceil(near_zero_keep_frac * n))

    qc_ok_near: list[Path] = []
    qc_ok_far: list[Path] = []
    qc_fail: list[Path] = []
    analyze_errors: list[tuple[Path, str]] = []

    for p in npz_files:
        if analyses is not None:
            ok, nz, err = analyses[str(p.resolve())]
        else:
            ok, nz, err = analyze_triplet(
                p,
                near_zero_eps,
                csd.MAX_PHI_INTERIOR_PX,
                csd.MAX_PHI_GLOBAL_PX,
                csd.MIN_WARPED_MEAN_RATIO,
            )
        if err:
            analyze_errors.append((p, err))
            qc_fail.append(p)
            continue
        if not ok:
            qc_fail.append(p)
        elif nz:
            qc_ok_near.append(p)
        else:
            qc_ok_far.append(p)

    rng = random.Random(seed + hash(split) % (2**31))
    qc_ok_near_shuffled = qc_ok_near[:]
    rng.shuffle(qc_ok_near_shuffled)

    # Preserve up to k_keep near-identity samples that already pass QC.
    preserved = set(qc_ok_near_shuffled[: min(len(qc_ok_near_shuffled), k_keep)])
    regen_near = [p for p in qc_ok_near if p not in preserved]

    to_regen = sorted(set(qc_fail) | set(regen_near))
    unchanged = sorted(
        set(npz_files) - set(to_regen),
        key=lambda x: x.name,
    )

    stats = {
        "n": n,
        "k_keep_near": k_keep,
        "qc_fail": len(qc_fail),
        "near_ok": len(qc_ok_near),
        "far_ok": len(qc_ok_far),
        "preserved_near": len(preserved),
        "regen_near": len(regen_near),
        "regen_total": len(to_regen),
        "analyze_errors": len(analyze_errors),
    }
    return to_regen, unchanged, stats


def resolve_raw_path(split: str, npz_path: Path, raw_root: Path) -> Path | None:
    name = npz_path.name
    npy = npz_stem_to_npy_name(name)
    if npy is None:
        return None
    return raw_root / split / npy


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-QC existing triplets; regenerate only failures and excess near-identity.",
    )
    p.add_argument(
        "--triplet-root",
        type=Path,
        default=Path("./data/IXI_2D_synth_trip"),
        help="Root containing Train/Val/Test/Atlas/*_triplet.npz",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("./data/IXI_2D"),
        help="Raw IXI_2D root with same splits + *.npy sources.",
    )
    p.add_argument(
        "--near-zero-eps",
        type=float,
        default=1e-4,
        help="max|phi| on full slice below this counts as near-identity (same as check_synth_data).",
    )
    p.add_argument(
        "--near-zero-keep-frac",
        type=float,
        default=0.10,
        help="Per split, keep at least this fraction of samples as near-identity (no regen).",
    )
    p.add_argument(
        "--workers",
        "--worker",
        type=int,
        metavar="N",
        default=None,
        help="Parallel processes for analyze + regenerate (default: all logical CPUs). "
        "Each regen worker runs TorchIO with 1 OpenMP thread to avoid oversubscription.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not write npz files.",
    )
    args = p.parse_args()
    if args.workers is None:
        args.workers = _default_parallel_workers()

    triplet_root = args.triplet_root.resolve()
    raw_root = args.raw_root.resolve()

    if not triplet_root.is_dir():
        print(f"ERROR: triplet root not found: {triplet_root}", file=sys.stderr)
        return 2

    all_regen: list[tuple[str, Path, Path]] = []
    summary: dict[str, dict] = {}

    print(f"Triplet root: {triplet_root}")
    print(f"Raw root:     {raw_root}")
    print(
        f"QC + transform geometry from create_synth_data.py "
        f"(INTERIOR_MARGIN={csd.INTERIOR_MARGIN}, "
        f"MAX_PHI_INTERIOR={csd.MAX_PHI_INTERIOR_PX}, "
        f"MAX_PHI_GLOBAL={csd.MAX_PHI_GLOBAL_PX})"
    )
    print(
        f"Regen transform apply probability (override): "
        f"affine p={REGEN_AFFINE_P}, elastic p={REGEN_ELASTIC_P} "
        f"(create_synth_data uses {csd.AFFINE_P}, {csd.ELASTIC_P})"
    )
    print(
        f"Regen QC attempts per file: {REGEN_MAX_TRANSFORM_ATTEMPTS} "
        f"(create_synth_data MAX_TRANSFORM_ATTEMPTS={csd.MAX_TRANSFORM_ATTEMPTS})"
    )
    print(
        f"Near-identity: eps={args.near_zero_eps}, "
        f"keep ≥{args.near_zero_keep_frac:.0%} of split as near-identity without regen"
    )
    print(f"Parallel workers: {args.workers} (shared pool: analyze, then regenerate)")
    print("-" * 60)

    analyze_tasks: list[tuple[str, str, float, float | None, float | None, float | None]] = []
    split_npz_paths: dict[str, list[Path] | None] = {}
    for split in SPLITS:
        split_dir = triplet_root / split
        if not split_dir.is_dir():
            split_npz_paths[split] = None
            continue
        paths = sorted(split_dir.glob(f"*{TRIPLET_SUFFIX}"))
        split_npz_paths[split] = paths
        for p in paths:
            analyze_tasks.append(
                (
                    split,
                    str(p.resolve()),
                    args.near_zero_eps,
                    csd.MAX_PHI_INTERIOR_PX,
                    csd.MAX_PHI_GLOBAL_PX,
                    csd.MIN_WARPED_MEAN_RATIO,
                )
            )

    analysis_by_split: dict[str, dict[str, tuple[bool, bool, str | None]]] = defaultdict(
        dict
    )
    if analyze_tasks:
        n_an = min(args.workers, len(analyze_tasks))
        ch = max(1, len(analyze_tasks) // (4 * n_an))
        with ProcessPoolExecutor(max_workers=n_an) as ex:
            for row in tqdm(
                ex.map(_worker_analyze_split, analyze_tasks, chunksize=ch),
                total=len(analyze_tasks),
                desc="Analyze",
            ):
                sp, ps, ok, nz, err = row
                analysis_by_split[sp][ps] = (ok, nz, err)

    for split in SPLITS:
        paths = split_npz_paths.get(split)
        split_analyses = analysis_by_split[split] if paths else None
        to_regen, _unchanged, stats = plan_split(
            split,
            triplet_root,
            raw_root,
            args.near_zero_eps,
            args.near_zero_keep_frac,
            args.seed,
            analyses=split_analyses,
        )
        summary[split] = stats
        if stats.get("skipped"):
            print(f"{split}: (no triplet folder)")
            continue
        if stats.get("empty"):
            print(f"{split}: 0 triplets")
            continue

        print(
            f"{split}: n={stats['n']}  regen={stats['regen_total']}  "
            f"(qc_fail={stats['qc_fail']}, excess_near={stats['regen_near']}, "
            f"preserved_near={stats['preserved_near']}/{stats['k_keep_near']} target)  "
            f"unchanged={stats['n'] - stats['regen_total']}"
        )
        if stats.get("analyze_errors"):
            print(f"      analyze_errors={stats['analyze_errors']}")

        for npz_path in to_regen:
            raw = resolve_raw_path(split, npz_path, raw_root)
            if raw is None or not raw.is_file():
                print(f"  SKIP missing raw: {npz_path.name} -> {raw}", file=sys.stderr)
                continue
            all_regen.append((split, npz_path, raw))

    print("-" * 60)
    print(f"Total to regenerate: {len(all_regen)}")
    if args.dry_run:
        print("Dry run — exiting without writing.")
        return 0

    if not all_regen:
        print("Nothing to regenerate.")
        return 0

    tasks: list[tuple[str, str, int]] = []
    for i, (_split, npz_path, raw_path) in enumerate(all_regen):
        tasks.append(
            (
                str(raw_path),
                str(npz_path),
                (args.seed + i * 100003 + hash(str(npz_path)) % 999983) % (2**31),
            )
        )

    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_worker_regenerate, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Regenerate"):
            out_s, ok, err = fut.result()
            if not ok:
                failed += 1
                if err:
                    tqdm.write(f"WARN {out_s} regenerate failed: {err}")
                else:
                    tqdm.write(
                        f"WARN {out_s}: saved with qc_passed=False "
                        f"(no draw passed QC within {REGEN_MAX_TRANSFORM_ATTEMPTS} attempts)"
                    )

    # Refresh manifest of qc_flagged
    flagged: list[str] = []
    for split in SPLITS:
        sd = triplet_root / split
        if not sd.is_dir():
            continue
        for fp in sorted(sd.glob(f"*{TRIPLET_SUFFIX}")):
            try:
                with np.load(fp) as z:
                    if "qc_passed" not in z.files:
                        continue
                    if not bool(np.asarray(z["qc_passed"]).item()):
                        flagged.append(f"{split}/{fp.name}")
            except OSError:
                pass

    man = triplet_root / "qc_flagged_paths.txt"
    if flagged:
        with open(man, "w", encoding="utf-8") as f:
            f.write("# Triplets with qc_passed=False — delete or regenerate\n")
            for line in sorted(flagged):
                f.write(f"{line}\n")
        print(f"Wrote {man} ({len(flagged)} path(s))")
    else:
        if man.is_file():
            man.unlink()
        print("No qc_passed=False triplets; removed qc_flagged_paths.txt if present.")

    print(f"Done. Regenerated {len(all_regen) - failed} ok, {failed} with qc_passed=False.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
