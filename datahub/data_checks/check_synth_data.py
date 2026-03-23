#!/usr/bin/env python3
"""
Post-generation review for IXI_2D synthetic triplets (*_triplet.npz).

  1) Integrity: load each archive, required keys, shapes, finiteness.
  2) Summary: per-split counts, qc_passed breakdown, ‖φ‖ stats (see nomenclature below).

  Per split: (1) near-identity φ (full slice: max ‖φ‖ < ε), (2) mean of per-sample-φ-means on full
  slice, (3) same on interior (valid_mask), (4) distribution of per-sample max ‖φ‖ (full slice)
  across samples. Same near-identity rule as ``modify_synth_data``.

  Plots: ``datahub/visualize_synth_data.py``

Examples:
  python data_checks/check_synth_data.py
  python data_checks/check_synth_data.py --data-dir ./data/IXI_2D_synth_trip/
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED_KEYS = frozenset({"image", "warped", "phi"})
SPLITS = ("Train", "Val", "Test", "Atlas")
# Matches names from create_synth_data.py (*_triplet.npz per slice).
TRIPLET_GLOB = "*_triplet.npz"


def phi_magnitude(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] * phi[0] + phi[1] * phi[1])


def _unpack_qc_passed(raw) -> tuple[bool | None, str | None]:
    """
    Accept 0-d array or any array with exactly one element (e.g. shape () or (1,)).
    Returns (value, None) or (None, error message).
    """
    a = np.asarray(raw)
    if a.size != 1:
        return None, f"qc_passed must be a single value, got shape {a.shape}"
    v = a.reshape(-1)[0]
    if isinstance(v, (np.floating, float)) and not np.isfinite(float(v)):
        return None, "qc_passed is non-finite"
    try:
        return bool(v), None
    except (ValueError, TypeError) as e:
        return None, f"qc_passed not bool-convertible: {e}"


def _valid_mask_to_bool(
    vm: np.ndarray, ref_shape: tuple[int, ...]
) -> tuple[np.ndarray | None, str | None]:
    """
    Accept bool mask or numeric 0/1 (int or float). Returns (H,W) bool or (None, err).
    """
    x = np.asarray(vm)
    if x.shape != ref_shape:
        return None, f"valid_mask shape {x.shape} != image {ref_shape}"
    if not np.isfinite(x).all():
        return None, "valid_mask has non-finite values"
    if x.dtype.kind == "b":
        return x.astype(bool, copy=False), None
    flat = x.ravel()
    if np.issubdtype(x.dtype, np.integer):
        if not np.logical_or(flat == 0, flat == 1).all():
            return None, "integer valid_mask must be 0 or 1"
        return x.astype(bool), None
    if np.issubdtype(x.dtype, np.floating):
        z = np.isclose(flat, 0.0)
        o = np.isclose(flat, 1.0)
        if not (z | o).all():
            return None, "float valid_mask must be ~0 or ~1"
        return o.reshape(x.shape), None
    return None, f"valid_mask unsupported dtype {x.dtype}"


def validate_triplet(path: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    try:
        with np.load(path) as data:
            keys = set(data.files)
            miss = REQUIRED_KEYS - keys
            if miss:
                return False, [f"missing keys: {sorted(miss)}"]

            image = data["image"]
            warped = data["warped"]
            phi = data["phi"]

            if image.ndim != 2:
                errs.append(f"image: expected 2D, got shape {image.shape}")
            if warped.ndim != 2:
                errs.append(f"warped: expected 2D, got shape {warped.shape}")
            if image.shape != warped.shape:
                errs.append(f"image {image.shape} vs warped {warped.shape} mismatch")
            exp_phi = (2, image.shape[0], image.shape[1])
            if phi.ndim != 3 or phi.shape != exp_phi:
                errs.append(f"phi: expected {exp_phi}, got {phi.shape}")

            if not np.isfinite(image).all():
                errs.append("image: non-finite values")
            if not np.isfinite(warped).all():
                errs.append("warped: non-finite values")
            if not np.isfinite(phi).all():
                errs.append("phi: non-finite values")

            if "valid_mask" in keys:
                _, vm_err = _valid_mask_to_bool(data["valid_mask"], image.shape)
                if vm_err:
                    errs.append(f"valid_mask: {vm_err}")

            if "qc_passed" in keys:
                _, q_err = _unpack_qc_passed(data["qc_passed"])
                if q_err:
                    errs.append(f"qc_passed: {q_err}")
    except Exception as e:
        return False, [f"load/parse: {e}"]

    return (len(errs) == 0), errs


def _mean_of_means_summary_lines(
    heading: str,
    per_sample_mean_phi: np.ndarray,
    *,
    indent: str = "         ",
) -> list[str]:
    """
    Each sample contributes one scalar: its mean(‖φ‖) over the chosen region.
    Report how those sample-level means vary across the split (compact).
    """
    v = np.asarray(per_sample_mean_phi, dtype=np.float64)
    if v.size == 0:
        return [f"{indent}{heading}", f"{indent}  (no samples)"]
    return [
        f"{indent}{heading}",
        f"{indent}  n={v.size}  mean_of_per_sample_means={v.mean():.4f} px  "
        f"min={v.min():.4f} px  max={v.max():.4f} px  std={v.std():.4f} px",
    ]


def _distribution_across_samples_lines(
    heading: str,
    per_sample_values: np.ndarray,
    *,
    indent: str = "         ",
) -> list[str]:
    """
    Full distribution of one scalar per sample (used for max(‖φ‖) per slice only).
    """
    v = np.asarray(per_sample_values, dtype=np.float64)
    if v.size == 0:
        return [f"{indent}{heading}", f"{indent}  (no samples)"]
    lines = [
        f"{indent}{heading}",
        f"{indent}  n={v.size}  mean_of_slice_maxima={v.mean():.4f} px  "
        f"std_across_slice_maxima={v.std():.4f} px",
        f"{indent}  min={v.min():.4f}  p10={np.percentile(v, 10):.4f}  "
        f"p25={np.percentile(v, 25):.4f}  median={np.median(v):.4f}  "
        f"p75={np.percentile(v, 75):.4f}  p90={np.percentile(v, 90):.4f}  "
        f"p99={np.percentile(v, 99):.4f}  max={v.max():.4f} px",
        f"{indent}  → p10 = 10th pct of **slice maxima** (~10% of slices have max ‖φ‖ below this)"
    ]
    return lines


@dataclass
class SplitStats:
    n_files: int = 0
    n_corrupt: int = 0
    qc_true: int = 0
    qc_false: int = 0
    qc_missing: int = 0
    phi_near_zero_count: int = 0
    qc_fail_rel_paths: list[str] = field(default_factory=list)
    phi_max_per_file: list[float] = field(default_factory=list)
    phi_mean_per_file: list[float] = field(default_factory=list)
    phi_mean_interior_per_file: list[float] = field(default_factory=list)


@dataclass
class ScanReport:
    data_dir: Path
    pattern: str = TRIPLET_GLOB
    phi_near_zero_eps: float = 1e-4
    corrupt: list[tuple[str, str]] = field(default_factory=list)
    per_split: dict[str, SplitStats] = field(default_factory=lambda: defaultdict(SplitStats))
    manifest_flagged: set[str] = field(default_factory=set)
    manifest_path: Path | None = None


def scan_dataset(
    data_dir: Path,
    pattern: str = TRIPLET_GLOB,
    phi_near_zero_eps: float = 1e-4,
) -> ScanReport:
    report = ScanReport(
        data_dir=data_dir.resolve(),
        pattern=pattern,
        phi_near_zero_eps=phi_near_zero_eps,
    )
    man = data_dir / "qc_flagged_paths.txt"
    if man.is_file():
        report.manifest_path = man.resolve()
        for line in man.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            report.manifest_flagged.add(line.replace("\\", "/"))

    for split in SPLITS:
        sd = data_dir / split
        st = report.per_split[split]
        if not sd.is_dir():
            continue
        files = sorted(sd.glob(pattern))
        st.n_files = len(files)
        for fp in files:
            ok, reasons = validate_triplet(fp)
            rel = f"{split}/{fp.name}".replace("\\", "/")
            if not ok:
                st.n_corrupt += 1
                report.corrupt.append((str(fp), "; ".join(reasons)))
                continue

            with np.load(fp) as data:
                if "qc_passed" in data.files:
                    qc_val, _ = _unpack_qc_passed(data["qc_passed"])
                    if qc_val:
                        st.qc_true += 1
                    else:
                        st.qc_false += 1
                        st.qc_fail_rel_paths.append(rel)
                else:
                    st.qc_missing += 1
                phi = np.asarray(data["phi"])
                mag = phi_magnitude(phi.astype(np.float64))
                if float(np.max(mag)) < phi_near_zero_eps:
                    st.phi_near_zero_count += 1
                st.phi_max_per_file.append(float(np.max(mag)))
                st.phi_mean_per_file.append(float(np.mean(mag)))
                if "valid_mask" in data.files:
                    vm_bool, _ = _valid_mask_to_bool(data["valid_mask"], mag.shape)
                    if vm_bool is not None and np.any(vm_bool):
                        st.phi_mean_interior_per_file.append(float(np.mean(mag[vm_bool])))

    return report


def print_report(report: ScanReport) -> None:
    print("=" * 60)
    print(f"Synthetic triplet review: {report.data_dir}")
    print("=" * 60)
    print("(1) Near-identity φ (full slice: max ‖φ‖ < ε)")
    print("(2) Mean of per-sample-φ-means on full slice")
    print("(3) Mean of per-sample-φ-means on interior (valid_mask)")
    print("(4) Distribution of max ‖φ‖ per sample (full slice) across samples in split")
    print()

    if report.manifest_path and report.manifest_path.exists():
        print(f"Using qc_flagged_paths.txt (from create_synth_data): {report.manifest_path}")
        if report.manifest_flagged:
            print(f"  Listed: {len(report.manifest_flagged)} path(s) with qc_passed=False")
        print()

    for split in SPLITS:
        st = report.per_split[split]
        if st.n_files == 0 and st.n_corrupt == 0:
            print(f"{split:6s}  (no files)")
            print()
            continue
        good = st.n_files - st.n_corrupt
        line = (
            f"{split:6s}  files={st.n_files}  load_ok={good}  corrupt={st.n_corrupt}"
        )
        if good > 0:
            line += f"  qc_ok={st.qc_true}  qc_fail={st.qc_false}  qc_absent={st.qc_missing}"
        print(line)
        if good > 0:
            nz = st.phi_near_zero_count
            pct = 100.0 * nz / good
            print(
                f"         (1) Near-identity φ (full slice: max ‖φ‖ < {report.phi_near_zero_eps:g} px) "
                f"→ {nz} / {good} ({pct:.1f}%)"
            )
        if st.phi_mean_per_file:
            mean_arr = np.array(st.phi_mean_per_file, dtype=np.float64)
            max_arr = np.array(st.phi_max_per_file, dtype=np.float64)

            for line in _mean_of_means_summary_lines(
                "(2) Mean of per-sample-φ-means on full slice:",
                mean_arr,
            ):
                print(line)

            if st.phi_mean_interior_per_file:
                int_mean = np.array(st.phi_mean_interior_per_file, dtype=np.float64)
                n_int, n_all = len(int_mean), len(st.phi_mean_per_file)
                note = f" [n={n_int} w/ valid_mask]"
                if n_int < n_all:
                    note += f"; {n_all - n_int} skipped"
                for line in _mean_of_means_summary_lines(
                    "(3) Mean of per-sample-φ-means on interior (valid_mask):" + note,
                    int_mean,
                ):
                    print(line)

            for line in _distribution_across_samples_lines(
                "(4) Distribution of max ‖φ‖ per sample (full slice) across samples:",
                max_arr,
            ):
                print(line)

        print()

    if report.manifest_path:
        on_disk = set()
        for split in SPLITS:
            sd = report.data_dir / split
            if not sd.is_dir():
                continue
            for fp in sd.glob(report.pattern):
                on_disk.add(f"{split}/{fp.name}".replace("\\", "/"))
        flagged_on_disk = report.manifest_flagged & on_disk
        stale = report.manifest_flagged - on_disk
        print("-" * 60)
        print("qc_flagged_paths.txt:")
        print(f"  Listed: {len(report.manifest_flagged)}  |  On disk (still present): {len(flagged_on_disk)}")
        if stale:
            print(f"  Stale (not on disk): {len(stale)}")
        npz_fail_paths = set()
        for split in SPLITS:
            for rel in report.per_split[split].qc_fail_rel_paths:
                npz_fail_paths.add(rel)
        manifest_set = report.manifest_flagged
        in_manifest_not_npz = manifest_set - npz_fail_paths
        in_npz_not_manifest = npz_fail_paths - manifest_set
        if in_manifest_not_npz or in_npz_not_manifest:
            print(
                f"  Mismatch vs npz qc_passed=False: "
                f"in_manifest_only={len(in_manifest_not_npz)}  "
                f"in_npz_only={len(in_npz_not_manifest)}"
            )
        elif len(manifest_set) == 0 and len(npz_fail_paths) == 0:
            print("  Manifest and npz: no QC failures listed")
        else:
            print("  Manifest matches npz qc_passed=False paths: yes")

    print("-" * 60)
    if report.corrupt:
        print(f"CORRUPT / INVALID: {len(report.corrupt)} file(s)")
        for p, msg in report.corrupt[:15]:
            print(f"  {p}\n    -> {msg}")
        if len(report.corrupt) > 15:
            print(f"  ... and {len(report.corrupt) - 15} more")
    else:
        print("Integrity: no corrupt or schema-invalid triplets in scanned splits.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Review synthetic *_triplet.npz: integrity and summary stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        "--input-dir",
        type=Path,
        default=Path("./data/IXI_2D_synth_trip"),
        dest="data_dir",
        help="Root with Train/Val/Test/Atlas subfolders.",
    )
    p.add_argument(
        "--corrupt-log",
        type=Path,
        default=Path("synth_trip_corrupt_list.txt"),
        help=(
            "If the scan finds corrupt/invalid .npz files, write their paths and errors here "
            "(tab-separated). Not created when everything passes."
        ),
    )
    p.add_argument(
        "--phi-near-zero-eps",
        type=float,
        default=1e-4,
        metavar="PX",
        help="Count near-identity phi when max ‖φ‖ on slice < this (pixels).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir

    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2
    report = scan_dataset(
        data_dir,
        pattern=TRIPLET_GLOB,
        phi_near_zero_eps=args.phi_near_zero_eps,
    )
    print_report(report)
    if report.corrupt:
        args.corrupt_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.corrupt_log, "w", encoding="utf-8") as log:
            for path, msg in report.corrupt:
                log.write(f"{path}\t{msg}\n")
        print(f"Wrote corrupt list: {args.corrupt_log}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
