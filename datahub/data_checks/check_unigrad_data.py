#!/usr/bin/env python3
"""
Post-generation review for UniGradICON fivers (*_fiver.npz).

  1) Integrity: load each archive, required keys, shapes, consistency of phi_diff / error_map.
  2) Summary: per-split counts, qc_passed breakdown (if present), and **mean statistics**:
     per sample, the **mean over the slice** of ‖φ_true‖, ‖φ_pred‖, and ‖φ_true−φ_pred‖
     (the latter equals ``error_map``). When ``valid_mask`` is present, the same three
     **interior** means.

Examples:
  python data_checks/check_unigrad_data.py --data-dir ./data/IXI_2D_unigrad_fiver/
  python data_checks/check_unigrad_data.py --data-dir ./data/IXI_2D_unigrad_fiver/ --verbose
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED_KEYS = frozenset(
    {"image", "warped", "phi_true", "phi_pred", "phi_diff", "error_map", "valid_mask", "qc_passed"}
)
EXPECTED_HW = (160, 192)  # IXI 2D slice
SPLITS = ("Train", "Val", "Test", "Atlas")
FIVER_GLOB = "*_fiver.npz"


def check_one_fiver(path: Path, rtol: float, atol: float) -> list[str]:
    errors: list[str] = []
    try:
        with np.load(path) as data:
            keys = set(data.files)
            missing = REQUIRED_KEYS - keys
            extra = keys - REQUIRED_KEYS
            if missing:
                errors.append(f"missing keys {sorted(missing)}")
            if extra:
                errors.append(f"unexpected keys {sorted(extra)}")
            if missing or extra:
                return errors

            image = data["image"]
            warped = data["warped"]
            phi_true = data["phi_true"]
            phi_pred = data["phi_pred"]
            phi_diff = data["phi_diff"]
            error_map = data["error_map"]

            for name, arr in [
                ("image", image),
                ("warped", warped),
                ("error_map", error_map),
            ]:
                if arr.ndim != 2:
                    errors.append(f"{name} ndim={arr.ndim}, expected 2")

            for name, arr in [
                ("phi_true", phi_true),
                ("phi_pred", phi_pred),
                ("phi_diff", phi_diff),
            ]:
                if arr.ndim != 3 or arr.shape[0] != 2:
                    errors.append(
                        f"{name} shape={getattr(arr, 'shape', None)}, expected (2, H, W)"
                    )

            if image.shape != warped.shape:
                errors.append(
                    f"image shape {image.shape} != warped shape {warped.shape}"
                )
            if image.shape != error_map.shape:
                errors.append(
                    f"image shape {image.shape} != error_map shape {error_map.shape}"
                )
            if phi_true.shape != phi_pred.shape or phi_true.shape != phi_diff.shape:
                errors.append(
                    f"phi shape mismatch: true {phi_true.shape} pred {phi_pred.shape} diff {phi_diff.shape}"
                )

            if image.shape != EXPECTED_HW:
                errors.append(
                    f"image shape {image.shape}, expected {EXPECTED_HW} (IXI 2D)"
                )

            if not errors:
                diff_expected = phi_true - phi_pred
                if not np.allclose(phi_diff, diff_expected, rtol=rtol, atol=atol):
                    max_err = float(np.max(np.abs(phi_diff - diff_expected)))
                    errors.append(
                        f"phi_diff != phi_true - phi_pred (max abs err {max_err:.6g})"
                    )

                err_expected = np.sqrt(np.sum(phi_diff**2, axis=0))
                if not np.allclose(error_map, err_expected, rtol=rtol, atol=atol):
                    max_err = float(np.max(np.abs(error_map - err_expected)))
                    errors.append(
                        f"error_map != ‖phi_diff‖ (L2 norm per pixel; max abs err {max_err:.6g})"
                    )

                if np.any(~np.isfinite(phi_true)) or np.any(~np.isfinite(phi_pred)):
                    errors.append("non-finite values in phi_true or phi_pred")

    except Exception as e:
        errors.append(f"load failed: {e}")
    return errors


def _phi_mag(phi: np.ndarray) -> np.ndarray:
    return np.sqrt(phi[0] ** 2 + phi[1] ** 2)


def _unpack_qc_passed(raw) -> tuple[bool | None, str | None]:
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


def _mean_of_means_summary_lines(
    heading: str,
    per_sample_means: np.ndarray,
    *,
    indent: str = "         ",
) -> list[str]:
    """
    Each sample contributes one scalar (e.g. mean ‖φ‖ on slice). Report spread across split.
    """
    v = np.asarray(per_sample_means, dtype=np.float64)
    if v.size == 0:
        return [f"{indent}{heading}", f"{indent}  (no samples)"]
    return [
        f"{indent}{heading}",
        f"{indent}  n={v.size}  mean_of_per_sample_means={v.mean():.4f} px  "
        f"min={v.min():.4f} px  max={v.max():.4f} px  std={v.std():.4f} px",
    ]


@dataclass
class FiverSplitStats:
    n_files: int = 0
    n_corrupt: int = 0
    qc_true: int = 0
    qc_false: int = 0
    qc_missing: int = 0
    qc_fail_rel_paths: list[str] = field(default_factory=list)
    mean_norm_phi_true: list[float] = field(default_factory=list)
    mean_norm_phi_pred: list[float] = field(default_factory=list)
    mean_norm_phi_diff: list[float] = field(default_factory=list)
    mean_norm_phi_true_interior: list[float] = field(default_factory=list)
    mean_norm_phi_pred_interior: list[float] = field(default_factory=list)
    mean_norm_phi_diff_interior: list[float] = field(default_factory=list)


@dataclass
class FiverScanReport:
    data_dir: Path
    pattern: str = FIVER_GLOB
    rtol: float = 1e-5
    atol: float = 1e-6
    corrupt: list[tuple[str, str]] = field(default_factory=list)
    per_split: dict[str, FiverSplitStats] = field(
        default_factory=lambda: defaultdict(FiverSplitStats)
    )


def scan_fivers(
    data_dir: Path,
    pattern: str = FIVER_GLOB,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    *,
    verbose: bool = False,
) -> FiverScanReport:
    report = FiverScanReport(
        data_dir=data_dir.resolve(),
        pattern=pattern,
        rtol=rtol,
        atol=atol,
    )
    for split in SPLITS:
        sd = data_dir / split
        st = report.per_split[split]
        if not sd.is_dir():
            continue
        for fp in sorted(sd.glob(pattern)):
            st.n_files += 1
            rel = f"{split}/{fp.name}".replace("\\", "/")
            errs = check_one_fiver(fp, rtol, atol)
            if errs:
                st.n_corrupt += 1
                report.corrupt.append((str(fp.resolve()), "; ".join(errs)))
                continue

            if verbose:
                print(f"  OK   {split}/{fp.name}")

            with np.load(fp) as data:
                phi_true = np.asarray(data["phi_true"])
                phi_pred = np.asarray(data["phi_pred"])
                error_map = np.asarray(data["error_map"])
                mag_t = _phi_mag(phi_true.astype(np.float64))
                mag_p = _phi_mag(phi_pred.astype(np.float64))
                st.mean_norm_phi_true.append(float(np.mean(mag_t)))
                st.mean_norm_phi_pred.append(float(np.mean(mag_p)))
                st.mean_norm_phi_diff.append(float(np.mean(error_map)))

                if "qc_passed" in data.files:
                    qc_val, _ = _unpack_qc_passed(data["qc_passed"])
                    if qc_val:
                        st.qc_true += 1
                    else:
                        st.qc_false += 1
                        st.qc_fail_rel_paths.append(rel)
                else:
                    st.qc_missing += 1

                if "valid_mask" in data.files:
                    vm_bool, vm_err = _valid_mask_to_bool(
                        data["valid_mask"], error_map.shape
                    )
                    if vm_err is None and vm_bool is not None and np.any(vm_bool):
                        m = vm_bool
                        st.mean_norm_phi_true_interior.append(float(np.mean(mag_t[m])))
                        st.mean_norm_phi_pred_interior.append(float(np.mean(mag_p[m])))
                        st.mean_norm_phi_diff_interior.append(
                            float(np.mean(error_map[m]))
                        )

    return report


def print_fiver_report(report: FiverScanReport) -> None:
    total = sum(report.per_split[s].n_files for s in SPLITS)
    print("=" * 60)
    print(f"UniGradICON fiver review: {report.data_dir}")
    print(f"Total *_fiver.npz scanned: {total}")
    print("=" * 60)
    print("Per sample: mean over slice (full image) of ‖φ_true‖, ‖φ_pred‖, ‖φ_true−φ_pred‖.")
    print("  (‖φ_true−φ_pred‖ equals error_map at each pixel.)")
    print(
        "When valid_mask is present: (4–6) same three quantities on interior only "
        "(valid_mask=True)."
    )
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
            line += (
                f"  qc_ok={st.qc_true}  qc_fail={st.qc_false}  "
                f"qc_absent={st.qc_missing}"
            )
        print(line)

        if st.mean_norm_phi_true:
            mt = np.array(st.mean_norm_phi_true, dtype=np.float64)
            mp = np.array(st.mean_norm_phi_pred, dtype=np.float64)
            md = np.array(st.mean_norm_phi_diff, dtype=np.float64)

            for line in _mean_of_means_summary_lines(
                "(1) Per-sample mean of ‖φ_true‖ on full slice:",
                mt,
            ):
                print(line)
            for line in _mean_of_means_summary_lines(
                "(2) Per-sample mean of ‖φ_pred‖ on full slice:",
                mp,
            ):
                print(line)
            for line in _mean_of_means_summary_lines(
                "(3) Per-sample mean of ‖φ_true−φ_pred‖ on full slice (= mean error_map):",
                md,
            ):
                print(line)

            if st.mean_norm_phi_true_interior:
                n_int = len(st.mean_norm_phi_true_interior)
                n_all = len(st.mean_norm_phi_true)
                note = f" [n={n_int} w/ valid_mask]"
                if n_int < n_all:
                    note += f"; {n_all - n_int} skipped"
                mit = np.array(st.mean_norm_phi_true_interior, dtype=np.float64)
                mip = np.array(st.mean_norm_phi_pred_interior, dtype=np.float64)
                mid = np.array(st.mean_norm_phi_diff_interior, dtype=np.float64)
                for line in _mean_of_means_summary_lines(
                    "(4) Per-sample mean of ‖φ_true‖ on interior (valid_mask):" + note,
                    mit,
                ):
                    print(line)
                for line in _mean_of_means_summary_lines(
                    "(5) Per-sample mean of ‖φ_pred‖ on interior (valid_mask):" + note,
                    mip,
                ):
                    print(line)
                for line in _mean_of_means_summary_lines(
                    "(6) Per-sample mean of ‖φ_true−φ_pred‖ on interior (valid_mask):" + note,
                    mid,
                ):
                    print(line)

        print()

    print("-" * 60)
    if report.corrupt:
        print(f"CORRUPT / INVALID: {len(report.corrupt)} file(s)")
        for p, msg in report.corrupt[:15]:
            print(f"  {p}\n    -> {msg}")
        if len(report.corrupt) > 15:
            print(f"  ... and {len(report.corrupt) - 15} more")
    else:
        print("Integrity: no corrupt or schema-invalid fivers in scanned splits.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review UniGradICON *_fiver.npz: integrity and ‖φ‖ / error mean stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        "--input-dir",
        type=Path,
        dest="data_dir",
        default=Path("./data/IXI_2D_unigrad_fiver/"),
        help="Root with Train/Val/Test/Atlas subfolders of *_fiver.npz",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for consistency checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for consistency checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print OK for each file (noisy for large dirs).",
    )
    parser.add_argument(
        "--corrupt-log",
        type=Path,
        default=Path("unigrad_fiver_corrupt_list_ixi_2d.txt"),
        help=(
            "If the scan finds corrupt/invalid .npz files, write path and errors here "
            "(tab-separated). Not created when everything passes."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    print(f"Scanning fivers under: {data_dir.resolve()}")
    print("Required keys:", sorted(REQUIRED_KEYS))
    print("-" * 40)

    report = scan_fivers(
        data_dir,
        pattern=FIVER_GLOB,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )
    print_fiver_report(report)

    if report.corrupt:
        args.corrupt_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.corrupt_log, "w", encoding="utf-8") as log:
            for path, msg in report.corrupt:
                log.write(f"{path}\t{msg}\n")
        print(f"Wrote corrupt list: {args.corrupt_log}")

    return 1 if report.corrupt else 0


if __name__ == "__main__":
    raise SystemExit(main())
