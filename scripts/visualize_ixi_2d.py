"""
Visualize random 2D images from .npy and .npz files.

Example usage:
  python3 scripts/visualize_2d_ixi.py --input-dir ./data/raw/IXI_2D/Train --num-samples 9
  python3 scripts/visualize_2d_ixi.py --input-dir ./data/raw/IXI_2D --recursive --num-samples 9 --no-show --save-path ./assets/images/ixi_2d.png
  python3 scripts/visualize_2d_ixi.py --input-dir ./data/raw/IXI_2D --pattern "atlas_slice_*.npy" --num-samples 10
  python3 scripts/visualize_2d_ixi.py --input-dir ./some_npz_dir --array-key image --num-samples 8
"""

import argparse
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np


def collect_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        candidates = sorted(input_dir.rglob(pattern))
    else:
        candidates = sorted(input_dir.glob(pattern))
    return [p for p in candidates if p.suffix.lower() in {".npy", ".npz"}]


def load_image(path: Path, array_key: str | None) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        chosen_key = "npy"
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            keys = list(archive.keys())
            if not keys:
                raise ValueError(f"NPZ file has no arrays: {path}")
            if array_key is not None:
                if array_key not in archive:
                    raise KeyError(
                        f"Array key '{array_key}' not found in {path.name}. "
                        f"Available keys: {keys}"
                    )
                chosen_key = array_key
            else:
                chosen_key = keys[0]
            image = archive[chosen_key]
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape} from {path.name}"
        )
    return image, chosen_key


def visualize_samples(
    input_dir: Path,
    pattern: str,
    recursive: bool,
    num_samples: int,
    seed: int,
    array_key: str | None,
    save_path: Path | None,
    no_show: bool,
) -> None:
    files = collect_files(input_dir, pattern, recursive)
    if not files:
        raise FileNotFoundError(
            f"No .npy/.npz files found in '{input_dir}' with pattern '{pattern}'."
        )

    rng = random.Random(seed)
    sample_count = min(num_samples, len(files), 3)
    selected = rng.sample(files, sample_count)

    cols = sample_count
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4))
    axes = np.atleast_1d(axes).ravel()

    for idx, file_path in enumerate(selected):
        image, key = load_image(file_path, array_key=array_key)
        axes[idx].imshow(image, cmap="gray")
        if file_path.suffix.lower() == ".npz":
            title = f"{file_path.name}\n[{key}]"
        else:
            title = file_path.name
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis("off")

    for idx in range(sample_count, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Random 2D samples - showing {sample_count} of {len(files)} (3-column layout)",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    examples = (
        "Examples:\n"
        "python3 scripts/visualize_ixi_2d.py --input-dir ./data/raw/IXI_2D/Train"
        "python3 scripts/visualize_ixi_2d.py --input-dir ./data/raw/IXI_2D --recursive --no-show --save-path ./assets/images/ixi_2d.png"
        "python3 scripts/visualize_ixi_2d.py --input-dir ./data/raw/IXI_2D --pattern \"atlas_slice_*.npy\""
        "python3 scripts/visualize_ixi_2d.py --input-dir ./data/raw/IXI_2D --array-key image"
    )
    parser = argparse.ArgumentParser(
        description="Visualize random 2D images from .npy and .npz files.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data/raw/IXI_2D"),
        help="Directory containing .npy/.npz image files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern used inside input-dir (e.g. '*.npy', '*.npz', 'atlas_slice_*.npy').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories under input-dir.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random images to display (max 3; shown as 3 columns).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--array-key",
        type=str,
        default=None,
        help="For .npz files, choose which key to visualize. Defaults to first key.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional output path for saving the figure (e.g. ./outputs/ixi2d.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; useful for remote/headless runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_samples(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        num_samples=args.num_samples,
        seed=args.seed,
        array_key=args.array_key,
        save_path=args.save_path,
        no_show=args.no_show,
    )
