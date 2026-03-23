"""
Generate UniGradICON fivers (*_fiver.npz) from synthetic triplet npz files.

Reads Train/Val/Test/Atlas under --input-path; writes phi_pred, phi_true, phi_diff,
error_map, **valid_mask**, and **qc_passed** (copied from the triplet).

Notes:
- By default **only triplets with ``qc_passed`` True** are processed (skips False or missing key).
Pass ``--process-all-triplets`` to include failures / legacy archives without the flag.
- phi_AB_vectorfield from UniGradICON is a *position map* (identity + displacement) in ICON's
normalized coordinates; this script uses (phi - identity) and scales to pixel displacement.

Example usage:
  python create_unigrad_data.py
  python create_unigrad_data.py --max-per-split 2 --output-path ./data/IXI_2D_unigrad_fiver/
  python create_unigrad_data.py --input-path ./data/IXI_2D_synth_trip/ --output-path ./data/IXI_2D_unigrad_fiver/

"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unigradicon import get_unigradicon

_DH = Path(__file__).resolve().parent
if str(_DH) not in sys.path:
    sys.path.insert(0, str(_DH))
import create_synth_data as csd


def resolve_device(mode: str) -> torch.device:
    """Pick torch.device. 'auto' probes CUDA; falls back to CPU if kernels fail (common on DataHub)."""
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    # auto
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.randn(1, 1, 8, 8, 8, device="cuda")
        y = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        torch.cuda.synchronize()
        del x, y
        return torch.device("cuda")
    except Exception as exc:
        print(
            f"CUDA reported available but a probe failed ({exc!r}); "
            "using CPU. Pass --device cpu to skip this message, or fix PyTorch/CUDA for this GPU."
        )
        return torch.device("cpu")


def _valid_mask_for_triplet(data: np.lib.npyio.NpzFile, image: np.ndarray) -> np.ndarray:
    """Use triplet mask if present; else same interior mask as create_synth_data."""
    if "valid_mask" in data.files:
        return np.asarray(data["valid_mask"], dtype=bool)
    h, w = int(image.shape[0]), int(image.shape[1])
    return csd.interior_valid_mask(h, w, csd.INTERIOR_MARGIN)


def _qc_passed_for_triplet(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Copy triplet flag if present; else ``True`` (only used with ``--process-all-triplets``)."""
    if "qc_passed" in data.files:
        return np.asarray(data["qc_passed"])
    return np.array(True)


def _triplet_should_process(data: np.lib.npyio.NpzFile, *, process_all_triplets: bool) -> bool:
    """Default: only ``qc_passed`` present and True. With ``process_all_triplets``, always True."""
    if process_all_triplets:
        return True
    if "qc_passed" not in data.files:
        return False
    q = np.asarray(data["qc_passed"])
    if q.size != 1:
        return False
    return bool(q.reshape(-1)[0])


def preprocess_for_unigrad(img_tensor):
    """Robust preprocessing with epsilon to prevent NaNs on flat slices"""
    # 1. Normalization (Quantile-based)
    im_min = torch.min(img_tensor)
    im_max = torch.quantile(img_tensor.view(-1), 0.99)
    
    # Epsilon prevents division by zero if im_max == im_min
    denom = torch.clamp(im_max - im_min, min=1e-5)
    img = torch.clip(img_tensor, im_min, im_max)
    img = (img - im_min) / denom
    
    # 2. Pseudo-3D Stacking: UniGradICON expects 3D
    # (1, 1, 160, 192) -> (1, 1, 5, 160, 192)
    img = img.unsqueeze(2).repeat(1, 1, 5, 1, 1)
    
    # 3. Required Resolution: 175x175x175
    return F.interpolate(img, [175, 175, 175], mode="trilinear", align_corners=False)

def run_fiver_generation(
    input_root,
    output_root,
    max_per_split=None,
    device: torch.device | None = None,
    *,
    process_all_triplets: bool = False,
):
    if device is None:
        device = resolve_device("auto")
    print(f"Loading UniGradICON on {device}...")
    net = get_unigradicon().to(device)
    net.eval()

    splits = ['Train', 'Val', 'Test', 'Atlas']

    for split in splits:
        input_dir = os.path.join(input_root, split)
        output_dir = os.path.join(output_root, split)
        os.makedirs(output_dir, exist_ok=True)
        
        files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npz'))
        if max_per_split is not None:
            files = files[: max_per_split]
        print(
            f"Scanning {len(files)} triplet(s) in {split}"
            + (f" (max {max_per_split} filenames)" if max_per_split else "")
            + ("; only qc_passed=True" if not process_all_triplets else "; all triplets")
            + "..."
        )

        n_skip_qc = 0
        for fname in tqdm(files):
            path = os.path.join(input_dir, fname)
            with np.load(path) as data:
                if not _triplet_should_process(data, process_all_triplets=process_all_triplets):
                    n_skip_qc += 1
                    continue
                image = np.asarray(data["image"])
                warped = np.asarray(data["warped"])
                phi_raw = np.asarray(data["phi"])
                valid_mask = _valid_mask_for_triplet(data, image)
                qc_passed = _qc_passed_for_triplet(data)

            orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
            I_fixed = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            I_moving = torch.from_numpy(warped).float().unsqueeze(0).unsqueeze(0)
            # Synthetic triplet phi is [row_disp, col_disp] (meshgrid indexing='ij').
            # Align with [dx, dy] = [col, row] to match phi_pred scaling (dx ~ width, dy ~ height).
            phi_true = np.stack([phi_raw[1], phi_raw[0]], axis=0)

            # 1. Preprocess and Inference
            source = preprocess_for_unigrad(I_moving).to(device)
            target = preprocess_for_unigrad(I_fixed).to(device)

            # --- UniGradICON geometry (see preprocess_for_unigrad) ---
            # One 2D slice is NOT stacked with the other image. Each of (fixed) and (moving) is
            # processed alone: (1,1,160,192) -> repeat same slice 5x along depth -> (1,1,5,160,192)
            # -> trilinear resize -> (1,1,175,175,175). The network always sees a full 3D cube.
            # The model returns a 3D displacement field: shape (1, 3, 175, 175, 175), one vector
            # per voxel (3 channels = depth, H, W axes of the 5D tensor), NOT a single 175x175 map.
            with torch.no_grad():
                net(source, target)
                # phi_AB_vectorfield is warped *position* in ICON coords: identity + displacement.
                # identity_map uses spacing=1/(N-1); coords span ~[0,1] per axis (see icon_registration).
                identity = net.identity_map
                phi_disp_175 = net.phi_AB_vectorfield - identity

            # 2. Bring the 3D field back to the same in-plane grid as your 2D triplet (160 x 192).
            # We resample (D,H,W) from (175,175,175) -> (5, orig_h, orig_w): depth back to 5 (like the
            # pseudo-stack), height/width back to original slice size. Then we take the *middle* depth
            # slice (index 2) as the single 2D in-plane displacement for that slice.
            phi_rescaled = F.interpolate(
                phi_disp_175,
                [5, orig_h, orig_w],
                mode="trilinear",
                align_corners=True,
            )
            # Layout (B, C, D, H, W): channel 0 = displacement along depth axis, 1 along H, 2 along W.
            # For a 2D slice we want in-plane motion only -> channels 1 and 2 (not 0).
            phi_plane = phi_rescaled[0, 1:3, 2, :, :].cpu().numpy()

            # 3. ICON displacement is in normalized coord units (~[0,1] span per axis); convert to pixels.
            # delta_px ≈ delta_coord * (orig_dim - 1) (align_corners-style indexing).
            phi_pred_px = np.zeros((2, orig_h, orig_w), dtype=np.float32)
            phi_pred_px[0] = phi_plane[1] * (orig_w - 1)  # col / width
            phi_pred_px[1] = phi_plane[0] * (orig_h - 1)  # row / height

            # 4. Calculate Vector Residual (phi_diff) and Scalar Error Map
            phi_diff = phi_true - phi_pred_px
            # Euclidean distance: sqrt(dx_diff^2 + dy_diff^2)
            error_map = np.sqrt(np.sum(phi_diff**2, axis=0))

            # 5. Save fiver (always include valid_mask + qc_passed for downstream training)
            out_name = fname.replace('_triplet.npz', '_fiver.npz')
            np.savez_compressed(
                os.path.join(output_dir, out_name),
                image=image,
                warped=warped,
                phi_true=phi_true,
                phi_pred=phi_pred_px,
                phi_diff=phi_diff,
                error_map=error_map,
                valid_mask=valid_mask,
                qc_passed=qc_passed,
            )

        if n_skip_qc and not process_all_triplets:
            print(
                f"  {split}: skipped {n_skip_qc} triplet(s) "
                f"(qc_passed=False or missing qc_passed)"
            )

def parse_args():
    examples = """
        Examples:
        python create_unigrad_data.py
        python create_unigrad_data.py --max-per-split 2 --output-path ./data/IXI_2D_unigrad_fiver/
        python create_unigrad_data.py --process-all-triplets
        python create_unigrad_data.py --device cpu
        python create_unigrad_data.py --input-path ./data/IXI_2D_synth_trip/ --output-path ./data/IXI_2D_unigrad_fiver/
    """.strip()
    
    p = argparse.ArgumentParser(
        description="Generate UniGradICON fivers (phi_pred, phi_diff, error_map) from synthetic triplets.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-path",
        type=str,
        default="./data/IXI_2D_synth_trip/",
        help="Root folder with Train/Val/Test/Atlas triplet .npz files.",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="./data/IXI_2D_unigrad_fiver/",
        help="Where to write _fiver.npz outputs (mirrors split subfolders).",
    )
    p.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        metavar="N",
        help="If set, only process the first N files per split (sorted by name). Use 2 for a smoke test.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. Use 'cpu' if CUDA fails with no kernel image (GPU/PyTorch mismatch). "
        "Default 'auto' tries CUDA then falls back to CPU if a probe fails.",
    )
    p.add_argument(
        "--process-all-triplets",
        action="store_true",
        help="Process every .npz (including qc_passed=False and archives without qc_passed). "
        "Default: only triplets with qc_passed=True.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = resolve_device(args.device)
    run_fiver_generation(
        args.input_path,
        args.output_path,
        max_per_split=args.max_per_split,
        device=device,
        process_all_triplets=args.process_all_triplets,
    )