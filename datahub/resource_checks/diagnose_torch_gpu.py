#!/usr/bin/env python3
"""
Print DataHub GPU + PyTorch CUDA info and run a small CUDA op (same family as UniGradICON).

Run inside your venv after setup:
  source /tmp/unigrad/bin/activate   # or your venv path
  python datahub/diagnose_torch_gpu.py

If the probe fails with "no kernel image", reinstall PyTorch with a CUDA wheel that matches
the driver (see setup_unigrad.sh and https://pytorch.org/get-started/locally/).
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        print("--- nvidia-smi (first lines) ---")
        lines = (r.stdout or "").splitlines()[:24]
        print("\n".join(lines) if lines else "(empty)")
    except FileNotFoundError:
        print("nvidia-smi not found (no NVIDIA driver in PATH?).")
    except Exception as e:
        print(f"nvidia-smi error: {e}")

    print("--- python / torch ---")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        print(f"import torch failed: {e}")
        sys.exit(2)

    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.version.cuda = {torch.version.cuda}")
        print(f"device[0] = {torch.cuda.get_device_name(0)}")
        print(f"capability (major, minor) = {torch.cuda.get_device_capability(0)}")

    if not torch.cuda.is_available():
        print("\nCUDA not available to PyTorch — install a CUDA build of torch (see setup_unigrad.sh).")
        sys.exit(1)

    try:
        x = torch.randn(1, 1, 8, 8, 8, device="cuda")
        y = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        torch.cuda.synchronize()
        print("\nCUDA probe (avg_pool3d): OK — GPU kernels run; use --device cuda or auto.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCUDA probe FAILED: {e!r}")
        print(
            "Typical fix: reinstall PyTorch from https://download.pytorch.org/whl/ "
            "with a cu* tag matching your driver (e.g. cu124, cu121). "
            "See datahub/setup_unigrad.sh variable CUDA_WHEEL."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
