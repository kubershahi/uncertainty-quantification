#!/usr/bin/env bash
# UniGradICON env on DataHub (GPU): install CUDA-enabled PyTorch *before* unigradicon so pip
# does not leave you with a CPU-only or mismatched torch.
#
# 1) Check driver: nvidia-smi
# 2) Pick a wheel tag that matches (see https://pytorch.org/get-started/locally/)
#    Common: cu124 (CUDA 12.x), cu121, cu118
# 3) Run:
#    CUDA_WHEEL=cu124 bash datahub/setup_unigrad.sh
#
set -euo pipefail

VENV_DIR="${VENV_DIR:-/tmp/unigrad}"
# PyTorch CUDA wheel index suffix: cu124, cu121, cu118, etc.
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"

echo "Creating venv at ${VENV_DIR} (CUDA wheel: ${CUDA_WHEEL})..."

python3 -m venv "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

echo "Installing PyTorch + torchvision (GPU) from pytorch.org..."
pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

echo "Installing UniGradICON + ipykernel..."
pip install --no-cache-dir unigradicon ipykernel torchio

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=unigrad --display-name "unigrad"

echo ""
echo "Setup complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then verify GPU kernels:"
echo "  python datahub/diagnose_torch_gpu.py"
