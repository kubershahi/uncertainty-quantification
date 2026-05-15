#!/usr/bin/env bash
# UniGradICON env (GPU): install CUDA-enabled PyTorch *before* unigradicon so pip
# does not leave you with a CPU-only or mismatched torch.
#
# Default venv: ${HOME}/venvs/unc (DataHub). On Nautilus use:
#   VENV_DIR=/files/venvs/unc bash datahub/setup_unc.sh
#
# 1) Check driver:    nvidia-smi
# 2) Pick a wheel tag (https://pytorch.org/get-started/locally/) — common: cu124
# 3) First-time setup:
#      CUDA_WHEEL=cu124 bash datahub/setup_unc.sh
# 4) Re-run is safe (reuses venv). Rebuild: FORCE=1 bash datahub/setup_unc.sh
#
# Activate: source ${VENV_DIR:-$HOME/venvs/unc}/bin/activate

set -euo pipefail

VENV_DIR="${VENV_DIR:-${HOME}/venvs/unc}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"
FORCE="${FORCE:-0}"

mkdir -p "$(dirname "${VENV_DIR}")"

if [[ -f "${VENV_DIR}/bin/activate" && "${FORCE}" != "1" ]]; then
  echo "Existing venv found at ${VENV_DIR}. Reusing it. (Set FORCE=1 to rebuild.)"
else
  if [[ -d "${VENV_DIR}" ]]; then
    echo "FORCE=1 set: removing existing ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
  fi
  echo "Creating venv at ${VENV_DIR} (CUDA wheel: ${CUDA_WHEEL})..."
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

echo "Installing PyTorch + torchvision (GPU) from pytorch.org..."
pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

echo "Installing UniGradICON + ipykernel..."
pip install --no-cache-dir unigradicon ipykernel torchio

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=unc --display-name "unc"

echo ""
echo "Setup complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then verify GPU:"
echo "  python datahub/resource_checks/diagnose_torch_gpu.py"
