#!/usr/bin/env bash
# CUDA PyTorch + UniGradICON into ${VENV_DIR:-$HOME/venvs/unc}. Example Nautilus: VENV_DIR=/files/venvs/unc bash scripts/setup_unc.sh
# Vars: CUDA_WHEEL=cu124 (default), FORCE=1 to recreate venv.

set -euo pipefail

VENV_DIR="${VENV_DIR:-${HOME}/venvs/unc}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"
FORCE="${FORCE:-0}"

mkdir -p "$(dirname "${VENV_DIR}")"

if [[ -f "${VENV_DIR}/bin/activate" && "${FORCE}" != "1" ]]; then
  echo "Existing venv found at ${VENV_DIR}. Reusing it. (Set FORCE=1 to rebuild.)"
else
  if [[ -d "${VENV_DIR}" ]]; then
    echo "FORCE=1: removing existing ${VENV_DIR} (large tree on PVC — may take several minutes)..."
    rm -rf "${VENV_DIR}"
    echo "Removed ${VENV_DIR}."
  fi
  echo "Creating venv at ${VENV_DIR} (CUDA wheel: ${CUDA_WHEEL})..."
  python3 -m venv "${VENV_DIR}"
  echo "Venv created."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch + torchvision (GPU) — often 15–30 min, downloads ~2–3 GB..."
pip install --no-cache-dir torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
echo "PyTorch install done."

echo "Installing UniGradICON, JupyterLab, ipykernel, wandb, …"
pip install --no-cache-dir unigradicon ipykernel torchio jupyterlab wandb
echo "Extra packages done."

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=unc --display-name "unc"

echo ""
echo "Setup complete. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then verify GPU:"
echo "python experiments/resource_checks/diagnose_torch_gpu.py"
