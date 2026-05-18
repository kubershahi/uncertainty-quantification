#!/usr/bin/env bash
# One-time venv on PVC (persists across pods). Run from unc-dev, unc-heavy, or unc-jupyter.
#   bash /files/repo/uncertainty-quantification/deploy/nautilus/scripts/setup_venv.sh
# Rebuild: FORCE=1 bash .../setup_venv.sh  (mv old venv aside; rm runs in background on PVC)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VENV_DIR=/files/venvs/unc
export CUDA_WHEEL="${CUDA_WHEEL:-cu124}"
export FORCE="${FORCE:-0}"

bash "${SCRIPT_DIR}/setup_unc.sh"

echo ""
echo "Nautilus venv ready at ${VENV_DIR}"
echo "Activate in any pod: source ${VENV_DIR}/bin/activate"
echo "W&B (optional): wandb login  →  run data under /files/home/root/wandb"
