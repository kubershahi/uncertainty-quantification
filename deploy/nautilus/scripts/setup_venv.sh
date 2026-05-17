#!/usr/bin/env bash
# One-time venv on PVC (persists across pods). Prefer unc-heavy / unc-jupyter (pytorch image).
#   bash /files/repo/uncertainty-quantification/deploy/nautilus/scripts/setup_venv.sh
# On unc-dev (ubuntu:22.04) first: apt-get update && apt-get install -y python3-venv python3-pip
# If venv was built on another image: FORCE=1 bash .../setup_venv.sh
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
