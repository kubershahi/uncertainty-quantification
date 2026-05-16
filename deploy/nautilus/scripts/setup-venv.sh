#!/usr/bin/env bash
# One-time venv on PVC (persists across pods). Run inside unc-dev:
#   bash /files/repo/uncertainty-quantification/deploy/nautilus/scripts/setup-venv.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export VENV_DIR=/files/venvs/unc
export CUDA_WHEEL="${CUDA_WHEEL:-cu124}"
export FORCE="${FORCE:-0}"

bash "${REPO_ROOT}/experiments/setup_unc.sh"

echo ""
echo "Nautilus venv ready at ${VENV_DIR}"
echo "Activate in any pod: source ${VENV_DIR}/bin/activate"
