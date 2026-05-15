#!/usr/bin/env bash
# Smoke test before full dataset job:
#   bash deploy/nautilus/scripts/run-io-data-smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"
python datahub/unigrad-io/create_unigrad_io_data.py \
  --ixi-root "${IXI_ROOT}" \
  --output-path "${UNIGRAD_IO_OUT}_smoke" \
  --splits Train \
  --max-per-split 3

echo "Smoke outputs: ${UNIGRAD_IO_OUT}_smoke/Train/"
