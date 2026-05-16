#!/usr/bin/env bash
# Start Jupyter Lab on NRP: uses venv on PVC and repo under /files (see docs/nautilus.md).
# Optional: set env JUPYTER_TOKEN on the Deployment for auth (otherwise empty token; use port-forward only).
set -euo pipefail

VENV="/files/venvs/unc"
VENV_PY="${VENV}/bin/python"
VENV_JUPYTER="${VENV}/bin/jupyter"

err() { echo "ERROR: $*" >&2; exit 1; }

if [[ ! -f "${VENV}/bin/activate" ]]; then
  err "Missing venv at ${VENV}. Exec into unc-dev / unc-heavy and run: bash deploy/nautilus/scripts/setup_venv.sh"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

if [[ ! -x "${VENV_PY}" ]]; then
  err "Missing ${VENV_PY}"
fi

REPO="${UNC_REPO:-/files/repo/uncertainty-quantification}"
if [[ ! -d "${REPO}" ]]; then
  echo "WARNING: ${REPO} missing — using /files as notebook root until you git clone there." >&2
  REPO=/files
fi
export PYTHONPATH="${REPO}"
cd "${REPO}"

_ENV="${REPO}/deploy/nautilus/scripts/env.sh"
if [[ -f "${_ENV}" ]]; then
  # shellcheck source=env.sh
  source "${_ENV}"
fi

# Prefer venv Jupyter (avoid conda/image "jupyter" on PATH).
if [[ -x "${VENV_JUPYTER}" ]]; then
  JLAUNCH=("${VENV_JUPYTER}" lab)
elif [[ -x "${VENV}/bin/jupyter-lab" ]]; then
  JLAUNCH=("${VENV}/bin/jupyter-lab")
elif "${VENV_PY}" -c "import jupyterlab" 2>/dev/null; then
  JLAUNCH=("${VENV_PY}" -m jupyterlab)
else
  err "jupyterlab not in venv. Run: source ${VENV}/bin/activate && pip install jupyterlab"
fi

if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--ServerApp.token="${JUPYTER_TOKEN}")
else
  TOKEN_ARGS=(--ServerApp.token="" --ServerApp.password="")
fi

echo "Starting Jupyter Lab on 0.0.0.0:8888, root=${REPO}" >&2
# NRP pods run as root; Jupyter refuses to start without this flag.
exec "${JLAUNCH[@]}" \
  --allow-root \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.root_dir="${REPO}" \
  --ServerApp.allow_origin="*" \
  "${TOKEN_ARGS[@]}"
