# Source in pods or Jupyter: source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
export FILES_ROOT=/files
export REPO_ROOT=/files/repo/uncertainty-quantification
export VENV_DIR=/files/venvs/unc
# Persistent $HOME on PVC (git config, SSH keys, known_hosts — not in the container layer).
export HOME=/files/home/root
mkdir -p "${HOME}"

_ssh_dir="${HOME}/.ssh"
mkdir -p "${_ssh_dir}"
chmod 700 "${_ssh_dir}" 2>/dev/null || true

# One-time migration from legacy /files/.ssh/ (remove after all PVCs are migrated).
_legacy_ssh=/files/.ssh
if [[ -d "${_legacy_ssh}" ]]; then
  for f in "${_legacy_ssh}"/*; do
    [[ -e "${f}" ]] || continue
    _dest="${_ssh_dir}/$(basename "${f}")"
    if [[ -L "${_dest}" ]]; then
      rm -f "${_dest}"
    fi
    if [[ ! -e "${_dest}" ]]; then
      mv "${f}" "${_dest}"
    fi
  done
  rmdir "${_legacy_ssh}" 2>/dev/null || true
fi

_ssh_key="${_ssh_dir}/id_ed25519"
if [[ -f "${_ssh_key}" ]]; then
  _ssh_cmd="ssh -i ${_ssh_key} -o StrictHostKeyChecking=accept-new"
  export GIT_SSH_COMMAND="${_ssh_cmd}"
  git config --global core.sshCommand "${_ssh_cmd}" 2>/dev/null || true
fi

# Data under the repo clone on PVC (gitignored via datasets/** in .gitignore)
export DATASETS_ROOT="${REPO_ROOT}/datasets"
export IXI_2D_ROOT="${DATASETS_ROOT}/IXI_2D"
export IXI_ROOT="${DATASETS_ROOT}/IXI"
export UNIGRAD_IO_2D_OUT="${DATASETS_ROOT}/IXI_2D_unigrad_io"
export UNIGRAD_IO_3D_OUT="${DATASETS_ROOT}/IXI_unigrad_io"

# Experiment outputs (in repo — git pull on laptop for figures / run logs)
export ASSETS_ROOT="${REPO_ROOT}/assets"
export ASSETS_IMAGES="${ASSETS_ROOT}/images"
export ASSETS_RUNS="${ASSETS_ROOT}/runs"
export RUNS_ROOT="${ASSETS_RUNS}"
export SWEEP_IO_SAVE="${ASSETS_IMAGES}/unigrad-io/sweep_io.png"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B run data on PVC (not /files/wandb). Login later: wandb login → ~/.netrc under $HOME.
export WANDB_DIR="${HOME}/wandb"
mkdir -p "${WANDB_DIR}"

_venv_py="${VENV_DIR}/bin/python"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -x "${_venv_py}" ]] && "${_venv_py}" -c "import sys" >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
  else
    echo "WARNING: venv at ${VENV_DIR} is broken (Python interpreter missing or wrong path)." >&2
    echo "  Rebuild (unc-dev, unc-heavy, or unc-jupyter):" >&2
    echo "    FORCE=1 bash ${REPO_ROOT}/deploy/nautilus/scripts/setup_venv.sh" >&2
  fi
fi
