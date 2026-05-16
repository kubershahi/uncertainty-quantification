# Source in pods or Jupyter: source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
export FILES_ROOT=/files
export REPO_ROOT=/files/repo/uncertainty-quantification
export VENV_DIR=/files/venvs/unc
# Persistent $HOME on PVC (git config, SSH known_hosts, credential store — not in the container layer).
export HOME=/files/home/root
mkdir -p "${HOME}" /files/.ssh

# SSH keys + optional config on PVC at /files/.ssh/ (not under $HOME by default).
# Symlink into ${HOME}/.ssh so plain `ssh -T git@github.com` finds them after `source env.sh`.
if [[ -d /files/.ssh ]]; then
  mkdir -p "${HOME}/.ssh"
  chmod 700 /files/.ssh "${HOME}/.ssh" 2>/dev/null || true
  for f in /files/.ssh/*; do
    [[ -e "${f}" ]] || continue
    ln -sfn "${f}" "${HOME}/.ssh/$(basename "${f}")"
  done
fi
if [[ -f /files/.ssh/id_ed25519 ]]; then
  export GIT_SSH_COMMAND="ssh -i /files/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new"
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

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi
