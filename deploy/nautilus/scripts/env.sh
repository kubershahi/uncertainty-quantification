# Source in pods or Jupyter: source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
export FILES_ROOT=/files
export REPO_ROOT=/files/repo/uncertainty-quantification
export VENV_DIR=/files/venvs/unc

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
