# Source inside any pod with /files mounted:
#   source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
export FILES_ROOT=/files
export REPO_ROOT=/files/repo/uncertainty-quantification
export VENV_DIR=/files/venvs/unc
export IXI_ROOT=/files/datasets/IXI_2D
export UNIGRAD_IO_OUT=/files/datasets/IXI_2D_unigrad_io
export RUNS_ROOT=/files/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi
