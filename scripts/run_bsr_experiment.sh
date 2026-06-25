#!/usr/bin/env bash
set -euo pipefail

# Robust launcher for BSR-SPT experiments.
# Usage:
#   bash scripts/run_bsr_experiment.sh semantic/dales_bsr
#   bash scripts/run_bsr_experiment.sh semantic/dales_bsr_no_consistency trainer.max_epochs=10

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_bsr_experiment.sh <experiment_alias> [hydra_overrides...]"
    echo "Example: bash scripts/run_bsr_experiment.sh semantic/dales_bsr trainer.max_epochs=10"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/workspace/intel_1t/jts/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "conda.sh not found: ${CONDA_ROOT}/etc/profile.d/conda.sh"
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_ROOT}"

EXPERIMENT_ALIAS="$1"
shift

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "Running: python src/train.py experiment=${EXPERIMENT_ALIAS} $*"

python src/train.py "experiment=${EXPERIMENT_ALIAS}" "$@"
