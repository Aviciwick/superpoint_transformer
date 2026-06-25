#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${SEED:-0}"
TRAINER_PRECISION="${TRAINER_PRECISION:-16}"
DALES_XY_TILING="${DALES_XY_TILING:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-225}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QUEUE_LOG_DIR="${REPO_ROOT}/logs/ablation_queue"
mkdir -p "${QUEUE_LOG_DIR}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
QUEUE_LOG="${QUEUE_LOG_DIR}/dales_bsr_ablation_24g_${TIMESTAMP}.log"

EXPERIMENTS=(
  "semantic/dales_bsr"
  "semantic/dales_bsr_low_budget"
  "semantic/dales_bsr_random_sampling"
  "semantic/dales_bsr_boundary_hybrid"
  "semantic/dales_bsr_uncertainty_only_prior"
  "semantic/dales_bsr_no_boundary_prior"
  "semantic/dales_bsr_single_direct"
  "semantic/dales_bsr_single_residual_gated"
  "semantic/dales_bsr_no_consistency"
  "semantic/dales_bsr_no_boundary_head"
  "semantic/dales_bsr_mixture_no_diversity"
  "semantic/dales_bsr_no_score_weighting"
  "semantic/dales_bsr_no_loss_warmup"
  "semantic/dales_bsr_no_fusion_warmup"
  "semantic/dales_bsr_three_slot"
)

cd "${REPO_ROOT}"

echo "Queue log: ${QUEUE_LOG}" | tee -a "${QUEUE_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${QUEUE_LOG}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" | tee -a "${QUEUE_LOG}"
echo "SEED=${SEED} TRAINER_PRECISION=${TRAINER_PRECISION} DALES_XY_TILING=${DALES_XY_TILING} MAX_EPOCHS=${MAX_EPOCHS}" | tee -a "${QUEUE_LOG}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  NAME="${EXPERIMENT#semantic/}"
  echo "" | tee -a "${QUEUE_LOG}"
  echo "[$(date +%F' '%T)] Starting ${EXPERIMENT}" | tee -a "${QUEUE_LOG}"
  bash "${REPO_ROOT}/scripts/run_bsr_experiment.sh" \
    "${EXPERIMENT}" \
    "seed=${SEED}" \
    "trainer.precision=${TRAINER_PRECISION}" \
    "datamodule.xy_tiling=${DALES_XY_TILING}" \
    "trainer.max_epochs=${MAX_EPOCHS}" \
    "logger.wandb.name=${NAME}-24G-seed${SEED}" \
    2>&1 | tee -a "${QUEUE_LOG}"
  echo "[$(date +%F' '%T)] Finished ${EXPERIMENT}" | tee -a "${QUEUE_LOG}"
done

echo "[$(date +%F' '%T)] All DALES-BSR ablation runs finished." | tee -a "${QUEUE_LOG}"
