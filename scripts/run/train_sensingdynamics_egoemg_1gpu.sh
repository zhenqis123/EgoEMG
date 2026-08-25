#!/usr/bin/env bash
# Train the SensingDynamics baseline on both hands of EgoEMG.
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d)}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/logs/${RUN_DATE}/sensingdynamics_egoemg}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

export EGOEMG_ROOT="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python -m egoemg.train \
  experiment=emg2pose/regression_sensingdynamics_egoemg \
  train=true eval=true \
  trainer.devices='[0]' \
  "batch_size=${BATCH_SIZE}" \
  "hydra.run.dir=${RUN_ROOT}/hydra" \
  "logger.save_dir=${RUN_ROOT}" \
  logger.name=train \
  logger.version=0 \
  2>&1 | tee -a "${RUN_ROOT}/console.log"
