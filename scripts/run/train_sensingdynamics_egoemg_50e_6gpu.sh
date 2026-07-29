#!/usr/bin/env bash
# Full-speed SensingDynamics reproduction on EgoEMG (six RTX 4090 GPUs).
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/logs/20260728/sensingdynamics_egoemg_50e_lr1e-4_eta5e-6_bs320_6gpu}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

export EMG2POSE_ROOT="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m emg2pose.train \
  experiment=emg2pose/regression_sensingdynamics_egoemg \
  train=true eval=false \
  trainer.devices='[0,1,2,3,4,5]' \
  trainer.max_epochs=50 \
  trainer.precision=bf16-mixed \
  optimizer.lr=1.0e-4 \
  lr_scheduler.scheduler.T_max=50 \
  lr_scheduler.scheduler.eta_min=5.0e-6 \
  batch_size=320 \
  "hydra.run.dir=${RUN_ROOT}/hydra" \
  "logger.save_dir=${RUN_ROOT}" \
  logger.name=train \
  logger.version=0 \
  2>&1 | tee -a "${RUN_ROOT}/console.log"
