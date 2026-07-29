#!/usr/bin/env bash
# RN50 + EMGFormer-M, controlled 50-epoch EgoEMG-only no-augmentation run.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260725
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'
RUN_NAME=fusion_rn50_m_egoemg_only_noaug_wl12000_25e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"

mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m emg2pose.train \
  experiment=fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_25e \
  batch_size=220 val_batch_size=220 \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=25 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  2>&1 | tee -a "${RUN_DIR}/console.log"
