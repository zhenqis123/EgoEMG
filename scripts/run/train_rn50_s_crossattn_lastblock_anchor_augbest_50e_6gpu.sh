#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260727
RUN_NAME=fusion_rn50_s_egoemg_only_wl12000_crossattn_lastblock_anchor_augbest_50e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

conda activate emg2pose_env
mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m emg2pose.train \
  experiment=fusion/fusion_rn50_s_egoemg_only_wl12000_crossattn_lastblock_anchor_augbest_50e \
  batch_size=200 val_batch_size=200 \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=50 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  "$@" 2>&1 | tee -a "${RUN_DIR}/console.log"
