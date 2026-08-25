#!/usr/bin/env bash
# Frozen RN50 + multi-scale visual cross-attention + EMGFormer-M.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260725
RUN_NAME=fusion_rn50_m_egoemg_only_noaug_wl12000_frozen_crossattn_100e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
  experiment=fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_frozen_crossattn_100e \
  batch_size=400 val_batch_size=400 \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=100 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  "$@" \
  2>&1 | tee -a "${RUN_DIR}/console.log"
