#!/usr/bin/env bash
# RN50 final Bottleneck + zero-EMG anchor loss; BN/head locked; 20-epoch pilot.
# Same as lastblock_20e but with anchor_loss_weight=1.0 to suppress the
# vision-only shortcut through the residual head.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260726
RUN_NAME=fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_20e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

# Activate the project conda env (tmux shells do not inherit it).
conda activate egoemg_env

mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
  experiment=fusion/fusion_rn50_m_egoemg_only_noaug_wl12000_crossattn_lastblock_anchor_20e \
  batch_size=300 val_batch_size=300 \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=20 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  "$@" \
  2>&1 | tee -a "${RUN_DIR}/console.log"

