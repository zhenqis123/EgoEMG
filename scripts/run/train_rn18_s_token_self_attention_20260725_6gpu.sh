#!/usr/bin/env bash
# RN18 + EMGFormer-S direct joint-token pose decoder, 150-epoch continuation.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260725
RUN_NAME=fusion_rn18_s_token_self_attention_direct_pose_continue_lr5e-5_eta5e-6_150e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'
CONTINUE_CKPT=${REPO}/logs/20260725/fusion_rn18_s_token_self_attention_direct_pose_egoemg_only_noaug_wl12000_50e/train/train/version_0/checkpoints/best-val_mae=val_mae=0.1016.ckpt

mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
  experiment=fusion/fusion_rn18_s_token_self_attention_egoemg_only \
  batch_size=360 val_batch_size=360 \
  checkpoint="\"${CONTINUE_CKPT}\"" \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=150 \
  optimizer.lr=5e-5 lr_scheduler.scheduler.T_max=150 \
  lr_scheduler.scheduler.eta_min=5e-6 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  "$@" \
  2>&1 | tee -a "${RUN_DIR}/console.log"
