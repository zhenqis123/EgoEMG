#!/usr/bin/env bash
# Fresh EgoEMG-only RN18 + EMGFormer-S fusion experiment on all local GPUs.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/fusion_rn18_s_egoemg_only_augbest_no_mixup_dense_lr5e-4_300e_20260724
RUN_DIR="${RUN_ROOT}/train"

mkdir -p "${RUN_DIR}"
cd "${REPO}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python -m egoemg.train \
  experiment=fusion/fusion_rn18_s_egoemg_only_augbest_pretrained \
  batch_size=200 \
  val_batch_size=200 \
  train=true eval=false \
  trainer.devices='[0,1,2,3,4,5]' \
  trainer.max_epochs=300 \
  hydra.run.dir="${RUN_DIR}/hydra" \
  logger.save_dir="${RUN_DIR}" logger.name=train logger.version=0 \
  2>&1 | tee "${RUN_DIR}/console.log"
