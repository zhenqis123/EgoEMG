#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_NAME=fusion_rn50_s_egoemg_only_noaug_wl12000_lr1e-5_50e
OUTPUT_ROOT=${REPO}/logs/20260727/${RUN_NAME}/train

cd "${REPO}"
mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m egoemg.train \
  experiment=fusion/fusion_rn50_s_egoemg_only_noaug_wl12000_lr1e-5_50e \
  batch_size=200 val_batch_size=200 \
  train=true eval=false \
  'trainer.devices=[0,1,2,3,4,5]' trainer.max_epochs=50 \
  "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
  "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
  2>&1 | tee "${OUTPUT_ROOT}/console.log"
