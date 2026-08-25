#!/usr/bin/env bash
# ViT-B + EMGFormer-S simple residual fusion. The pretrained visual predictor
# remains frozen; the EMG branch and residual pathway are trained from their
# configured initializations on six GPUs.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=python
RUN_ROOT=${RUN_ROOT:-${REPO}/logs/20260728/vitb_s_simple_frozen_lr5e-4_eta5e-6_30e_stride400_repeat1}

cd "${REPO}"
mkdir -p "${RUN_ROOT}"

export EGOEMG_ROOT="${REPO}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" -m egoemg.train \
  experiment=fusion/fusion_vitb_s_simple_frozen_augbest_30e \
  train=true eval=false \
  trainer.devices='[0,1,2,3,4,5]' \
  trainer.max_epochs=30 \
  batch_size=480 val_batch_size=480 \
  datamodule.stride=400 \
  datamodule.dataset_repeat=1 \
  optimizer.lr=5.0e-4 \
  lr_scheduler.scheduler.T_max=30 \
  lr_scheduler.scheduler.eta_min=5.0e-6 \
  callbacks.0.save_top_k=1 \
  callbacks.0.save_last=false \
  "hydra.run.dir=${RUN_ROOT}/hydra" \
  "logger.save_dir=${RUN_ROOT}" \
  logger.name=train \
  logger.version=0 \
  2>&1 | tee -a "${RUN_ROOT}/console.log"
