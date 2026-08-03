#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${REPO}/logs/20260728/wilor_s_simple_frozen_exactvision_augbest_lr1e-5_eta5e-6_30e

conda activate emg2pose_env
cd "$REPO"

exec python -m egoemg.train \
  experiment=fusion/fusion_wilor_s_simple_frozen_augbest_30e \
  batch_size=400 \
  val_batch_size=400 \
  train=true \
  eval=false \
  trainer.devices='[0,1,2,3,4,5]' \
  trainer.max_epochs=30 \
  optimizer.lr=1.0e-5 \
  lr_scheduler.scheduler.T_max=30 \
  lr_scheduler.scheduler.eta_min=5.0e-6 \
  callbacks.0.save_top_k=1 \
  resume_ckpt="${RUN_ROOT}/train/version_0/checkpoints/last-v1.ckpt" \
  hydra.run.dir="${RUN_ROOT}/hydra" \
  logger.save_dir="$RUN_ROOT" \
  logger.name=train \
  logger.version=0
