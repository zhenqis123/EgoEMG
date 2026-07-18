#!/usr/bin/env bash
set -euo pipefail

cd /home/xiziheng/develop/emg2pose

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/xiziheng/miniconda3/envs/emg2pose_env/bin/python -u \
  scripts/hparam/optuna_aug_search_wl12000.py \
  --gpus 0,1,2,3,4,5 \
  --n-trials 50 \
  --objective-metric val_user_mae
