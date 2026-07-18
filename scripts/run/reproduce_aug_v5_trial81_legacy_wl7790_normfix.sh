#!/usr/bin/env bash
set -euo pipefail

cd /home/xiziheng/develop/emg2pose

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/xiziheng/miniconda3/envs/emg2pose_env/bin/python -u -m emg2pose.train \
  --config-name reproduce_aug_v5_trial81_legacy_wl7790_normfix \
  hydra.run.dir=/home/xiziheng/develop/emg2pose/logs/reproduce_aug_v5_trial81_legacy_wl7790_normfix/run_2026-06-19_bs500
