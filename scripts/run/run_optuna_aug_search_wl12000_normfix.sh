#!/usr/bin/env bash
set -euo pipefail

cd ${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u \
  scripts/hparam/optuna_aug_search_wl12000.py \
  --gpus 0,1,2,3,4,5 \
  --n-trials 50 \
  --objective-metric val_user_mae
