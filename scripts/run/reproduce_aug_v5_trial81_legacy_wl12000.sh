#!/usr/bin/env bash
set -euo pipefail

cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u -m egoemg.train \
  --config-name reproduce_aug_v5_trial81_legacy_wl12000 \
  hydra.run.dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/reproduce_aug_v5_trial81_legacy_wl12000/run_2026-06-17_bs250_acc2
