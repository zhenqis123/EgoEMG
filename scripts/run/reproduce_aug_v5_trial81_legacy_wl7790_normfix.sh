#!/usr/bin/env bash
set -euo pipefail

cd ${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python -u -m egoemg.train \
  --config-name reproduce_aug_v5_trial81_legacy_wl7790_normfix \
  hydra.run.dir=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/reproduce_aug_v5_trial81_legacy_wl7790_normfix/run_2026-06-19_bs500
