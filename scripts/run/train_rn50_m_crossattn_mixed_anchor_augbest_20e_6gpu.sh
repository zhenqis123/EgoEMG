#!/usr/bin/env bash
# Controlled hard-negative anchor experiment: 50% zero EMG, 50% same-hand
# mismatched EMG in the auxiliary anchor forward.  All other settings match
# the best zero-anchor augbest 20-epoch experiment.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=${REPO}/logs/20260726
RUN_NAME=fusion_rn50_m_egoemg_only_wl12000_crossattn_mixed_anchor_augbest_20e
RUN_DIR="${LOG_ROOT}/${RUN_NAME}/train"
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

conda activate egoemg_env

if [[ -e "${RUN_DIR}/train/version_0" ]]; then
  echo "Refusing to overwrite existing run: ${RUN_DIR}/train/version_0" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
  experiment=fusion/fusion_rn50_m_egoemg_only_wl12000_crossattn_mixed_anchor_augbest_20e \
  batch_size=200 val_batch_size=200 \
  train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=20 \
  "hydra.run.dir=${RUN_DIR}/hydra" \
  "logger.save_dir=${RUN_DIR}" logger.name=train logger.version=0 \
  "$@" \
  2>&1 | tee -a "${RUN_DIR}/console.log"
