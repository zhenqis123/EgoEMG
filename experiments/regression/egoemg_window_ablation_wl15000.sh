#!/usr/bin/env bash
# EgoEMG window-ablation WL=15000 — same recipe as wl14638, only WL differs.
# Reference config:
#   config/experiment/emgformer/regression_egoemg_window_ablation_wl14638.yaml
#
# All other knobs (featurizer, decoder, head, batch_size, loss_weights,
# batch_augmentation, transforms) match the historical trial_0006 recipe.
# We only change window_length, stride, and val_test_stride via CLI.
#
# Layout: target_hand (raw 8 channels per hand, no 16-ch sparse-place).
# This is the "no channel interpolation" variant.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

conda activate emg2pose_env

BASE_CONFIG="emgformer/regression_egoemg_window_ablation_wl14638"
GPUS="${GPUS:-0,1,2,3,4,5}"
WL=15000
STRIDE="${STRIDE:-1500}"                # 1/10 of WL
VAL_TEST_STRIDE="${VAL_TEST_STRIDE:-15000}"  # non-overlapping val windows
EMG_FIELD="${EMG_FIELD:-filtered}"
BS="${BS:-194}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LR="${LR:-0.0005}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-6}"
BASE_LOG="logs/regression/egoemg_window_ablation_wl${WL}_${EMG_FIELD}_8ch"
EGOEMG_DATA="${EGOEMG_DATA:-data/EgoEMG_unified_memmap}"

echo "=== EgoEMG window-ablation WL=${WL} (aligned recipe, 8ch target_hand) ==="
echo "Base config: ${BASE_CONFIG}"
echo "Layout: target_hand (raw 8ch, no interpolation)"
echo "WL/stride/val_stride: ${WL}/${STRIDE}/${VAL_TEST_STRIDE}"
echo "EMG: ${EMG_FIELD}"
echo "Memmap: ${EGOEMG_DATA}"
echo "Batch size: ${BS}"
echo "Seed: ${SEED}"
echo "Logs: ${BASE_LOG}"
echo ""

mkdir -p "${BASE_LOG}"

CMD=(
python -m egoemg.train
  experiment=${BASE_CONFIG} \
  egoemg_unified_memmap_dir=${EGOEMG_DATA} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  hydra.run.dir=${BASE_LOG} \
  datamodule.window_length=${WL} \
  datamodule.val_test_window_length=${WL} \
  datamodule.stride=${STRIDE} \
  datamodule.val_test_stride=${VAL_TEST_STRIDE} \
  dataset.train.0.dataset_name=egoemg_8ch \
  dataset.train.1.dataset_name=egoemg_8ch \
  dataset.val.0.dataset_name=egoemg_8ch \
  dataset.val.1.dataset_name=egoemg_8ch \
  dataset.test.0.dataset_name=egoemg_8ch \
  dataset.test.1.dataset_name=egoemg_8ch \
  batch_size=${BS} \
  num_workers=${NUM_WORKERS} \
  optimizer.lr=${LR} \
  seed=${SEED}
)

if [[ -t 1 && "${LOG_TO_FILE:-0}" != "1" ]]; then
  "${CMD[@]}"
else
  "${CMD[@]}" 2>&1 | tee "${BASE_LOG}/console.log"
fi
