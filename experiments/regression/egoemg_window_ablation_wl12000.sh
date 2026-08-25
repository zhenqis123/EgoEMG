#!/usr/bin/env bash
# EMGFormer Middle EgoEMG window-length ablation reproduction.
# Matches ablation_study/window_length recipe, with WL set near 12k.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

conda activate emg2pose_env

BASE_CONFIG="emgformer/regression_egoemg_window_ablation_wl12000"
GPUS="${GPUS:-0,1,2,3,4,5}"
WL="${WL:-12000}"
EMG_FIELD="${EMG_FIELD:-filtered_paper}"
STRIDE="${STRIDE:-$((WL/10))}"
BS="${BS:-260}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LR="${LR:-0.0005}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-42}"
BASE_LOG="logs/regression/egoemg_window_ablation_wl${WL}_${EMG_FIELD}"
EGOEMG_DATA="${PROJECT_DIR}/data/EgoEMG_full_memmap"

echo "=== EgoEMG window-ablation recipe reproduction ==="
echo "Config: ${BASE_CONFIG}"
echo "WL/stride: ${WL}/${STRIDE}"
echo "Model: EMGFormer middle aggressive"
echo "EMG: ${EMG_FIELD}, target_hand 8ch, per-dataset norm"
echo "Aug: batch_aug.yaml + rotation"
echo "Train: EgoEMG [train] left/right"
echo "Val/Test: EgoEMG [user, gesture, both] left/right"
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
  datamodule.val_test_stride=${WL} \
  egoemg_emg_field_preference=${EMG_FIELD} \
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
