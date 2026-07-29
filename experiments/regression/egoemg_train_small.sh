#!/usr/bin/env bash
# Small EMGFormer.
# Train: EgoEMG train split only.
# Val: EgoEMG user/gesture/both splits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

conda activate emg2pose_env

BASE_CONFIG="emgformer/regression_egoemg_train_incre_small"
GPUS="0,1,2,3,4,5"
WL=12000
BS="${BS:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR=0.0005
EPOCHS=150
SEED=42
export EMG2POSE_DEBUG_STEPS="${EMG2POSE_DEBUG_STEPS:-0}"
BASE_LOG="logs/regression/egoemg_only_small"
EGOEMG_DATA="${PROJECT_DIR}/data/EgoEMG_memmap"

echo "=== EgoEMG-only small training ==="
echo "Config: ${BASE_CONFIG}"
echo "Train:  EgoEMG [train] left/right"
echo "Val:    EgoEMG [user, gesture, both] left/right"
echo "Metric: val_user_mae, val_stage_mae, val_user_stage_mae"
echo "Logs:   ${BASE_LOG}"
echo "Debug:  EMG2POSE_DEBUG_STEPS=${EMG2POSE_DEBUG_STEPS}"
echo "Workers:${NUM_WORKERS}"
echo ""

mkdir -p "${BASE_LOG}"

CMD=(
python -m emg2pose.train
  experiment=${BASE_CONFIG} \
  egoemg_memmap_dir=${EGOEMG_DATA} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  hydra.run.dir=${BASE_LOG} \
  datamodule.window_length=${WL} \
  datamodule.val_test_window_length=${WL} \
  datamodule.stride=$((WL/10)) \
  datamodule.val_test_stride=${WL} \
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
