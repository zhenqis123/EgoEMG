#!/bin/bash
# Clean EgoEMG-only training with per-split val MAE reporting
# Train: EgoEMG train split (left + right hands)
# Val:   EgoEMG user, gesture, both splits → val_stage_mae, val_user_mae, val_user_stage_mae
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BASE_CONFIG="emgformer/regression_egoemg_clean"
GPUS="0,1,2,3,4,5"
WL=12000
BS=300
LR=0.0005
EPOCHS=150
SEED=42
BASE_LOG="logs/regression/egoemg_clean"
DATA_DIR="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap"

# No augmentation — clean baseline
AUG="batch_augmentation=null"

echo "=== Clean EgoEMG Training ==="
echo "Config: ${BASE_CONFIG}"
echo "Train: EgoEMG [train] split (left+right hands, eps 29/30 excluded on right)"
echo "Val:   EgoEMG [user, gesture, both] splits"
echo "Aug:   NONE (clean baseline)"
echo "Logs:  ${BASE_LOG}"
echo ""

mkdir -p "${BASE_LOG}"

python -m egoemg.train \
  experiment=${BASE_CONFIG} \
  egoemg_unified_memmap_dir=${DATA_DIR} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  hydra.run.dir=${BASE_LOG} \
  datamodule.window_length=${WL} \
  datamodule.val_test_window_length=${WL} \
  datamodule.stride=$((WL/10)) \
  datamodule.val_test_stride=${WL} \
  batch_size=${BS} \
  optimizer.lr=${LR} \
  seed=${SEED} \
  ${AUG} \
  'module.featurizer.conv_blocks.0.in_channels=8' \
  2>&1 | tee ${BASE_LOG}/console.log
