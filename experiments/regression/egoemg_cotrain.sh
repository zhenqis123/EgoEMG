#!/bin/bash
# Co-training: EgoEMG + emg2pose (8ch aligned)
# Both datasets use per-dataset normalization.
# EgoEMG: target_hand 8ch (left + right)
# emg2pose: 16ch → 8ch via channel_indices [10,12,0,1,2,4,5,6]
# Val: EgoEMG only (user/gesture/both generalization splits)
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

BASE_CONFIG="emgformer/regression_egoemg_cotrain"
GPUS="0,1,2,3,4,5"
WL=12000
BS=300
LR=0.0005
EPOCHS=150
SEED=42
BASE_LOG="logs/regression/egoemg_cotrain"
DATA_DIR="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap"
EMG2POSE_DATA="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/emg2pose_memmap"

echo "=== Co-Training: EgoEMG + emg2pose (8ch aligned) ==="
echo "Config: ${BASE_CONFIG}"
echo "Train: EgoEMG [train] (left+right, target_hand 8ch) + emg2pose [train] (16ch→8ch)"
echo "Val:   EgoEMG [user, gesture, both] splits"
echo "Norm:  per-dataset (egoemg + emg2pose_8ch_aligned)"
echo "Logs:  ${BASE_LOG}"
echo ""

mkdir -p "${BASE_LOG}"

python -m egoemg.train \
  experiment=${BASE_CONFIG} \
  egoemg_unified_memmap_dir=${DATA_DIR} \
  emg2pose_memmap_dir=${EMG2POSE_DATA} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  hydra.run.dir=${BASE_LOG} \
  batch_size=${BS} \
  optimizer.lr=${LR} \
  seed=${SEED} \
  'module.featurizer.conv_blocks.0.in_channels=8' \
  2>&1 | tee ${BASE_LOG}/console.log
