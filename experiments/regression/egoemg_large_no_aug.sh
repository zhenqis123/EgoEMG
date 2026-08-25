#!/bin/bash
# EgoEMG Large 无增强回归训练
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_egoemg"
GPUS="0,1,2,3,4,5"
WL=7790
BS=300
LR=0.0003
EPOCHS=150
SEED=42
LOG_DIR="logs/regression/egoemg_large_no_aug"

python -m egoemg.train \
  experiment=${EXPERIMENT} \
  egoemg_unified_memmap_dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  seed=${SEED} \
  hydra.run.dir=${LOG_DIR} \
  "override /module/decoder=transformer/preset/large" \
  "override /transforms=emgformer_regression_aug_best" \
  datamodule.window_length=${WL} \
  datamodule.val_test_window_length=${WL} \
  datamodule.stride=$((WL/10)) \
  datamodule.val_test_stride=${WL} \
  batch_size=${BS} \
  optimizer.lr=${LR}
