#!/bin/bash
# EgoEMG Middle + Incre Manus 联合训练
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_egoemg_with_incre"
GPUS="0,1,2,3,4,5"
WL=12000
BS=300
LR=0.0005
EPOCHS=150
SEED=42
LOG_DIR="logs/regression/egoemg_middle_with_incre"

python -m egoemg.train \
  experiment=${EXPERIMENT} \
  egoemg_unified_memmap_dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_unified_memmap \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  seed=${SEED} \
  hydra.run.dir=${LOG_DIR} \
  datamodule.window_length=${WL} \
  datamodule.val_test_window_length=${WL} \
  datamodule.stride=$((WL/10)) \
  datamodule.val_test_stride=${WL} \
  batch_size=${BS} \
  optimizer.lr=${LR} \
  +trainer.val_check_interval=1
