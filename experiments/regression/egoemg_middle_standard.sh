#!/bin/bash
# EgoEMG Middle 标准回归训练
set -e
cd /home/xiziheng/develop/emg2pose

EXPERIMENT="emgformer/regression_egoemg"
GPUS="0,1,2,3,4,5"
WL=7790
BS=500
LR=0.0005
EPOCHS=150
SEED=42
LOG_DIR="logs/regression/egoemg_middle_standard"

python -m emg2pose.train \
  experiment=${EXPERIMENT} \
  egoemg_memmap_dir=/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap \
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
  optimizer.lr=${LR}
