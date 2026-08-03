#!/bin/bash
# emg2pose_v3 Middle 回归训练
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_emg2pose"
GPUS="0,1,2,3,4,5"
BS=600
LR=0.0001
EPOCHS=250
SEED=42
LOG_DIR="logs/regression/emg2pose_middle"

python -m egoemg.train \
  experiment=${EXPERIMENT} \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  seed=${SEED} \
  hydra.run.dir=${LOG_DIR} \
  batch_size=${BS} \
  optimizer.lr=${LR}
