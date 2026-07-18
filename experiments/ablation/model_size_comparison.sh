#!/bin/bash
# 模型大小对比实验 (small/middle/large on EgoEMG)
set -e
cd /home/xiziheng/develop/emg2pose

EXPERIMENT="emgformer/regression_egoemg"
GPUS="0,1,2,3,4,5"
WL=7790
LR=0.0005
EPOCHS=150
SEED=42
LOG_BASE="logs/ablation/model_size"

declare -A SIZE_BS
SIZE_BS[small]=700
SIZE_BS[middle]=500
SIZE_BS[large]=300

for SIZE in small middle large; do
    BS=${SIZE_BS[$SIZE]}
    TRIAL_DIR="${LOG_BASE}/${SIZE}"

    echo "[$(date)] Size=${SIZE}, BS=${BS}"

    python -m emg2pose.train \
      experiment=${EXPERIMENT} \
      egoemg_memmap_dir=/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap \
      trainer.devices=[${GPUS}] \
      +trainer.strategy=ddp \
      trainer.max_epochs=${EPOCHS} \
      seed=${SEED} \
      hydra.run.dir=${TRIAL_DIR} \
      "override /module/decoder=transformer/preset/${SIZE}" \
      datamodule.window_length=${WL} \
      datamodule.val_test_window_length=${WL} \
      datamodule.stride=$((WL/10)) \
      datamodule.val_test_stride=${WL} \
      batch_size=${BS} \
      optimizer.lr=${LR}

    echo "[$(date)] ${SIZE} done. Waiting 30s..."
    sleep 30
done

echo "[$(date)] ALL SIZES COMPLETE"
