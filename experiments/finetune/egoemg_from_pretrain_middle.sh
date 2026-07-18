#!/bin/bash
# 从预训练 checkpoint finetune 到 EgoEMG (Middle)
set -e
cd /home/xiziheng/develop/emg2pose

EXPERIMENT="emgformer/finetune_egoemg"
GPUS="0,1,2,3,4,5"
BS=700
LR=0.00001
EPOCHS=50
SEED=42
LOG_DIR="logs/finetune/egoemg_from_pretrain_middle"

# 预训练 checkpoint 路径
CHECKPOINT="logs/pretrain/latest/checkpoints/last.ckpt"

python -m emg2pose.train \
  experiment=${EXPERIMENT} \
  egoemg_memmap_dir=/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap \
  trainer.devices=[${GPUS}] \
  +trainer.strategy=ddp \
  trainer.max_epochs=${EPOCHS} \
  seed=${SEED} \
  hydra.run.dir=${LOG_DIR} \
  'checkpoint="'"${CHECKPOINT}"'"' \
  batch_size=${BS} \
  optimizer.lr=${LR}
