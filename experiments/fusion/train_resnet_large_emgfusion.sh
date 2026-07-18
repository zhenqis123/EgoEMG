#!/bin/bash
set -euo pipefail

# ResNet50 + EMGFormer Small Fusion
# Frozen ResNet50 vision backbone + frozen EMGFormer small → trainable fusion layers

CONFIG="experiment=fusion/vision_resnet_large_emgfusion"

EGOEMG_MEMMAP="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
CROPS_LMDB="/mnt/nvme/xiziheng/EgoEMG_v2_crops"

python -m emg2pose.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  train=true \
  eval=true
