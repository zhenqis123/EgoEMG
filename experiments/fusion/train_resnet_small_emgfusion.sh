#!/bin/bash
set -euo pipefail

# ResNet18 + EMGFormer Small Fusion
# Frozen ResNet18 vision backbone + frozen EMGFormer small → trainable fusion layers

CONFIG="experiment=fusion/vision_resnet_small_emgfusion"

EGOEMG_MEMMAP="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap"
CROPS_LMDB="data/EgoEMG_v2_crops"

RESNET_CKPT="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/fusion/vision_resnet/version_7/checkpoints/last.ckpt"

python -m egoemg.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  vision_resnet_checkpoint="${RESNET_CKPT}" \
  train=true \
  eval=true
