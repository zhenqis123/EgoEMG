#!/bin/bash
set -euo pipefail

# ViT Small + EMGFormer Small Fusion
# Cached ViT features (1280-dim) → 128-dim projection + frozen EMGFormer small → trainable fusion layers

CONFIG="experiment=fusion/vision_vit_small_emgfusion"

EGOEMG_MEMMAP="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
VIT_FEATURES="/mnt/nvme/xiziheng/EgoEMG_v2_vit_features_lmdb"

python -m emg2pose.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  cached_vit_features_dir="${VIT_FEATURES}" \
  train=true \
  eval=true
