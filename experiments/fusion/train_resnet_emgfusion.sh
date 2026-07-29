#!/bin/bash
set -euo pipefail

# ResNet18 + EMGFormer Small Fusion
# Frozen ResNet18 vision backbone + frozen EMGFormer small → trainable fusion layers
# Stage 2: vision_proj, fusion_proj, head, head_vision, gate_proj trained from scratch

CONFIG="experiment=fusion/vision_resnet_emgfusion"

# ── Data paths (override if needed) ────────────────────────────────────────────
EGOEMG_MEMMAP="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap"
CROPS_LMDB="data/EgoEMG_v2_crops"

# ── Run training ───────────────────────────────────────────────────────────────
python -m emg2pose.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  train=true \
  eval=true
