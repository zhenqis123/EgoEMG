#!/bin/bash
set -euo pipefail

# ResNet50 Vision-Only Baseline
# Pre-cropped 256×256 images → ResNet50 → 22 joint angles
# Center-frame supervision: one image predicts one pose.

CONFIG="experiment=fusion/vision_resnet50"

# ── Data paths (override if needed) ────────────────────────────────────────────
EGOEMG_MEMMAP="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap"
CROPS_LMDB="data/EgoEMG_v2_crops"

# ── Run training ───────────────────────────────────────────────────────────────
python -m egoemg.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  train=true \
  eval=true
