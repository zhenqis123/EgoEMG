#!/bin/bash
set -euo pipefail

# WiLoR ViT Vision-Only Full Fine-tuning
# Pre-cropped 256×256 images → WiLoR ViT → 22 joint angles
# Center-frame supervision: one image predicts one pose.

CONFIG="experiment=fusion/vision_wilor_vit"

# ── Data paths (override if needed) ────────────────────────────────────────────
EGOEMG_MEMMAP="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
CROPS_LMDB="/mnt/nvme/xiziheng/EgoEMG_v2_crops"

# ── Run training ───────────────────────────────────────────────────────────────
python -m emg2pose.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  train=true \
  eval=true
