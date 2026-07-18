#!/bin/bash
set -euo pipefail

# ViT-Huge/Giant (DINOv2) Vision-Only Baseline
# Pre-cropped 256×256 images → ViT-G/14 → 22 joint angles
# Center-frame supervision: one image predicts one pose.
# DINOv2 has no "huge" tier; uses vit_giant_patch14_dinov2 (1536-dim).

CONFIG="experiment=fusion/vision_vit_huge"

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
