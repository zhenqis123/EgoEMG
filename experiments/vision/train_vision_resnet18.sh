#!/bin/bash
set -euo pipefail

# ResNet18 Vision-Only Baseline
# Pre-cropped 256×256 images → ResNet18 → 22 joint angles
# Center-frame supervision: one image predicts one pose.

CONFIG="experiment=fusion/vision_resnet18"

# ── Data paths (override if needed) ────────────────────────────────────────────
EGOEMG_MEMMAP="${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap"
CROPS_LMDB="data/EgoEMG_crops"

# ── Run training ───────────────────────────────────────────────────────────────
python -m egoemg.train \
  ${CONFIG} \
  egoemg_unified_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  train=true \
  eval=true
