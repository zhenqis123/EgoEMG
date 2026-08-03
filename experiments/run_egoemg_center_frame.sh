#!/bin/bash
set -euo pipefail

# Center-frame only supervision on EgoEMG — three EMG-only sizes + two fusion.
# Full EMG window for temporal context, loss computed on center frame only.
# EMG models use temporal attention pooling to aggregate decoder output.
# Rotation-only augmentation (no channel/time/freq masking, no noise).

EGOEMG_MEMMAP="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap"

# Best full-window pretrained checkpoints for weight initialization
BEST_SMALL="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/2026-05-05/01-57-33_emg2pose/regression_emgformer_small_aggressive_egoemg/version_0/checkpoints/egoemg-small-epoch=259-val_mae=0.2551.ckpt"
BEST_MIDDLE="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/2026-04-29/11-06-44_emg2pose/regression_emgformer_middle_aggressive_egoemg/version_0/checkpoints/egoemg-epoch=107-val_mae=0.2623.ckpt"
BEST_LARGE="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/2026-05-05/05-24-34_emg2pose/regression_emgformer_large_aggressive_egoemg/version_0/checkpoints/egoemg-large-epoch=188-val_mae=0.2618.ckpt"

COMMON_OPTS=(
  egoemg_memmap_dir="${EGOEMG_MEMMAP}"
  transforms=rotation_augmentation
  +center_target_only=true
  +module.center_supervised=true
  optimizer.lr=0.0001
  lr_scheduler.scheduler.eta_min=0.00001
  train=true
  eval=true
)

run_size() {
  local size=$1
  local devices=$2
  local batch_size=$3
  local pretrained_ckpt=$4
  local name="regression_emgformer_${size}_aggressive_egoemg_center"

  echo "============================================"
  echo "  Running: ${name}"
  echo "  Devices: ${devices}"
  echo "  Epochs:  100"
  echo "  Batch:   ${batch_size}"
  echo "  Pretrained: ${pretrained_ckpt}"
  echo "============================================"

  python -m egoemg.train \
    experiment="emgformer/regression_emgformer_${size}_aggressive_egoemg" \
    "${COMMON_OPTS[@]}" \
    trainer.devices="${devices}" \
    trainer.max_epochs=100 \
    batch_size="${batch_size}" \
    logger.name="${name}" \
    pretrained_checkpoint="'${pretrained_ckpt}'"

  echo "Done: ${name}"
  echo ""
}

# ── EMG-only center-frame (small / middle / large) ──────────────
run_size "small"   "[1,2,3,4,5]"  500  "${BEST_SMALL}"
run_size "middle"  "[1,2,3,4,5]"  500  "${BEST_MIDDLE}"
run_size "large"   "[1,2,3,4,5]"  400  "${BEST_LARGE}"

# ── Fusion center-frame (ResNet + ViT) ──────────────────────────
echo "============================================"
echo "  Running: resnet_small_emgfusion_center"
echo "============================================"
bash experiments/fusion/train_resnet_small_emgfusion_center.sh

echo "============================================"
echo "  Running: vit_small_emgfusion_center"
echo "============================================"
bash experiments/fusion/train_vit_small_emgfusion_center.sh

# ── WiLoR ViT Vision-Only from Scratch ────────────────────────────
echo "============================================"
echo "  Running: vision_wilor_vit_scratch"
echo "============================================"
python -m egoemg.train \
  experiment=fusion/vision_wilor_vit_scratch \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="data/EgoEMG_v2_crops" \
  optimizer.lr=0.00005 \
  lr_scheduler.scheduler.eta_min=0.000005 \
  trainer.max_epochs=100 \
  trainer.devices="[1,2,3,4,5]" \
  train=true \
  eval=true
