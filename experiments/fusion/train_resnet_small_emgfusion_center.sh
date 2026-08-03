#!/bin/bash
set -euo pipefail

# ResNet18 + EMGFormer Small Fusion — Center-Supervised
# Full EMG temporal window → attention-pooled → fused with vision → center frame only
# Trainable backbones (vision + EMG), trainable fusion layers.
# Rotation-only augmentation.

CONFIG="experiment=fusion/vision_resnet_small_emgfusion_center"

EGOEMG_MEMMAP="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap"
CROPS_LMDB="data/EgoEMG_v2_crops"

BEST_VISION_RESNET="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/fusion/vision_resnet/version_9/checkpoints/resnet-vision-epoch=011-val_mae=0.1022.ckpt"
BEST_FUSION_RESNET="${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/logs/fusion/resnet_small_emgfusion_center/version_14/checkpoints/resnet-small-centerfusion-epoch=137-val_mae=0.0945.ckpt"

python -m egoemg.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  vision_resnet_checkpoint="'${BEST_VISION_RESNET}'" \
  checkpoint="'${BEST_FUSION_RESNET}'" \
  transforms=rotation_augmentation \
  optimizer.lr=0.0001 \
  lr_scheduler.scheduler.eta_min=0.00001 \
  trainer.max_epochs=100 \
  train=true \
  eval=true
