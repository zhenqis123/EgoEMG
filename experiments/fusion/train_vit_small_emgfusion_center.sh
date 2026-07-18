#!/bin/bash
set -euo pipefail

# ViT-Small (DINOv2) + EMGFormer Small Fusion — Center-Supervised
# Full EMG temporal window → attention-pooled → fused with vision → center frame only
# Trainable backbones (vision + EMG), trainable fusion layers.
# Rotation-only augmentation.

CONFIG="experiment=fusion/vision_vit_small_emgfusion_center"

EGOEMG_MEMMAP="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap"
CROPS_LMDB="/mnt/nvme/xiziheng/EgoEMG_v2_crops"

BEST_VISION_VIT="/home/xiziheng/develop/emg2pose/logs/fusion/vision_vit_small/version_0/checkpoints/vit-small-epoch=179-val_mae=0.1053.ckpt"
BEST_FUSION_VIT="/home/xiziheng/develop/emg2pose/logs/fusion/vit_small_emgfusion_center/version_7/checkpoints/vit-small-centerfusion-epoch=088-val_mae=0.0968.ckpt"

python -m emg2pose.train \
  ${CONFIG} \
  egoemg_memmap_dir="${EGOEMG_MEMMAP}" \
  per_episode_crops_dir="${CROPS_LMDB}" \
  vision_vit_checkpoint="'${BEST_VISION_VIT}'" \
  checkpoint="'${BEST_FUSION_VIT}'" \
  transforms=rotation_augmentation \
  optimizer.lr=0.0001 \
  lr_scheduler.scheduler.eta_min=0.00001 \
  trainer.max_epochs=100 \
  train=true \
  eval=true
