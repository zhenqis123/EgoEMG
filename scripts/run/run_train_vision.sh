#!/usr/bin/env bash
# One-click WiLoR fine-tuning on EgoEMG vision data.
#
# Usage:
#   bash scripts/run/run_train_vision.sh
#   DEVICES="[0,1]" bash scripts/run/run_train_vision.sh
#   EXPERIMENT=vision/mano_simple_freeze MAX_EPOCHS=50 bash scripts/run/run_train_vision.sh

set -euo pipefail

# Fix libffi/libstdc++ conflicts between conda base and system libraries.
export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7:${CONDA_PREFIX:-$HOME/miniconda3/envs/emg2pose_env}/lib/libstdc++.so.6"

EXPERIMENT="${EXPERIMENT:-vision/mano_simple_freeze}"
DEVICES="${DEVICES:-[1,2,3,4,5]}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-960}"
LR="${LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"

MEMMAP_DIR="${MEMMAP_DIR:-/mnt/nvme/xiziheng/EgoEMG_v2_memmap}"
VIDEO_ROOT="${VIDEO_ROOT:-/home/xiziheng/develop/emg2pose/data/EgoEMG}"
ALLINTRA_ROOT="${ALLINTRA_ROOT:-/mnt/nvme/xiziheng/EgoEMG_allintra}"
CROPS_DIR="${CROPS_DIR:-/mnt/nvme/xiziheng/EgoEMG_v2_crops}"
VISION_INDEX_DIR="${VISION_INDEX_DIR:-/mnt/nvme/xiziheng/EgoEMG_v2_memmap/vision_index}"
MANO_MODEL_PATH="${MANO_MODEL_PATH:-/home/xiziheng/develop/WiLoR/mano_data}"
WILOR_CHECKPOINT="${WILOR_CHECKPOINT:-/home/xiziheng/develop/WiLoR/pretrained_models/wilor_final.ckpt}"
CALIBRATION_PATH="${CALIBRATION_PATH:-/home/xiziheng/develop/emg2pose/data/EgoEMG/reprojection_assets/GX010023_standard_calibration.json}"

echo "=============================================="
echo "WiLoR EgoEMG Fine-tuning"
echo "=============================================="
echo "Experiment:    ${EXPERIMENT}"
echo "Devices:       ${DEVICES}"
echo "Max epochs:    ${MAX_EPOCHS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Num workers:   ${NUM_WORKERS}"
echo "Memmap dir:    ${MEMMAP_DIR}"
echo "Crops dir:     ${CROPS_DIR}"
echo "=============================================="

python -m emg2pose.train_vision \
    +experiment="${EXPERIMENT}" \
    train=True \
    eval=True \
    max_epochs="${MAX_EPOCHS}" \
    devices="${DEVICES}" \
    batch_size="${BATCH_SIZE}" \
    optimizer.lr="${LR}" \
    num_workers="${NUM_WORKERS}" \
    data_location="${MEMMAP_DIR}" \
    video_root="${VIDEO_ROOT}" \
    allintra_root="${ALLINTRA_ROOT}" \
    per_episode_crops_dir="${CROPS_DIR}" \
    vision_index_dir="${VISION_INDEX_DIR}" \
    mano_model_path="${MANO_MODEL_PATH}" \
    wilor_checkpoint_path="${WILOR_CHECKPOINT}" \
    calibration_path="${CALIBRATION_PATH}"
