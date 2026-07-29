#!/usr/bin/env bash
# Fresh full-data RN18 + random EMGFormer-S fusion control, no augmentation.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721
RUN_NAME=fresh_imagenet_emgscratch_noaug_lr1e-4_eta5e-6_150e
OUTPUT_ROOT=${RUN_ROOT}/fusion_rn18_s/${RUN_NAME}

VISIBLE_GPUS=1,2,3,4,5
TRAINER_GPUS='[0,1,2,3,4]'
BATCH_SIZE=200
MAX_EPOCHS=150
BASE_LR=1e-4
ETA_MIN=5e-6

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"

CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
    python -m emg2pose.train \
    experiment=fusion/fusion_rn18_s_imagenet_emgscratch_noaug \
    "batch_size=${BATCH_SIZE}" "val_batch_size=${BATCH_SIZE}" \
    "optimizer.lr=${BASE_LR}" \
    "lr_scheduler.scheduler.T_max=${MAX_EPOCHS}" \
    "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    train=true eval=false \
    "trainer.devices=${TRAINER_GPUS}" "trainer.max_epochs=${MAX_EPOCHS}" \
    "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
    "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
    2>&1 | tee "${OUTPUT_ROOT}/console.log"
