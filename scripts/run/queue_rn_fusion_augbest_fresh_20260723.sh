#!/usr/bin/env bash
# Fresh RN fusion experiments with batch_aug_best_v2.
# Each model starts from its vision-only and EMGFormer-S checkpoints; no fusion
# checkpoint, optimizer state, or scheduler state is resumed.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721
INPUT_ROOT=data/experiment_inputs
VISION_ROOT="${INPUT_ROOT}/vision_checkpoints"
VISIBLE_GPUS=2,3,4,5
TRAINER_GPUS='[0,1,2,3]'
STAGE=fresh_augbest_lr5e-4_eta1e-6_250e
MAX_EPOCHS=250
BASE_LR=5e-4
ETA_MIN=1e-6

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

run_job() {
    local name=$1 backbone=$2 vision_dim=$3 vision_ckpt=$4 batch_size=$5
    local stage_dir="${RUN_ROOT}/${name}/${STAGE}"
    mkdir -p "$stage_dir"

    log "START ${name}: fresh fusion, ${MAX_EPOCHS} epochs, best batch augmentation"
    CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
        python -m emg2pose.train \
        experiment=fusion/fusion_allvision_s_egoemg_showee \
        +augmentation=batch_aug_best_v2 \
        "module.vision_backbone_type=${backbone}" \
        "module.vision_embed_dim=${vision_dim}" \
        "module.vision_pretrained_checkpoint=${vision_ckpt}" \
        "pretrained_emg_checkpoint=${INPUT_ROOT}/emgformer_s_full_showee.ckpt" \
        "batch_size=${batch_size}" \
        "val_batch_size=${batch_size}" \
        "optimizer.lr=${BASE_LR}" \
        "lr_scheduler.scheduler.T_max=${MAX_EPOCHS}" \
        "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
        train=true eval=false \
        "trainer.devices=${TRAINER_GPUS}" \
        "trainer.max_epochs=${MAX_EPOCHS}" \
        "hydra.run.dir=${stage_dir}/hydra" \
        "logger.save_dir=${stage_dir}" logger.name=train logger.version=0 \
        2>&1 | tee "${stage_dir}/console.log"
    log "DONE ${name}"
}

cd "$REPO"
run_job fusion_rn18_s resnet18 512 "${VISION_ROOT}/rn18.ckpt" 200
run_job fusion_rn50_s resnet50 2048 "${VISION_ROOT}/rn50.ckpt" 210
run_job fusion_rn152_s resnet152 2048 "${VISION_ROOT}/rn152.ckpt" 40
log 'ALL FRESH AUGBEST RN FUSION RUNS COMPLETED'
