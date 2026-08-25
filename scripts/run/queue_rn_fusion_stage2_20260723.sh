#!/usr/bin/env bash
# Stage-3 RN fusion fine-tuning with the best GPU batch augmentation.
# Each run gets 150 additional epochs with a fresh cosine: 1e-5 -> 1e-6.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721
INPUT_ROOT=data/experiment_inputs
VISION_ROOT="${INPUT_ROOT}/vision_checkpoints"
VISIBLE_GPUS=2,3,4,5
TRAINER_GPUS='[0,1,2,3]'
STAGE=stage3_augbest_lr1e-5_eta1e-6_150e

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["epoch"]))
PY
}

prepare_stage2_checkpoint() {
    local source=$1
    local destination=$2
    python - "$source" "$destination" <<'PY'
import sys
from pathlib import Path

import torch

source, destination = map(Path, sys.argv[1:])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
base_lr, eta_min = 1e-5, 1e-6

for optimizer in checkpoint.get("optimizer_states", []):
    for group in optimizer.get("param_groups", []):
        group["initial_lr"] = base_lr
        group["lr"] = base_lr

for scheduler in checkpoint.get("lr_schedulers", []):
    scheduler["T_max"] = 150
    scheduler["eta_min"] = eta_min
    scheduler["base_lrs"] = [base_lr for _ in scheduler.get("base_lrs", [base_lr])]
    scheduler["last_epoch"] = 0
    scheduler["_step_count"] = 1
    scheduler["_last_lr"] = [base_lr for _ in scheduler["base_lrs"]]

torch.save(checkpoint, destination)
PY
}

run_job() {
    local name=$1 backbone=$2 vision_dim=$3 vision_ckpt=$4 batch_size=$5 best_ckpt=$6
    local source_epoch target_max stage_dir initial_ckpt
    source_epoch=$(checkpoint_epoch "$best_ckpt")
    target_max=$((source_epoch + 151))
    stage_dir="${RUN_ROOT}/${name}/${STAGE}"
    initial_ckpt="${stage_dir}/initial_from_epoch_${source_epoch}.ckpt"
    mkdir -p "$stage_dir"
    prepare_stage2_checkpoint "$best_ckpt" "$initial_ckpt"

    log "START ${name}: best epoch ${source_epoch}; 150 additional epochs (final epoch $((target_max - 1)))"
    CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
        python -m egoemg.train \
        experiment=fusion/fusion_allvision_s_egoemg_showee \
        +augmentation=batch_aug_best_v2 \
        "module.vision_backbone_type=${backbone}" \
        "module.vision_embed_dim=${vision_dim}" \
        "module.vision_pretrained_checkpoint=${vision_ckpt}" \
        "pretrained_emg_checkpoint=${INPUT_ROOT}/emgformer_s_full_showee.ckpt" \
        "batch_size=${batch_size}" \
        "val_batch_size=${batch_size}" \
        optimizer.lr=1e-5 \
        lr_scheduler.scheduler.T_max=150 \
        lr_scheduler.scheduler.eta_min=1e-6 \
        "resume_ckpt=${initial_ckpt}" \
        train=true eval=false \
        "trainer.devices=${TRAINER_GPUS}" \
        "trainer.max_epochs=${target_max}" \
        "hydra.run.dir=${stage_dir}/hydra" \
        "logger.save_dir=${stage_dir}" logger.name=train logger.version=0 \
        2>&1 | tee "${stage_dir}/console.log"
    log "DONE ${name}"
}

cd "$REPO"
run_job fusion_rn18_s resnet18 512 "${VISION_ROOT}/rn18.ckpt" 200 \
    "${RUN_ROOT}/fusion_rn18_s/stage2_lr1e-5_eta1e-6_150e/train/version_0/checkpoints/fusion-s-epoch=163-val_mae=0.0954.ckpt"
run_job fusion_rn50_s resnet50 2048 "${VISION_ROOT}/rn50.ckpt" 210 \
    "${RUN_ROOT}/fusion_rn50_s/stage2_lr1e-5_eta1e-6_150e/train/version_0/checkpoints/fusion-s-epoch=246-val_mae=0.0939.ckpt"
run_job fusion_rn152_s resnet152 2048 "${VISION_ROOT}/rn152.ckpt" 40 \
    "${RUN_ROOT}/fusion_rn152_s/train/version_0/checkpoints/fusion-s-epoch=247-val_mae=0.0959.ckpt"
log 'ALL RN STAGE-2 FUSION RUNS COMPLETED'
