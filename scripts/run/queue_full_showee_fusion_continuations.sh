#!/usr/bin/env bash
# Sequentially continue every existing full-data fusion run to epoch 249.
# RN50 is already running when this queue is launched, so it is only waited on.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721
INPUT_ROOT=data/experiment_inputs
VISION_ROOT="${INPUT_ROOT}/vision_checkpoints"
GPU_LIST='[0,1,2,3,4,5]'
TARGET_LAST_EPOCH=249

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

rn50_running() {
    pgrep -f 'module\.vision_backbone_type=resnet50.*fusion_rn50_s' >/dev/null
}

wait_for_rn50() {
    while rn50_running; do
        log 'RN50 continuation is active; queue is waiting before using all six GPUs.'
        sleep 60
    done
}

checkpoint_epoch() {
    python - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)))
PY
}

# Changes the serialized optimizer/scheduler state, because simply overriding
# Hydra values does not override state restored from a Lightning checkpoint.
prepare_cosine_resume() {
    local source=$1
    local base_lr=$2
    local eta_min=$3
    local label=$4
    python - "$source" "$base_lr" "$eta_min" "$label" <<'PY'
import math
import sys
from pathlib import Path

import torch

source = Path(sys.argv[1])
base_lr = float(sys.argv[2])
eta_min = float(sys.argv[3])
label = sys.argv[4]
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
phase = max(0, epoch + 1)
lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * phase / 250.0)) / 2

for optimizer in checkpoint.get("optimizer_states", []):
    for group in optimizer.get("param_groups", []):
        group["initial_lr"] = base_lr
        group["lr"] = lr

for scheduler in checkpoint.get("lr_schedulers", []):
    scheduler["T_max"] = 250
    scheduler["eta_min"] = eta_min
    scheduler["base_lrs"] = [base_lr for _ in scheduler.get("base_lrs", [base_lr])]
    scheduler["last_epoch"] = phase
    scheduler["_step_count"] = phase + 1
    scheduler["_last_lr"] = [lr for _ in scheduler["base_lrs"]]

destination = source.with_name(
    f"resume_250_{label}_from_epoch_{epoch:03d}.ckpt"
)
torch.save(checkpoint, destination)
print(destination)
PY
}

run_job() {
    local name=$1
    local backbone=$2
    local vision_dim=$3
    local vision_ckpt=$4
    local batch_size=$5
    local policy=$6
    local base_lr=$7
    local eta_min=$8
    local job_dir="${RUN_ROOT}/${name}"
    local checkpoint="${job_dir}/train/version_0/checkpoints/last.ckpt"

    [[ -f "$checkpoint" ]] || { log "ERROR: missing $checkpoint"; return 1; }
    local epoch
    epoch=$(checkpoint_epoch "$checkpoint")
    if (( epoch >= TARGET_LAST_EPOCH )); then
        log "SKIP ${name}: already at epoch ${epoch}"
        return
    fi

    local resume_ckpt=$checkpoint
    if [[ "$policy" == "new_cosine" ]]; then
        resume_ckpt=$(prepare_cosine_resume "$checkpoint" "$base_lr" "$eta_min" "$name")
    fi

    log "START ${name}: epoch ${epoch} -> ${TARGET_LAST_EPOCH}; resume=${resume_ckpt}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
        python -m egoemg.train \
        experiment=fusion/fusion_allvision_s_egoemg_showee \
        +augmentation=batch_aug_best_v2 \
        "module.vision_backbone_type=${backbone}" \
        "module.vision_embed_dim=${vision_dim}" \
        "module.vision_pretrained_checkpoint=${vision_ckpt}" \
        "pretrained_emg_checkpoint=${INPUT_ROOT}/emgformer_s_full_showee.ckpt" \
        "batch_size=${batch_size}" \
        "val_batch_size=${batch_size}" \
        "optimizer.lr=${base_lr}" \
        "lr_scheduler.scheduler.T_max=250" \
        "lr_scheduler.scheduler.eta_min=${eta_min}" \
        "resume_ckpt=${resume_ckpt}" \
        train=true eval=false \
        "trainer.devices=${GPU_LIST}" \
        trainer.max_epochs=250 \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" logger.name=train logger.version=0 \
        2>&1 | tee "${job_dir}/continuation_250_console.log"
    log "DONE ${name}"
}

cd "$REPO"
log "Queue started: ${RUN_ROOT}"
wait_for_rn50

# RN18 already has the prior 100-epoch continuation scheduler serialized in
# its checkpoint; preserve that state rather than resetting it a second time.
run_job fusion_rn18_s resnet18 512 "${VISION_ROOT}/rn18.ckpt" 200 preserve 1e-4 1e-5

# RN152 follows RN50's 250-epoch cosine family. Its original batch size (40)
# is deliberately retained: RN50's 210 batch was a model-specific request.
run_job fusion_rn152_s resnet152 2048 "${VISION_ROOT}/rn152.ckpt" 40 new_cosine 1e-4 5e-6

# All ViT fusion continuations use the requested 5e-5 / 1e-6 / 250 schedule.
run_job fusion_vits_s vit_small 384 "${VISION_ROOT}/vits.ckpt" 200 new_cosine 5e-5 1e-6
run_job fusion_vitb_s vit_base 768 "${VISION_ROOT}/vitb.ckpt" 100 new_cosine 5e-5 1e-6
run_job fusion_vitl_s vit_large 1024 "${VISION_ROOT}/vitl.ckpt" 32 new_cosine 5e-5 1e-6

log 'ALL QUEUED FUSION CONTINUATIONS COMPLETED'
