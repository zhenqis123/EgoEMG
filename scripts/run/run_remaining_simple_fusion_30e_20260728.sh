#!/usr/bin/env bash
# Complete WiLoR-S frozen, then run the five missing fully-unfrozen simple
# fusion cells. Safe to relaunch: completed jobs are skipped and incomplete
# jobs resume from last.ckpt.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${REPO}/logs/20260728/fusion_5vision_s_simple_unfrozen_augbest_30e
WILOR_FROZEN_ROOT=${REPO}/logs/20260728/wilor_s_simple_frozen_exactvision_augbest_lr1e-5_eta5e-6_30e
# Keep GPU 5 free for unified evaluation while this queue is running.
DEVICES='[0,1,2,3,4]'
GPUS=0,1,2,3,4

conda activate emg2pose_env
cd "${REPO}"
mkdir -p "${RUN_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

checkpoint_reached_epoch_29() {
    local checkpoint=$1
    [[ -f "${checkpoint}" ]] || return 1
    python - "${checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) >= 29 else 1)
PY
}

run_job() {
    local name=$1
    local experiment=$2
    local batch_size=$3
    local job_dir=${RUN_ROOT}/${name}
    local last_ckpt=${job_dir}/train/version_0/checkpoints/last.ckpt
    local -a resume_args=()

    if checkpoint_reached_epoch_29 "${last_ckpt}"; then
        log "SKIP completed ${name}"
        return
    fi
    if [[ -f "${last_ckpt}" ]]; then
        resume_args+=("resume_ckpt=${last_ckpt}")
        log "RESUME ${name} batch_size=${batch_size} from ${last_ckpt}"
    else
        mkdir -p "${job_dir}"
        log "START ${name} batch_size=${batch_size}"
    fi

    CUDA_VISIBLE_DEVICES="${GPUS}" python -m egoemg.train \
        "experiment=fusion/${experiment}" \
        "batch_size=${batch_size}" "val_batch_size=${batch_size}" \
        train=true eval=false "trainer.devices=${DEVICES}" \
        trainer.max_epochs=30 \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" logger.name=train logger.version=0 \
        "${resume_args[@]}" \
        2>&1 | tee -a "${job_dir}/console.log"

    checkpoint_reached_epoch_29 "${last_ckpt}"
    touch "${job_dir}/DONE"
    log "DONE ${name}"
}

run_wilor_frozen() {
    local last_ckpt=${WILOR_FROZEN_ROOT}/train/version_0/checkpoints/last.ckpt
    if checkpoint_reached_epoch_29 "${last_ckpt}"; then
        log "SKIP completed wilor_s_simple_frozen"
        return
    fi
    [[ -f "${last_ckpt}" ]] || {
        log "Missing WiLoR frozen resume checkpoint: ${last_ckpt}"
        return 1
    }

    log "RESUME wilor_s_simple_frozen from ${last_ckpt}"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m egoemg.train \
        experiment=fusion/fusion_wilor_s_simple_frozen_augbest_30e \
        batch_size=400 val_batch_size=400 \
        train=true eval=false "trainer.devices=${DEVICES}" \
        trainer.max_epochs=30 optimizer.lr=1.0e-5 \
        lr_scheduler.scheduler.T_max=30 \
        lr_scheduler.scheduler.eta_min=5.0e-6 \
        callbacks.0.save_top_k=1 \
        "resume_ckpt=${last_ckpt}" \
        "hydra.run.dir=${WILOR_FROZEN_ROOT}/hydra" \
        "logger.save_dir=${WILOR_FROZEN_ROOT}" logger.name=train logger.version=0 \
        2>&1 | tee -a "${WILOR_FROZEN_ROOT}/console.log"

    checkpoint_reached_epoch_29 "${last_ckpt}"
    touch "${WILOR_FROZEN_ROOT}/DONE"
    log "DONE wilor_s_simple_frozen"
}

run_wilor_frozen
run_job rn50_s fusion_rn50_s_simple_unfrozen_augbest_30e 210
run_job rn152_s fusion_rn152_s_simple_unfrozen_augbest_30e 40
# ViT-B unfrozen was stopped by request after epoch 27 because its validation
# MAE remained well below the pretrained vision-only baseline.
run_job vitl_s fusion_vitl_s_simple_unfrozen_augbest_30e 32
run_job wilor_s fusion_wilor_s_simple_unfrozen_augbest_30e 12
run_job vits_s_frozen fusion_vits_s_simple_frozen_augbest_30e 480

touch "${RUN_ROOT}/ALL_DONE"
log "ALL REMAINING SIMPLE FUSION EXPERIMENTS COMPLETED"
