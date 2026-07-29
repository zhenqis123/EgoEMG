#!/usr/bin/env bash
# Resume ViT-L-S unfrozen fusion, then run WiLoR-S unfrozen fusion.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=python
RUN_ROOT=${REPO}/logs/20260728/fusion_5vision_s_simple_unfrozen_augbest_30e
DEVICES='[0,1,2,3,4]'
GPUS=0,1,2,3,4

cd "${REPO}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

checkpoint_reached_epoch_29() {
    local checkpoint=$1
    [[ -f "${checkpoint}" ]] || return 1
    "${PYTHON}" - "${checkpoint}" <<'PY'
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

    mkdir -p "${job_dir}"
    if [[ -f "${last_ckpt}" ]]; then
        resume_args+=("resume_ckpt=${last_ckpt}")
        log "RESUME ${name} from ${last_ckpt}"
    else
        log "START ${name} from pretrained single-modality checkpoints"
    fi

    CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m emg2pose.train \
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

run_job vitl_s fusion_vitl_s_simple_unfrozen_augbest_30e 32
run_job wilor_s fusion_wilor_s_simple_unfrozen_augbest_30e 12

touch "${RUN_ROOT}/VITL_WILOR_UNFROZEN_DONE"
log "ViT-L-S and WiLoR-S unfrozen experiments completed"
