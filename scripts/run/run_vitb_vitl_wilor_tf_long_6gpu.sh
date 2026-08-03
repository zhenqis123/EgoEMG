#!/usr/bin/env bash
# Serial long-run sweep for simple EMGFormer-S fusion:
#   ViT-B (T, F) -> ViT-L (T, F) -> WiLoR (T, F)
# T: trainable visual feature backbone; F: frozen vision backbone + pose head.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=python
RUN_ROOT=${RUN_ROOT:-${REPO}/logs/20260728/vitb_vitl_wilor_s_simple_tf_lr1e-5_eta1e-6_50e_stride400}
EPOCHS=${EPOCHS:-50}
TRAIN_STRIDE=${TRAIN_STRIDE:-400}
DATASET_REPEAT=${DATASET_REPEAT:-1}
DEVICES='[0,1,2,3,4,5]'
GPUS=0,1,2,3,4,5

cd "${REPO}"
mkdir -p "${RUN_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "${RUN_ROOT}/queue.log"; }

run_job() {
    local name=$1
    local experiment=$2
    local batch_size=$3
    local job_dir=${RUN_ROOT}/${name}
    local done_file=${job_dir}/DONE
    local stopped_file=${job_dir}/STOPPED
    local checkpoint_dir=${job_dir}/train/version_0/checkpoints
    local -a resume_args=()

    if [[ -f "${done_file}" ]]; then
        log "SKIP completed ${name}"
        return
    fi
    if [[ -f "${stopped_file}" ]]; then
        log "SKIP manually stopped ${name}"
        return
    fi

    mkdir -p "${job_dir}"

    # Only the best checkpoint is retained to keep the six-job queue within
    # local disk limits. If a run was interrupted, resume from that checkpoint.
    if [[ -d "${checkpoint_dir}" ]]; then
        local resume_ckpt
        resume_ckpt=$(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.ckpt' \
            -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2- || true)
        if [[ -n "${resume_ckpt}" ]]; then
            resume_args+=("resume_ckpt=${resume_ckpt}")
            log "RESUME ${name} from ${resume_ckpt}"
        fi
    fi

    if (( ${#resume_args[@]} == 0 )); then
        log "START ${name} from pretrained vision and EMG checkpoints"
    fi
    log "CONFIG ${name}: batch_size=${batch_size}, epochs=${EPOCHS}, stride=${TRAIN_STRIDE}, repeat=${DATASET_REPEAT}, lr=1e-5->1e-6"

    CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m egoemg.train \
        "experiment=fusion/${experiment}" \
        "batch_size=${batch_size}" "val_batch_size=${batch_size}" \
        "datamodule.stride=${TRAIN_STRIDE}" \
        "datamodule.dataset_repeat=${DATASET_REPEAT}" \
        train=true eval=false "trainer.devices=${DEVICES}" \
        "trainer.max_epochs=${EPOCHS}" \
        optimizer.lr=1.0e-5 \
        "lr_scheduler.scheduler.T_max=${EPOCHS}" \
        lr_scheduler.scheduler.eta_min=1.0e-6 \
        callbacks.0.save_top_k=1 callbacks.0.save_last=false \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" logger.name=train logger.version=0 \
        "${resume_args[@]}" \
        2>&1 | tee -a "${job_dir}/console.log"

    touch "${done_file}"
    log "DONE ${name}"
}

# Requested order: B -> L -> W; trainable then frozen within each backbone.
run_job vitb_t  fusion_vitb_s_simple_unfrozen_augbest_30e 100
run_job vitb_f  fusion_vitb_s_simple_frozen_augbest_30e   480
run_job vitl_t  fusion_vitl_s_simple_unfrozen_augbest_30e  32
run_job vitl_f  fusion_vitl_s_simple_frozen_augbest_30e   480
run_job wilor_t fusion_wilor_s_simple_unfrozen_augbest_30e 12
run_job wilor_f fusion_wilor_s_simple_frozen_augbest_30e  400

touch "${RUN_ROOT}/ALL_DONE"
log "ALL ViT-B/ViT-L/WiLoR T/F experiments completed"
