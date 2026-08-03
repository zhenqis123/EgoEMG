#!/usr/bin/env bash
# Table 4 simple residual-fusion sweep. Every visual predictor is frozen; only
# EMGFormer-S and the lightweight residual pathway are optimized.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${REPO}/logs/20260727/fusion_5vision_s_simple_frozen_augbest_30e_bstuned
DEVICES='[0,1,2,3,4,5]'
GPUS=0,1,2,3,4,5

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

    if checkpoint_reached_epoch_29 "${last_ckpt}"; then
        log "SKIP completed ${name}"
        return
    fi
    if [[ -e "${job_dir}" ]]; then
        log "REFUSE non-empty/incomplete existing run: ${job_dir}"
        return 1
    fi

    mkdir -p "${job_dir}"
    log "START ${name} batch_size=${batch_size}"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m egoemg.train \
        "experiment=fusion/${experiment}" \
        "batch_size=${batch_size}" "val_batch_size=${batch_size}" \
        train=true eval=false "trainer.devices=${DEVICES}" \
        trainer.max_epochs=30 \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" logger.name=train logger.version=0 \
        2>&1 | tee "${job_dir}/console.log"

    checkpoint_reached_epoch_29 "${last_ckpt}"
    touch "${job_dir}/DONE"
    log "DONE ${name}"
}

run_job rn50_s fusion_rn50_s_simple_frozen_augbest_30e 600
run_job rn152_s fusion_rn152_s_simple_frozen_augbest_30e 600
run_job vitb_s fusion_vitb_s_simple_frozen_augbest_30e 480
run_job vitl_s fusion_vitl_s_simple_frozen_augbest_30e 480
run_job wilor_s fusion_wilor_s_simple_frozen_augbest_30e 400

touch "${RUN_ROOT}/ALL_DONE"
log "ALL SIMPLE FROZEN-VISION EXPERIMENTS COMPLETED"
