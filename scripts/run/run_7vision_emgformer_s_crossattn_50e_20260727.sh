#!/usr/bin/env bash
# Seven vision baselines + EMGFormer-S.  RN50 is the already-running first
# member; this queue waits for it, then runs the other six in order, WiLoR last.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${REPO}/logs/20260727/fusion_7vision_emgformer_s_crossattn_50e
RN50_ROOT=${REPO}/logs/20260727/fusion_rn50_s_egoemg_only_wl12000_crossattn_lastblock_anchor_augbest_50e
RN50_TMUX=train_rn50_s_crossattn_50e_20260727
DEVICES='[0,1,2,3,4,5]'
GPUS=0,1,2,3,4,5

conda activate egoemg_env
mkdir -p "$RUN_ROOT"
ln -sfn "$RN50_ROOT" "${RUN_ROOT}/rn50_s"
cd "$REPO"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

checkpoint_reached_epoch_49() {
    local checkpoint=$1
    [[ -f $checkpoint ]] || return 1
    python - "$checkpoint" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) >= 49 else 1)
PY
}

log "Waiting for the already-running RN50-S experiment"
while tmux has-session -t "$RN50_TMUX" 2>/dev/null; do
    sleep 60
done
RN50_LAST=${RN50_ROOT}/train/train/version_0/checkpoints/last.ckpt
if ! checkpoint_reached_epoch_49 "$RN50_LAST"; then
    log "RN50-S stopped before epoch 49; refusing to start a misleading queue"
    exit 1
fi
log "RN50-S complete"

run_job() {
    local name=$1
    local experiment=$2
    local batch_size=$3
    local job_dir=${RUN_ROOT}/${name}
    local last_ckpt=${job_dir}/train/version_0/checkpoints/last.ckpt
    local resume_args=()

    if checkpoint_reached_epoch_49 "$last_ckpt"; then
        log "SKIP completed ${name}"
        return
    fi
    if [[ -f $last_ckpt ]]; then
        resume_args=("resume_ckpt=${last_ckpt}")
        log "RESUME ${name} from ${last_ckpt}"
    fi

    mkdir -p "$job_dir"
    log "START ${name} batch_size=${batch_size}"
    CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
        "experiment=fusion/${experiment}" \
        "batch_size=${batch_size}" "val_batch_size=${batch_size}" \
        train=true eval=false "trainer.devices=${DEVICES}" \
        trainer.max_epochs=50 \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" logger.name=train logger.version=0 \
        "${resume_args[@]}" \
        2>&1 | tee -a "${job_dir}/console.log"
    checkpoint_reached_epoch_49 "$last_ckpt"
    touch "${job_dir}/DONE"
    log "DONE ${name}"
}

run_job rn18_s fusion_rn18_s_egoemg_only_wl12000_crossattn_lastblock_anchor_augbest_50e 200
run_job vits_s fusion_vits_s_egoemg_only_augbest_50e 200
run_job vitb_s fusion_vitb_s_egoemg_only_augbest_50e 100
run_job vitl_s fusion_vitl_s_egoemg_only_augbest_50e 32
run_job rn152_s fusion_rn152_s_egoemg_only_wl12000_crossattn_lastblock_anchor_augbest_50e 40
run_job wilor_s fusion_wilor_s_egoemg_only_augbest_50e 12

touch "${RUN_ROOT}/ALL_DONE"
log "ALL SEVEN VISION + EMGFORMER-S EXPERIMENTS COMPLETED"
