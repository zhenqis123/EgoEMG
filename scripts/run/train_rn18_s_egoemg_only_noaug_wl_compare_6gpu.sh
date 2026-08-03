#!/usr/bin/env bash
# Controlled 50-epoch comparison: WL12000 then WL4000, with no augmentation.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=data/logs
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

run_job() {
  local experiment=$1 run_name=$2
  local run_dir="${LOG_ROOT}/${run_name}/train"
  mkdir -p "$run_dir"
  echo "[$(date '+%F %T')] START ${experiment}" | tee "$run_dir/console.log"
  CUDA_VISIBLE_DEVICES="$GPUS" python -m egoemg.train \
    "experiment=${experiment}" \
    batch_size=200 val_batch_size=200 \
    train=true eval=false "trainer.devices=${DEVICES}" trainer.max_epochs=50 \
    "hydra.run.dir=${run_dir}/hydra" \
    "logger.save_dir=${run_dir}" logger.name=train logger.version=0 \
    2>&1 | tee -a "$run_dir/console.log"
}

cd "$REPO"
run_job fusion/fusion_rn18_s_egoemg_only_noaug_wl12000_50e \
  fusion_rn18_s_egoemg_only_noaug_wl12000_50e_20260724
run_job fusion/fusion_rn18_s_egoemg_only_noaug_wl4000_50e \
  fusion_rn18_s_egoemg_only_noaug_wl4000_50e_20260724
