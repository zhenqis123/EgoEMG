#!/usr/bin/env bash
# Queue two controlled RN18 fusion augmentation ablations on all six local GPUs.
# Both retain the finished run's data, initializers, optimizer, and schedule:
# EgoEMG-only, WL12000, stride=400, repeat=2, LR 5e-4 -> 5e-6, 300 epochs.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOG_ROOT=data/logs
GPUS=0,1,2,3,4,5
DEVICES='[0,1,2,3,4,5]'

run_experiment() {
  local experiment="$1"
  local run_name="$2"
  local run_dir="${LOG_ROOT}/${run_name}/train"

  mkdir -p "${run_dir}"
  echo "[$(date '+%F %T')] Starting ${experiment}" | tee "${run_dir}/console.log"
  CUDA_VISIBLE_DEVICES="${GPUS}" python -m egoemg.train \
    "experiment=${experiment}" \
    batch_size=200 \
    val_batch_size=200 \
    train=true eval=false \
    "trainer.devices=${DEVICES}" \
    trainer.max_epochs=300 \
    "hydra.run.dir=${run_dir}/hydra" \
    "logger.save_dir=${run_dir}" logger.name=train logger.version=0 \
    2>&1 | tee -a "${run_dir}/console.log"
}

cd "${REPO}"
run_experiment \
  fusion/fusion_rn18_s_egoemg_only_noaug_dense \
  fusion_rn18_s_egoemg_only_noaug_dense_lr5e-4_300e_20260724
run_experiment \
  fusion/fusion_rn18_s_egoemg_only_legacy_wl7790_aug_dense \
  fusion_rn18_s_egoemg_only_legacy_wl7790_aug_dense_lr5e-4_300e_20260724
