#!/usr/bin/env bash
# Sequential fresh RN50-S then RN152-S fusion runs for an 8x A800 80GB pod.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${EMG2POSE_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
SHARED_ROOT=${EMG2POSE_SHARED_ROOT:-${EMG2POSE_SHARED_ROOT:-/shared}/develop}
PYTHON_BIN=${PYTHON_BIN:-python}
INPUT_ROOT=${SHARED_ROOT}/experiment_inputs

EGOEMG_MEMMAP_DIR=${EGOEMG_MEMMAP_DIR:-${SHARED_ROOT}/data/EgoEMG_memmap}
EGOEMG_CROPS_DIR=${EGOEMG_CROPS_DIR:-${SHARED_ROOT}/data/EgoEMG_v2_crops}
EMG_CKPT=${EMG_CKPT:-${INPUT_ROOT}/emgformer_s_egoemg_incre_cotrain.ckpt}
MAX_EPOCHS=${MAX_EPOCHS:-300}
BASE_LR=${BASE_LR:-5e-4}
ETA_MIN=${ETA_MIN:-5e-6}
NUM_WORKERS=${NUM_WORKERS:-10}

required_paths=(
  "$REPO" "$PYTHON_BIN" "$EGOEMG_MEMMAP_DIR" "$EGOEMG_CROPS_DIR" "$EMG_CKPT"
  "${INPUT_ROOT}/vision_checkpoints/rn50.ckpt"
  "${INPUT_ROOT}/vision_checkpoints/rn152.ckpt"
  "${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
)
for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

mapfile -t gpu_info < <(nvidia-smi --query-gpu=name,memory.total \
  --format=csv,noheader,nounits)
[[ ${#gpu_info[@]} -eq 8 ]] || { echo "Expected 8 visible GPUs, got ${#gpu_info[@]}." >&2; exit 2; }
for gpu in "${gpu_info[@]}"; do
  [[ "$gpu" == *A800* && "$gpu" == *"80"* ]] || {
    echo "Expected A800 80GB GPUs, got: $gpu" >&2; exit 2;
  }
done

run_job() {
  local name=$1 experiment=$2 vision_ckpt=$3 per_gpu_batch=$4
  local output_root="${REPO}/logs/${name}/fresh_egoemg_only_augbest_no_mixup_wl12000_s400_r2_lr5e-4_300e"
  mkdir -p "$output_root"
  echo "[$(date '+%F %T')] START ${name}: batch=${per_gpu_batch}, epochs=${MAX_EPOCHS}, lr=${BASE_LR}->${ETA_MIN}"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m emg2pose.train \
    "experiment=${experiment}" \
    "egoemg_memmap_dir=${EGOEMG_MEMMAP_DIR}" \
    "per_episode_crops_dir=${EGOEMG_CROPS_DIR}" \
    "vision_resnet_checkpoint=${vision_ckpt}" \
    "module.vision_pretrained_checkpoint=${vision_ckpt}" \
    "pretrained_emg_checkpoint=${EMG_CKPT}" \
    "datamodule.per_dataset_norm_stats_path=${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json" \
    "batch_size=${per_gpu_batch}" "val_batch_size=${per_gpu_batch}" \
    "num_workers=${NUM_WORKERS}" \
    "optimizer.lr=${BASE_LR}" \
    "lr_scheduler.scheduler.T_max=${MAX_EPOCHS}" \
    "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    train=true eval=false \
    'trainer.devices=[0,1,2,3,4,5,6,7]' "trainer.max_epochs=${MAX_EPOCHS}" \
    "hydra.run.dir=${output_root}/hydra" \
    "logger.save_dir=${output_root}" logger.name=train logger.version=0 \
    2>&1 | tee "${output_root}/console.log"
  echo "[$(date '+%F %T')] DONE ${name}"
}

cd "$REPO"
run_job fusion_rn50_s_egoemg_only_augbest_no_mixup \
  fusion/fusion_rn50_s_egoemg_only_augbest_no_mixup \
  "${INPUT_ROOT}/vision_checkpoints/rn50.ckpt" 640
run_job fusion_rn152_s_egoemg_only_augbest_no_mixup \
  fusion/fusion_rn152_s_egoemg_only_augbest_no_mixup \
  "${INPUT_ROOT}/vision_checkpoints/rn152.ckpt" 480
