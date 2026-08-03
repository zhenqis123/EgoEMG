#!/usr/bin/env bash
# Queue RN50-S no-augmentation then legacy-WL7790-augmentation fusion runs.
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
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-640}
NUM_WORKERS=${NUM_WORKERS:-10}

required_paths=("$REPO" "$PYTHON_BIN" "$EGOEMG_MEMMAP_DIR" "$EGOEMG_CROPS_DIR" "$EMG_CKPT"
  "${INPUT_ROOT}/vision_checkpoints/rn50.ckpt"
  "${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json")
for required_path in "${required_paths[@]}"; do
  [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 1; }
done
mapfile -t gpu_info < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ ${#gpu_info[@]} -eq 8 ]] || { echo "Expected 8 visible GPUs, got ${#gpu_info[@]}." >&2; exit 2; }
for gpu in "${gpu_info[@]}"; do
  [[ "$gpu" == *A800* && "$gpu" == *"80"* ]] || { echo "Expected A800 80GB GPUs, got: $gpu" >&2; exit 2; }
done

run_job() {
  local experiment=$1 run_name=$2
  local output_root="${REPO}/logs/${run_name}/train"
  mkdir -p "$output_root"
  echo "[$(date '+%F %T')] START ${experiment}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m egoemg.train \
    "experiment=${experiment}" \
    "egoemg_memmap_dir=${EGOEMG_MEMMAP_DIR}" \
    "per_episode_crops_dir=${EGOEMG_CROPS_DIR}" \
    "vision_resnet_checkpoint=${INPUT_ROOT}/vision_checkpoints/rn50.ckpt" \
    "module.vision_pretrained_checkpoint=${INPUT_ROOT}/vision_checkpoints/rn50.ckpt" \
    "pretrained_emg_checkpoint=${EMG_CKPT}" \
    "datamodule.per_dataset_norm_stats_path=${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json" \
    "batch_size=${PER_GPU_BATCH_SIZE}" "val_batch_size=${PER_GPU_BATCH_SIZE}" "num_workers=${NUM_WORKERS}" \
    "optimizer.lr=${BASE_LR}" "lr_scheduler.scheduler.T_max=${MAX_EPOCHS}" "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    train=true eval=false 'trainer.devices=[0,1,2,3,4,5,6,7]' "trainer.max_epochs=${MAX_EPOCHS}" \
    "hydra.run.dir=${output_root}/hydra" "logger.save_dir=${output_root}" logger.name=train logger.version=0 \
    2>&1 | tee "${output_root}/console.log"
}

cd "$REPO"
run_job fusion/fusion_rn50_s_egoemg_only_noaug_dense fusion_rn50_s_egoemg_only_noaug_dense_lr5e-4_300e
run_job fusion/fusion_rn50_s_egoemg_only_legacy_wl7790_aug_dense fusion_rn50_s_egoemg_only_legacy_wl7790_aug_dense_lr5e-4_300e
