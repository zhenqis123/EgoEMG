#!/usr/bin/env bash
# Fresh RN50 + EMGFormer-S fusion, for an 8x A800 80GB allocation.
# It mirrors the local no-MixUp RN18 ablation while changing only RN18 -> RN50.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${EMG2POSE_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
SHARED_ROOT=${EMG2POSE_SHARED_ROOT:-${EMG2POSE_SHARED_ROOT:-/shared}/develop}
PYTHON_BIN=${PYTHON_BIN:-python}

INPUT_ROOT=${SHARED_ROOT}/experiment_inputs
EGOEMG_MEMMAP_DIR=${EGOEMG_MEMMAP_DIR:-${SHARED_ROOT}/data/EgoEMG_memmap}
EGOEMG_CROPS_DIR=${EGOEMG_CROPS_DIR:-${SHARED_ROOT}/data/EgoEMG_crops}
VISION_CKPT=${VISION_CKPT:-${INPUT_ROOT}/vision_checkpoints/rn50.ckpt}
EMG_CKPT=${EMG_CKPT:-${INPUT_ROOT}/emgformer_s_egoemg_incre_cotrain.ckpt}

# The RN50-M profile used batch 640 at about 64--66 GiB on A800 80GB.  This
# RN50-S variant is smaller, so this preserves safe headroom.
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-640}
NUM_WORKERS=${NUM_WORKERS:-10}
RUN_NAME=${RUN_NAME:-fresh_egoemg_only_augbest_no_mixup_wl12000_s400_r2_200e}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO}/logs/fusion_rn50_s/${RUN_NAME}}

required_paths=(
  "$REPO" "$PYTHON_BIN" "$EGOEMG_MEMMAP_DIR" "$EGOEMG_CROPS_DIR"
  "$VISION_CKPT" "$EMG_CKPT"
  "${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
)
for path in "${required_paths[@]}"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

mapfile -t gpu_info < <(nvidia-smi --query-gpu=name,memory.total \
  --format=csv,noheader,nounits)
[[ ${#gpu_info[@]} -eq 8 ]] || { echo "Expected 8 visible GPUs, got ${#gpu_info[@]}." >&2; exit 2; }
for gpu in "${gpu_info[@]}"; do
  [[ "$gpu" == *A800* ]] || { echo "Expected A800 GPUs, got: $gpu" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
echo "RN50-S fusion: EgoEMG only, WL12000/stride400/repeat2, no MixUp"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m egoemg.train \
  experiment=fusion/fusion_rn50_s_egoemg_only_augbest_no_mixup \
  "egoemg_memmap_dir=${EGOEMG_MEMMAP_DIR}" \
  "per_episode_crops_dir=${EGOEMG_CROPS_DIR}" \
  "vision_resnet_checkpoint=${VISION_CKPT}" \
  "module.vision_pretrained_checkpoint=${VISION_CKPT}" \
  "pretrained_emg_checkpoint=${EMG_CKPT}" \
  "datamodule.per_dataset_norm_stats_path=${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json" \
  "batch_size=${PER_GPU_BATCH_SIZE}" "val_batch_size=${PER_GPU_BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  train=true eval=false \
  'trainer.devices=[0,1,2,3,4,5,6,7]' trainer.max_epochs=200 \
  "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
  "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
  2>&1 | tee "${OUTPUT_ROOT}/console.log"
