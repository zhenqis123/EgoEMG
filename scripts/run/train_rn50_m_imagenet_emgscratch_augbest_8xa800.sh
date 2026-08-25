#!/usr/bin/env bash
# Fresh RN50 (torchvision ImageNet initialization) + randomly initialized
# EMGFormer-M fusion.  This is deliberately not a continuation experiment.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${EMG2POSE_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
SHARED_ROOT=${EMG2POSE_SHARED_ROOT:-${EMG2POSE_SHARED_ROOT:-/shared}/develop}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCH_HOME=${TORCH_HOME:-${SHARED_ROOT}/experiment_inputs/torch_cache}
RESNET50_WEIGHTS=${TORCH_HOME}/hub/checkpoints/resnet50-0676ba61.pth

EGOEMG_MEMMAP_DIR=${EGOEMG_MEMMAP_DIR:-${SHARED_ROOT}/data/EgoEMG_memmap}
SHOWEE_MEMMAP_DIR=${SHOWEE_MEMMAP_DIR:-${SHARED_ROOT}/data/ShowEE_202607_memmap}
EGOEMG_CROPS_DIR=${EGOEMG_CROPS_DIR:-${SHARED_ROOT}/data/EgoEMG_crops}
SHOWEE_CROPS_DIR=${SHOWEE_CROPS_DIR:-${SHARED_ROOT}/data/ShowEE_202607_crops}

# Profiled with the exact model/configuration: 19.50 GiB at batch 180 on a
# 24 GiB RTX 4090.  Batch 640 extrapolates to roughly 64--66 GiB on A800 80GB,
# preserving substantial headroom for DDP, allocator fragmentation, and IO.
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-640}
NUM_WORKERS=${NUM_WORKERS:-10}
RUN_NAME=${RUN_NAME:-fresh_imagenet_emgscratch_augbest_lr1e-4_eta1e-6_150e}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO}/logs/full_showee_s_l_all_fusions_20260721/fusion_rn50_m/${RUN_NAME}}

required_paths=(
    "$REPO" "$PYTHON_BIN" "$EGOEMG_MEMMAP_DIR" "$SHOWEE_MEMMAP_DIR"
    "$EGOEMG_CROPS_DIR" "$SHOWEE_CROPS_DIR"
    "$RESNET50_WEIGHTS"
    "${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
)
for required_path in "${required_paths[@]}"; do
    [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 1; }
done
"$PYTHON_BIN" -c 'import torch, torchvision, pytorch_lightning, lmdb' >/dev/null

mapfile -t gpu_info < <(nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader,nounits)
[[ ${#gpu_info[@]} -eq 8 ]] || { echo "Expected exactly 8 visible GPUs, found ${#gpu_info[@]}." >&2; exit 2; }
for gpu_info_line in "${gpu_info[@]}"; do
    [[ "$gpu_info_line" == *A800* ]] || { echo "Expected A800 GPUs, got: $gpu_info_line" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
echo "Fresh RN50 ImageNet + EMGFormer-M scratch fusion: 8x A800, per-GPU batch ${PER_GPU_BATCH_SIZE}"

TORCH_HOME="$TORCH_HOME" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m egoemg.train \
    experiment=fusion/fusion_rn50_m_imagenet_emgscratch_augbest \
    "egoemg_memmap_dir=${EGOEMG_MEMMAP_DIR}" \
    "showee_memmap_dir=${SHOWEE_MEMMAP_DIR}" \
    "egoemg_per_episode_crops_dir=${EGOEMG_CROPS_DIR}" \
    "showee_per_episode_crops_dir=${SHOWEE_CROPS_DIR}" \
    egoemg_video_root=null egoemg_allintra_root=null \
    showee_video_root=null showee_allintra_root=null \
    "datamodule.per_dataset_norm_stats_path=${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json" \
    "batch_size=${PER_GPU_BATCH_SIZE}" "val_batch_size=${PER_GPU_BATCH_SIZE}" \
    "num_workers=${NUM_WORKERS}" \
    train=true eval=false 'trainer.devices=[0,1,2,3,4,5,6,7]' \
    "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
    "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
    2>&1 | tee "${OUTPUT_ROOT}/console.log"
