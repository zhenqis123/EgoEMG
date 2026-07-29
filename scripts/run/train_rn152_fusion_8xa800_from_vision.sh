#!/usr/bin/env bash
# Fresh RN152 + EMGFormer-S fusion training on eight A800 80GB GPUs.
# No fusion checkpoint is resumed: the model is initialized from the RN152
# vision-only checkpoint and the supervised EMGFormer-S checkpoint.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${EMG2POSE_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
SHARED_ROOT=${EMG2POSE_SHARED_ROOT:-/share/being-h/xizh/develop}
PYTHON_BIN=${PYTHON_BIN:-/share/conda_envs/miniconda3/envs/v2c_env/bin/python}

EGOEMG_MEMMAP_DIR=${EGOEMG_MEMMAP_DIR:-${SHARED_ROOT}/data/EgoEMG_memmap}
SHOWEE_MEMMAP_DIR=${SHOWEE_MEMMAP_DIR:-${SHARED_ROOT}/data/ShowEE_202607_memmap}
EGOEMG_CROPS_DIR=${EGOEMG_CROPS_DIR:-${SHARED_ROOT}/data/EgoEMG_v2_crops}
SHOWEE_CROPS_DIR=${SHOWEE_CROPS_DIR:-${SHARED_ROOT}/data/ShowEE_202607_crops}
INPUT_ROOT=${EMG2POSE_INPUT_ROOT:-${SHARED_ROOT}/experiment_inputs}

PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-480}
MAX_EPOCHS=${MAX_EPOCHS:-250}
BASE_LR=${BASE_LR:-1e-5}
ETA_MIN=${ETA_MIN:-1e-6}
NUM_WORKERS=${NUM_WORKERS:-10}
RUN_NAME=${RUN_NAME:-fresh_8xa800_bs480_visioninit_lr1e-4_eta1e-6_250e}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO}/logs/full_showee_s_l_all_fusions_20260721/fusion_rn152_s/${RUN_NAME}}

required_paths=(
    "$REPO"
    "$PYTHON_BIN"
    "$EGOEMG_MEMMAP_DIR"
    "$SHOWEE_MEMMAP_DIR"
    "$EGOEMG_CROPS_DIR"
    "$SHOWEE_CROPS_DIR"
    "${INPUT_ROOT}/vision_checkpoints/rn152.ckpt"
    "${INPUT_ROOT}/emgformer_s_full_showee.ckpt"
    "${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json"
)
for required_path in "${required_paths[@]}"; do
    [[ -e "$required_path" ]] || {
        echo "Missing required path: $required_path" >&2
        exit 1
    }
done
"$PYTHON_BIN" -c 'import torch, pytorch_lightning, lmdb' >/dev/null

mapfile -t gpu_info < <(nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader,nounits)
if [[ ${#gpu_info[@]} -ne 8 ]]; then
    echo "Expected exactly 8 visible GPUs, found ${#gpu_info[@]}." >&2
    exit 2
fi
for gpu_index in "${!gpu_info[@]}"; do
    gpu_name=${gpu_info[$gpu_index]%,*}
    gpu_memory_mib=${gpu_info[$gpu_index]##*, }
    if [[ "$gpu_name" != *A800* || "$gpu_memory_mib" -lt 78000 ]]; then
        echo "GPU $gpu_index is '$gpu_name' (${gpu_memory_mib} MiB); expected A800 80GB." >&2
        exit 2
    fi
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"

echo "Starting fresh RN152 fusion on 8x A800 80GB"
echo "Per-GPU batch: $PER_GPU_BATCH_SIZE; global batch: $((PER_GPU_BATCH_SIZE * 8))"
echo "Epochs: $MAX_EPOCHS; LR: $BASE_LR -> $ETA_MIN"
echo "Vision init: ${INPUT_ROOT}/vision_checkpoints/rn152.ckpt"
echo "EMG init: ${INPUT_ROOT}/emgformer_s_full_showee.ckpt"
echo "Fusion resume checkpoint: none"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m emg2pose.train \
    experiment=fusion/fusion_allvision_s_egoemg_showee \
    +augmentation=batch_aug_best_v2 \
    module.vision_backbone_type=resnet152 \
    module.vision_embed_dim=2048 \
    "module.vision_pretrained_checkpoint=${INPUT_ROOT}/vision_checkpoints/rn152.ckpt" \
    "pretrained_emg_checkpoint=${INPUT_ROOT}/emgformer_s_full_showee.ckpt" \
    "egoemg_memmap_dir=${EGOEMG_MEMMAP_DIR}" \
    "showee_memmap_dir=${SHOWEE_MEMMAP_DIR}" \
    "egoemg_per_episode_crops_dir=${EGOEMG_CROPS_DIR}" \
    "showee_per_episode_crops_dir=${SHOWEE_CROPS_DIR}" \
    egoemg_video_root=null egoemg_allintra_root=null \
    showee_video_root=null showee_allintra_root=null \
    "datamodule.per_dataset_norm_stats_path=${REPO}/assets/per_dataset_norm_stats_repro_filtered_paper_alias.json" \
    datamodule.stride=600 \
    "batch_size=${PER_GPU_BATCH_SIZE}" \
    "val_batch_size=${PER_GPU_BATCH_SIZE}" \
    "num_workers=${NUM_WORKERS}" \
    "optimizer.lr=${BASE_LR}" \
    "lr_scheduler.scheduler.T_max=${MAX_EPOCHS}" \
    "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    train=true eval=false \
    'trainer.devices=[0,1,2,3,4,5,6,7]' "trainer.max_epochs=${MAX_EPOCHS}" \
    "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
    "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
    2>&1 | tee "${OUTPUT_ROOT}/console.log"
