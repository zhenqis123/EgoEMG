#!/usr/bin/env bash
# Single-A800 formal continuation for full EgoEMG + ShowEE RN152 fusion.
# The per-GPU batch size (480) was profiled from real RN152 forward/backward
# steps and leaves about 7.4 GiB of headroom on an 80 GiB A800.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
SHARED_ROOT=${EMG2POSE_SHARED_ROOT:-${EMG2POSE_SHARED_ROOT:-/shared}/develop}

first_existing_path() {
    local candidate
    for candidate in "$@"; do
        if [[ -e "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    # Preserve a useful error message from the required-path validation below.
    printf '%s\n' "$1"
}

resolve_python() {
    local candidate
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi

    if command -v python >/dev/null 2>&1 \
        && python -c 'import torch, pytorch_lightning, lmdb' >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    # Direct interpreter paths avoid relying on `conda init` inside a Pod.
    for candidate in \
        "${CONDA_PREFIX:-}/bin/python" \
        python \
        python \
        python \
        /root/miniconda3/envs/v2c_env/bin/python \
        /opt/conda/envs/emg2pose_env/bin/python; do
        if [[ -x "$candidate" ]] \
            && "$candidate" -c 'import torch, pytorch_lightning, lmdb' \
                >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "No usable training Python found. Set PYTHON_BIN=/path/to/python." >&2
    return 1
}

REPO=${EMG2POSE_REPO:-$SCRIPT_REPO}
INPUT_ROOT=${EMG2POSE_INPUT_ROOT:-$(first_existing_path \
    "${SHARED_ROOT}/experiment_inputs" data/experiment_inputs)}
EGOEMG_MEMMAP_DIR=${EGOEMG_MEMMAP_DIR:-$(first_existing_path \
    "${REPO}/data/EgoEMG_memmap" "${SHARED_ROOT}/data/EgoEMG_memmap")}
SHOWEE_MEMMAP_DIR=${SHOWEE_MEMMAP_DIR:-$(first_existing_path \
    "${REPO}/data/ShowEE_202607_memmap" "${SHARED_ROOT}/data/ShowEE_202607_memmap" \
    data/ShowEE_202607_memmap)}
EGOEMG_CROPS_DIR=${EGOEMG_CROPS_DIR:-$(first_existing_path \
    "${REPO}/data/EgoEMG_v2_crops" "${SHARED_ROOT}/data/EgoEMG_v2_crops" \
    data/EgoEMG_v2_crops)}
SHOWEE_CROPS_DIR=${SHOWEE_CROPS_DIR:-$(first_existing_path \
    "${REPO}/data/ShowEE_202607_crops" "${SHARED_ROOT}/data/ShowEE_202607_crops" \
    data/ShowEE_202607_crops)}
GPU_INDEX=${GPU_INDEX:-0}
BATCH_SIZE=${BATCH_SIZE:-480}
RUN_ROOT=${RUN_ROOT:-${SHARED_ROOT}/logs/full_showee_s_l_all_fusions_20260721}
SOURCE_CKPT=${SOURCE_CKPT:-$(first_existing_path \
    "${SHARED_ROOT}/experiment_inputs/fusion_checkpoints/fusion-rn152-epoch247.ckpt" \
    data/logs/full_showee_s_l_all_fusions_20260721/fusion_rn152_s/train/version_0/checkpoints/fusion-s-epoch=247-val_mae=0.0959.ckpt)}
STAGE=${STAGE:-single_a800_bs480_lr1e-5_eta1e-6_150e}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/fusion_rn152_s/${STAGE}}
BASE_LR=${BASE_LR:-1e-5}
ETA_MIN=${ETA_MIN:-1e-6}
ADDITIONAL_EPOCHS=${ADDITIONAL_EPOCHS:-150}

for required_path in \
    "$EGOEMG_MEMMAP_DIR" "$SHOWEE_MEMMAP_DIR" \
    "$EGOEMG_CROPS_DIR" "$SHOWEE_CROPS_DIR" "$INPUT_ROOT" "$SOURCE_CKPT"; do
    [[ -e "$required_path" ]] || {
        echo "Missing required path: $required_path" >&2
        exit 1
    }
done

PYTHON_BIN=$(resolve_python)

gpu_name=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=name \
    --format=csv,noheader | xargs)
gpu_memory_mib=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.total \
    --format=csv,noheader,nounits | xargs)
if [[ "$gpu_name" != *A800* || "$gpu_memory_mib" -lt 78000 ]]; then
    echo "Refusing to run: GPU $GPU_INDEX is '$gpu_name' (${gpu_memory_mib} MiB)," >&2
    echo "but this smoke script requires one A800 80GB." >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"

source_epoch=$("$PYTHON_BIN" - "$SOURCE_CKPT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["epoch"]))
PY
)
target_max_epochs=$((source_epoch + ADDITIONAL_EPOCHS + 1))
INITIAL_CKPT="${OUTPUT_ROOT}/initial_from_epoch_${source_epoch}.ckpt"

# A Lightning resume restores optimizer/scheduler states.  Create a private
# resume artifact with the requested fresh cosine schedule instead of mutating
# the selected best checkpoint.
"$PYTHON_BIN" - "$SOURCE_CKPT" "$INITIAL_CKPT" "$BASE_LR" "$ETA_MIN" "$ADDITIONAL_EPOCHS" <<'PY'
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

source, destination, base_lr, eta_min, total_epochs = sys.argv[1:]
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
base_lr = float(base_lr)
eta_min = float(eta_min)
total_epochs = int(total_epochs)

for optimizer in checkpoint.get("optimizer_states", []):
    for group in optimizer.get("param_groups", []):
        group["initial_lr"] = base_lr
        group["lr"] = base_lr

for scheduler in checkpoint.get("lr_schedulers", []):
    scheduler["T_max"] = total_epochs
    scheduler["eta_min"] = eta_min
    scheduler["base_lrs"] = [base_lr for _ in scheduler.get("base_lrs", [base_lr])]
    scheduler["last_epoch"] = 0
    scheduler["_step_count"] = 1
    scheduler["_last_lr"] = [base_lr for _ in scheduler["base_lrs"]]


def to_safe_container(value):
    """Remove OmegaConf objects rejected by PyTorch 2.6+ safe loading."""
    if isinstance(value, (DictConfig, ListConfig)):
        return to_safe_container(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {key: to_safe_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_safe_container(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_safe_container(item) for item in value)
    return value


# Lightning 2.6 delegates resume loading to torch.load(weights_only=True).
# The source checkpoint is trusted, but its OmegaConf metadata is not accepted
# by that loader.  This private continuation artifact keeps all train state
# while converting the metadata/optimizer containers to safe Python values.
checkpoint = to_safe_container(checkpoint)
Path(destination).parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, destination)

# Fail early with a clear message before launching a multi-day run.
torch.load(destination, map_location="cpu", weights_only=True)
PY

echo "Starting formal RN152 fusion continuation on $gpu_name (${gpu_memory_mib} MiB)"
echo "Per-GPU batch size: $BATCH_SIZE"
echo "Resume: epoch $source_epoch; additional epochs: $ADDITIONAL_EPOCHS; final epoch: $((target_max_epochs - 1))"
echo "Learning rate: $BASE_LR -> $ETA_MIN"

CUDA_VISIBLE_DEVICES=$GPU_INDEX "$PYTHON_BIN" -m egoemg.train \
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
    batch_size="$BATCH_SIZE" val_batch_size="$BATCH_SIZE" num_workers=10 \
    optimizer.lr="$BASE_LR" \
    "lr_scheduler.scheduler.T_max=${ADDITIONAL_EPOCHS}" \
    "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    "resume_ckpt=${INITIAL_CKPT}" \
    train=true eval=false \
    'trainer.devices=[0]' "trainer.max_epochs=${target_max_epochs}" \
    "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
    "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
    2>&1 | tee "${OUTPUT_ROOT}/console.log"
