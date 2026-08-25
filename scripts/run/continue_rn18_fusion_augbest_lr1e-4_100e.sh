#!/usr/bin/env bash
# Continue the fresh RN18+S best-augmentation run from its saved model and
# optimizer state, but reset the cosine schedule for another 100 epochs.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721/fusion_rn18_s
SOURCE_RUN=${SOURCE_RUN:-${RUN_ROOT}/fresh_augbest_lr5e-4_eta1e-6_250e}
SOURCE_CKPT=${SOURCE_CKPT:-${SOURCE_RUN}/train/version_0/checkpoints/last.ckpt}
STAGE=continue_lr1e-4_eta1e-6_100e
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/${STAGE}}
PREPARED_CKPT=${OUTPUT_ROOT}/resume_cosine_100.ckpt

INPUT_ROOT=data/experiment_inputs
BASE_LR=1e-4
ETA_MIN=1e-6
EXTRA_EPOCHS=100
VISIBLE_GPUS=2,3,4,5
TRAINER_GPUS='[0,1,2,3]'
BATCH_SIZE=200

[[ -f "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

read -r SOURCE_EPOCH TARGET_MAX_EPOCHS < <(
    python - "$SOURCE_CKPT" "$PREPARED_CKPT" "$BASE_LR" "$ETA_MIN" "$EXTRA_EPOCHS" <<'PY'
import sys
from pathlib import Path

import torch

source, destination, base_lr, eta_min, extra_epochs = sys.argv[1:]
base_lr = float(base_lr)
eta_min = float(eta_min)
extra_epochs = int(extra_epochs)
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
epoch = int(checkpoint["epoch"])

# Keep model and optimizer moments, but make the next scheduler step begin a
# new 100-epoch cosine at BASE_LR rather than inheriting the old phase.
for optimizer in checkpoint.get("optimizer_states", []):
    for group in optimizer.get("param_groups", []):
        group["initial_lr"] = base_lr
        group["lr"] = base_lr
for scheduler in checkpoint.get("lr_schedulers", []):
    scheduler.update(
        T_max=extra_epochs,
        eta_min=eta_min,
        base_lrs=[base_lr for _ in scheduler.get("base_lrs", [base_lr])],
        last_epoch=-1,
        _step_count=0,
        _last_lr=[base_lr for _ in scheduler.get("base_lrs", [base_lr])],
    )

torch.save(checkpoint, destination)
# Lightning resumes from epoch + 1, therefore max_epochs is an exclusive count.
print(epoch, epoch + 1 + extra_epochs)
PY
)

cd "$REPO"
echo "Continuing RN18 from epoch ${SOURCE_EPOCH} through epoch $((TARGET_MAX_EPOCHS - 1))"
echo "Cosine reset: ${BASE_LR} -> ${ETA_MIN} over ${EXTRA_EPOCHS} epochs"

CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
    python -m egoemg.train \
    experiment=fusion/fusion_allvision_s_egoemg_showee \
    +augmentation=batch_aug_best_v2 \
    module.vision_backbone_type=resnet18 \
    module.vision_embed_dim=512 \
    "module.vision_pretrained_checkpoint=${INPUT_ROOT}/vision_checkpoints/rn18.ckpt" \
    "pretrained_emg_checkpoint=${INPUT_ROOT}/emgformer_s_full_showee.ckpt" \
    "batch_size=${BATCH_SIZE}" "val_batch_size=${BATCH_SIZE}" \
    "optimizer.lr=${BASE_LR}" \
    "lr_scheduler.scheduler.T_max=${EXTRA_EPOCHS}" \
    "lr_scheduler.scheduler.eta_min=${ETA_MIN}" \
    "resume_ckpt=${PREPARED_CKPT}" \
    train=true eval=false \
    "trainer.devices=${TRAINER_GPUS}" "trainer.max_epochs=${TARGET_MAX_EPOCHS}" \
    "hydra.run.dir=${OUTPUT_ROOT}/hydra" \
    "logger.save_dir=${OUTPUT_ROOT}" logger.name=train logger.version=0 \
    2>&1 | tee "${OUTPUT_ROOT}/console.log"
