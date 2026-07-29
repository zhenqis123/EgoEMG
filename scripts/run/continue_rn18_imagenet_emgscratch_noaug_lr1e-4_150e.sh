#!/usr/bin/env bash
# Continue the fresh no-augmentation RN18 control from its latest complete
# checkpoint, while resetting the LR schedule for 150 additional epochs.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=data/logs/full_showee_s_l_all_fusions_20260721/fusion_rn18_s
SOURCE_CKPT=${SOURCE_CKPT:-${RUN_ROOT}/fresh_imagenet_emgscratch_noaug_lr1e-4_eta5e-6_150e/train/version_0/checkpoints/fusion-s-epoch=096-val_mae=0.1046.ckpt}
STAGE=continue_lr1e-4_eta5e-6_150e
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/${STAGE}}
PREPARED_CKPT=${OUTPUT_ROOT}/resume_cosine_150.ckpt

BASE_LR=1e-4
ETA_MIN=5e-6
EXTRA_EPOCHS=150
VISIBLE_GPUS=1,2,3,4,5
TRAINER_GPUS='[0,1,2,3,4]'
BATCH_SIZE=200

[[ -f "$SOURCE_CKPT" ]] || { echo "Missing source checkpoint: $SOURCE_CKPT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

read -r SOURCE_EPOCH TARGET_MAX_EPOCHS < <(
    python - "$SOURCE_CKPT" "$PREPARED_CKPT" "$BASE_LR" "$ETA_MIN" "$EXTRA_EPOCHS" <<'PY'
import sys
import torch

source, destination, base_lr, eta_min, extra_epochs = sys.argv[1:]
base_lr, eta_min, extra_epochs = float(base_lr), float(eta_min), int(extra_epochs)
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
epoch = int(checkpoint["epoch"])
for optimizer in checkpoint.get("optimizer_states", []):
    for group in optimizer.get("param_groups", []):
        group["initial_lr"] = base_lr
        group["lr"] = base_lr
for scheduler in checkpoint.get("lr_schedulers", []):
    num_groups = len(scheduler.get("base_lrs", [base_lr]))
    scheduler.update(
        T_max=extra_epochs, eta_min=eta_min,
        base_lrs=[base_lr] * num_groups, last_epoch=-1, _step_count=0,
        _last_lr=[base_lr] * num_groups,
    )
torch.save(checkpoint, destination)
print(epoch, epoch + 1 + extra_epochs)
PY
)

cd "$REPO"
echo "Continuing epoch ${SOURCE_EPOCH} through $((TARGET_MAX_EPOCHS - 1)); cosine ${BASE_LR} -> ${ETA_MIN}"
CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS} \
    python -m emg2pose.train \
    experiment=fusion/fusion_rn18_s_imagenet_emgscratch_noaug \
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
