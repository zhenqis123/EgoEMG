#!/usr/bin/env bash
# Sequential full-data study:
#   1) EMGFormer-S, 2) EMGFormer-L,
#   3) seven existing vision baselines fused with the new full-data S model.
# Fusion training intentionally excludes EgoEMG_incre (no matched vision).
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-data/logs/full_showee_s_l_all_fusions_20260721}
INPUT_ROOT=data/experiment_inputs
VISION_ROOT=${INPUT_ROOT}/vision_checkpoints
GPU_LIST='[0,1,2,3,4,5]'

cd "$REPO"
mkdir -p "$RUN_ROOT" "$INPUT_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

run_job() {
    local name=$1
    shift
    local job_dir=${RUN_ROOT}/${name}
    local last_ckpt=${job_dir}/train/version_0/checkpoints/last.ckpt
    local resume_arg=()
    local max_epochs=150
    local target_last_epoch
    local arg
    for arg in "$@"; do
        if [[ $arg == trainer.max_epochs=* ]]; then
            max_epochs=${arg#trainer.max_epochs=}
        fi
    done
    target_last_epoch=$((max_epochs - 1))
    mkdir -p "$job_dir"
    if [[ -f ${job_dir}/DONE ]]; then
        if [[ -f $last_ckpt ]] && python - "$last_ckpt" "$target_last_epoch" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(
    0 if int(checkpoint.get("epoch", -1)) >= int(sys.argv[2]) else 1
)
PY
        then
            log "SKIP completed through epoch ${target_last_epoch}: ${name}"
            return
        fi
        rm -f "${job_dir}/DONE"
        log "INCOMPLETE ${name}: DONE exists but epoch 149 was not reached"
    fi
    if [[ -f $last_ckpt ]]; then
        local resume_ckpt=$last_ckpt
        if [[ $max_epochs -eq 250 ]]; then
            resume_ckpt=$(python - "$last_ckpt" <<'PY'
import math
import sys
from pathlib import Path

import torch

source = Path(sys.argv[1])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
epoch = int(checkpoint.get("epoch", -1))
schedulers = checkpoint.get("lr_schedulers", [])

# Migrate only the original 150-epoch cosine state. Checkpoints saved after
# migration already carry a different T_max and must not be reset again.
if schedulers and int(schedulers[0].get("T_max", -1)) == 150:
    state = schedulers[0]
    eta_min = float(state.get("eta_min", 1e-5))
    if epoch >= 149:
        # A completed 150-epoch run gets a lower-LR 100-epoch continuation.
        base_lr = 1e-4
        state.update(
            T_max=100,
            base_lrs=[base_lr for _ in state["base_lrs"]],
            last_epoch=0,
            _step_count=1,
            _last_lr=[base_lr for _ in state["base_lrs"]],
        )
        new_lrs = state["_last_lr"]
    else:
        # An early-stopped run keeps its phase while extending the original
        # cosine horizon to the new total of 250 epochs.
        state["T_max"] = 250
        phase = max(0, epoch + 1)
        new_lrs = [
            eta_min
            + (base_lr - eta_min) * (1 + math.cos(math.pi * phase / 250)) / 2
            for base_lr in state["base_lrs"]
        ]
        state["_last_lr"] = new_lrs
    for optimizer_state in checkpoint.get("optimizer_states", []):
        for group, lr in zip(optimizer_state.get("param_groups", []), new_lrs):
            group["lr"] = lr
    destination = source.with_name(f"resume_250_from_epoch_{epoch:03d}.ckpt")
    torch.save(checkpoint, destination)
    print(destination)
else:
    print(source)
PY
)
        fi
        resume_arg=("resume_ckpt=${resume_ckpt}")
        log "RESUME ${name} from ${resume_ckpt} to epoch ${target_last_epoch}"
    fi
    log "START ${name}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
        python -m egoemg.train "$@" "${resume_arg[@]}" \
        train=true eval=false \
        "trainer.devices=${GPU_LIST}" \
        "hydra.run.dir=${job_dir}/hydra" \
        "logger.save_dir=${job_dir}" \
        logger.name=train logger.version=0 \
        2>&1 | tee "${job_dir}/console.log"
    touch "${job_dir}/DONE"
    log "DONE ${name}"
}

best_checkpoint() {
    local root=$1
    python - "$root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.rglob("*.ckpt"):
    if path.name == "last.ckpt":
        continue
    match = re.search(r"val_mae=([0-9]+(?:\.[0-9]+)?)", path.name)
    if match:
        candidates.append((float(match.group(1)), path))
if not candidates:
    raise SystemExit(f"No metric checkpoint found below {root}")
print(min(candidates, key=lambda item: item[0])[1])
PY
}

preflight() {
    local name=$1
    shift
    log "PREFLIGHT ${name}"
    CUDA_VISIBLE_DEVICES=0 \
        python -m egoemg.train "$@" \
        train=true eval=false +trainer.fast_dev_run=true trainer.devices=[0] \
        num_workers=0 \
        "hydra.run.dir=${RUN_ROOT}/preflight/${name}" \
        "logger.save_dir=${RUN_ROOT}/preflight/${name}" \
        logger.name=smoke logger.version=0
}

fusion_args() {
    local name=$1
    local backbone=$2
    local dim=$3
    local vision_ckpt=$4
    local batch=$5
    local emg_ckpt=$6
    shift 6
    echo experiment=fusion/fusion_allvision_s_egoemg_showee
    echo +augmentation=batch_aug_best_v2
    echo "module.vision_backbone_type=${backbone}"
    echo "module.vision_embed_dim=${dim}"
    echo "module.vision_pretrained_checkpoint=${vision_ckpt}"
    echo "pretrained_emg_checkpoint=${emg_ckpt}"
    echo "batch_size=${batch}"
    echo "val_batch_size=${batch}"
    for arg in "$@"; do echo "$arg"; done
}

log "Run root: ${RUN_ROOT}"
if [[ ${SKIP_PREFLIGHT:-0} != 1 ]]; then
    log "Preflight: EMG S/L and all seven fusion model/data/forward paths"
    preflight emg_s experiment=emgformer/regression_egoemg_showee_small batch_size=4
    preflight emg_l experiment=emgformer/regression_egoemg_showee_large batch_size=2

# Fusion preflight does not need EMG initialization: the S architecture and
# every vision checkpoint are still instantiated and run end-to-end.
while IFS='|' read -r name backbone dim ckpt batch extra; do
    mapfile -t args < <(fusion_args "$name" "$backbone" "$dim" "$ckpt" 1 null $extra)
    preflight "$name" "${args[@]}"
done <<EOF
fusion_rn18_s|resnet18|512|${VISION_ROOT}/rn18.ckpt|200|trainer.max_epochs=250
fusion_rn50_s|resnet50|2048|${VISION_ROOT}/rn50.ckpt|120|trainer.max_epochs=250
fusion_rn152_s|resnet152|2048|${VISION_ROOT}/rn152.ckpt|40|trainer.max_epochs=250
fusion_vits_s|vit_small|384|${VISION_ROOT}/vits.ckpt|200|
fusion_vitb_s|vit_base|768|${VISION_ROOT}/vitb.ckpt|100|
fusion_vitl_s|vit_large|1024|${VISION_ROOT}/vitl.ckpt|32|
fusion_wilor_s|vit|1280|${VISION_ROOT}/wilor.ckpt|12|module.mano_model_path=data/WiLoR/mano_data
EOF
fi

log "All preflights passed; starting full sequential training"
run_job emgformer_s experiment=emgformer/regression_egoemg_showee_small

S_BEST=$(best_checkpoint "${RUN_ROOT}/emgformer_s")
ln -sfn "$S_BEST" "${INPUT_ROOT}/emgformer_s_full_showee.ckpt"
S_LINK=${INPUT_ROOT}/emgformer_s_full_showee.ckpt
log "EMGFormer-S initializer: ${S_BEST}"

run_job emgformer_l experiment=emgformer/regression_egoemg_showee_large

while IFS='|' read -r name backbone dim ckpt batch extra; do
    mapfile -t args < <(fusion_args "$name" "$backbone" "$dim" "$ckpt" "$batch" "$S_LINK" $extra)
    run_job "$name" "${args[@]}"
done <<EOF
fusion_rn18_s|resnet18|512|${VISION_ROOT}/rn18.ckpt|200|trainer.max_epochs=250
fusion_rn50_s|resnet50|2048|${VISION_ROOT}/rn50.ckpt|120|trainer.max_epochs=250
fusion_rn152_s|resnet152|2048|${VISION_ROOT}/rn152.ckpt|40|trainer.max_epochs=250
fusion_vits_s|vit_small|384|${VISION_ROOT}/vits.ckpt|200|
fusion_vitb_s|vit_base|768|${VISION_ROOT}/vitb.ckpt|100|
fusion_vitl_s|vit_large|1024|${VISION_ROOT}/vitl.ckpt|32|
fusion_wilor_s|vit|1280|${VISION_ROOT}/wilor.ckpt|12|module.mano_model_path=data/WiLoR/mano_data
EOF

log "ALL NINE EXPERIMENTS COMPLETED"
touch "${RUN_ROOT}/ALL_DONE"
