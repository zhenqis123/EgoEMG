#!/usr/bin/env bash
# One-epoch low-LR probe for fully trainable WiLoR feature extraction + EMGFormer-S.
set -Eeuo pipefail

REPO=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=python
RUN_DIR=${REPO}/logs/20260728/wilor_s_unfrozen_lr1e-6_stride1560_repeat1_1epoch

cd "${REPO}"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "${PYTHON}" -m egoemg.train \
    experiment=fusion/fusion_wilor_s_simple_unfrozen_augbest_30e \
    batch_size=12 val_batch_size=12 \
    datamodule.stride=1560 datamodule.dataset_repeat=1 \
    train=true eval=false trainer.devices='[0,1,2,3,4,5]' \
    trainer.max_epochs=1 \
    optimizer.lr=1.0e-6 \
    lr_scheduler.scheduler.T_max=1 \
    lr_scheduler.scheduler.eta_min=1.0e-6 \
    hydra.run.dir="${RUN_DIR}/hydra" \
    logger.save_dir="${RUN_DIR}" logger.name=train logger.version=0 \
    2>&1 | tee -a "${RUN_DIR}/console.log"

touch "${RUN_DIR}/DONE"
