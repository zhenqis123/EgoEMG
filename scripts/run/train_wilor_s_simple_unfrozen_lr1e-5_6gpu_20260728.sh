#!/usr/bin/env bash
# Train the final Table 4 ablation: unfrozen WiLoR + EMGFormer-S simple fusion.
set -Eeuo pipefail

REPO=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=python
RUN_DIR=${REPO}/logs/20260728/fusion_5vision_s_simple_unfrozen_augbest_30e/wilor_s

cd "${REPO}"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 "${PYTHON}" -m egoemg.train \
    experiment=fusion/fusion_wilor_s_simple_unfrozen_augbest_30e \
    batch_size=12 val_batch_size=12 \
    train=true eval=false trainer.devices='[0,1,2,3,4,5]' \
    trainer.max_epochs=30 \
    optimizer.lr=1.0e-5 \
    lr_scheduler.scheduler.T_max=30 \
    lr_scheduler.scheduler.eta_min=5.0e-6 \
    hydra.run.dir="${RUN_DIR}/hydra" \
    logger.save_dir="${RUN_DIR}" logger.name=train logger.version=0 \
    2>&1 | tee -a "${RUN_DIR}/console.log"

touch "${RUN_DIR}/DONE"
