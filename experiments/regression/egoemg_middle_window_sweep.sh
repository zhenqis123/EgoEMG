#!/bin/bash
# EgoEMG 窗口长度 sweep 实验
# 用法: bash experiments/regression/egoemg_middle_window_sweep.sh
set -e
cd ${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_egoemg"
GPUS="0,1,2,3,4,5"
LR=0.0005
EPOCHS=150
SEED=42
LOG_BASE="logs/regression/window_sweep"

# 批量大小自动缩放: bs = 500 * (7790/wl)^1.5, 限制在 [50, 750]
compute_bs() {
    local wl=$1
    python3 -c "
wl = $wl
ratio = 7790 / wl
bs = int(500 * ratio ** 1.5)
bs = max(50, min(750, bs))
print(bs)
"
}

for WL in 1000 3000 5000 7000 9000 11000 13000 15000 20000 25000 30000 35000; do
    BS=$(compute_bs $WL)
    STRIDE=$((WL / 10))
    [ $STRIDE -lt 100 ] && STRIDE=100
    TRIAL_DIR="${LOG_BASE}/wl_${WL}"

    echo "[$(date)] WL=${WL}, BS=${BS}, stride=${STRIDE}"

    python -m egoemg.train \
      experiment=${EXPERIMENT} \
      egoemg_unified_memmap_dir=${EGOEMG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_full_memmap \
      trainer.devices=[${GPUS}] \
      +trainer.strategy=ddp \
      trainer.max_epochs=${EPOCHS} \
      seed=${SEED} \
      hydra.run.dir=${TRIAL_DIR} \
      datamodule.window_length=${WL} \
      datamodule.val_test_window_length=${WL} \
      datamodule.stride=${STRIDE} \
      datamodule.val_test_stride=${WL} \
      batch_size=${BS} \
      optimizer.lr=${LR}

    echo "[$(date)] WL=${WL} done. Waiting 30s..."
    sleep 30
done

echo "[$(date)] ALL DONE"
