#!/bin/bash
# 窗口长度消融实验 (ablation study)
# 从 1k 到 35k 的完整 sweep，用于论文
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_egoemg"
GPUS="0,1,2,3,4,5"
LR=0.0005
EPOCHS=150
SEED=42
LOG_BASE="logs/ablation/window_length"

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

# 可自定义 sweep 范围
WL_LIST="${1:-1000 3000 5000 7000 9000 11000 13000 15000 18000 20000 22000 25000 28000 30000 33000 35000}"

for WL in $WL_LIST; do
    BS=$(compute_bs $WL)
    STRIDE=$((WL / 10))
    [ $STRIDE -lt 100 ] && STRIDE=100
    TRIAL_DIR="${LOG_BASE}/wl_${WL}"

    echo "[$(date)] WL=${WL}, BS=${BS}, stride=${STRIDE}"
    echo "[$(date)] Output: ${TRIAL_DIR}"

    python -m egoemg.train \
      experiment=${EXPERIMENT} \
      egoemg_memmap_dir=${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/data/EgoEMG_memmap \
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

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date)] WL=${WL} FAILED (rc=${RC})"
    else
        echo "[$(date)] WL=${WL} DONE"
    fi

    echo "[$(date)] Waiting 30s for GPU memory release..."
    sleep 30
done

echo "[$(date)] ALL EXPERIMENTS COMPLETE"
