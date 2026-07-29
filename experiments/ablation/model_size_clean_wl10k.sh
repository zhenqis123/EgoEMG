#!/bin/bash
# Clean WL=10K no-aug model size scaling on emg2pose_v3
# All 5 model sizes with identical hyperparameters for fair comparison.
set -e
cd ${EMG2POSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

EXPERIMENT="emgformer/regression_emg2pose"
GPUS="0,1,2,3,4,5"
WL=10000
STRIDE=5000
VAL_STRIDE=10000
BS=300
LR=0.0001
EPOCHS=150
SEED=42
LOG_BASE="logs/clean_wl10k"

# Model configurations: NAME MODEL_DIM NUM_HEADS NUM_LAYERS FFN_DIM
declare -a MODELS=(
    "middle:256:8:6:1024"
    "large:384:12:8:1536"
    "xlarge:512:8:8:2048"
    "xxlarge:640:10:10:2560"
    "huge:768:12:14:3072"
)

for MODEL_SPEC in "${MODELS[@]}"; do
    IFS=: read -r NAME DIM HEADS LAYERS FFN <<< "$MODEL_SPEC"
    TRIAL_DIR="${LOG_BASE}/${NAME}"

    echo "[$(date)] Model=${NAME}, dim=${DIM}, heads=${HEADS}, layers=${LAYERS}, ffn=${FFN}"

    python -m emg2pose.train \
      experiment=${EXPERIMENT} \
      transforms=emgformer_regression_no_aug \
      trainer.devices=[${GPUS}] \
      +trainer.strategy=ddp \
      trainer.max_epochs=${EPOCHS} \
      seed=${SEED} \
      hydra.run.dir=${TRIAL_DIR} \
      datamodule.window_length=${WL} \
      datamodule.stride=${STRIDE} \
      datamodule.val_test_window_length=${WL} \
      datamodule.val_test_stride=${VAL_STRIDE} \
      batch_size=${BS} \
      optimizer.lr=${LR} \
      module.decoder.model_dim=${DIM} \
      module.decoder.num_heads=${HEADS} \
      module.decoder.num_layers=${LAYERS} \
      module.decoder.ffn_dim=${FFN} \
      module.head.in_channels=${DIM}

    echo "[$(date)] ${NAME} done. Waiting 30s..."
    sleep 30
done

echo "[$(date)] ALL MODELS COMPLETE"
