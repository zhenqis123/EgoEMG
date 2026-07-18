#!/usr/bin/env bash
set -euo pipefail

python -m emg2pose.test_analysis_fusion \
    --config-name experiment/fusion/vision_resnet18 \
    --checkpoint logs/fusion/vision_resnet/version_9/checkpoints/resnet-vision-epoch=011-val_mae=0.1022.ckpt \
    --splits user gesture both \
    --hands left right \
    --batch-size 256 \
    --device cpu \
    --output results.csv
