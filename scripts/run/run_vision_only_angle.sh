#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

python -m emg2pose.train_vision +experiment=vision/vision_only_angle
