#!/usr/bin/env bash
set -euo pipefail

python -m emg2pose.train experiment=fusion/vision_cached_angle
