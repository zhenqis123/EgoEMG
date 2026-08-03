#!/usr/bin/env bash
set -euo pipefail

python -m egoemg.train experiment=fusion/vision_cached_angle
