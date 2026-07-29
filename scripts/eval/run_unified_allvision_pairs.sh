#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID BACKBONE"
  echo "BACKBONE: rn18 | rn50 | rn152 | vits | vitb | vitl"
  exit 2
fi

gpu_id="$1"
backbone="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
models_json="$repo_root/scripts/eval/unified_allvision_models.json"
out_dir="$repo_root/test_results/fusion/unified_center_allvision_20260722"

CUDA_VISIBLE_DEVICES="$gpu_id" python "$repo_root/scripts/eval/unified_center_eval.py" \
  --models-json "$models_json" \
  --model-names "fusion_${backbone}_s,vision_${backbone}" \
  --center-window-length 12000 \
  --output "$out_dir/${backbone}.json"
