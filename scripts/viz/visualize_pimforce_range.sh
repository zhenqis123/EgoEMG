#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/visualize_pimforce_range.sh --root ROOT --start IDX --end IDX --out-dir DIR [options]

Required:
  --root       Root dir of processed_raw (contains pimforce_metadata.csv)
  --start      Start index (inclusive)
  --end        End index (inclusive)
  --out-dir    Output directory for HTML files

Optional:
  --target-fs  Target visualization frame rate (default: 10)
  --stop       Stop index in samples (default: -1, i.e. to end)
  --stride     Custom index stride (default: 1)

Example:
  scripts/visualize_pimforce_range.sh \
    --root data/emg_corpus/PiMforce/processed_raw \
    --start 0 --end 20 --out-dir /tmp/pimforce_vis \
    --target-fs 10 --stop 1000000
USAGE
}

root=""
start=""
end=""
out_dir=""
target_fs=10
stop=-1
stride=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      root="$2"
      shift 2
      ;;
    --start)
      start="$2"
      shift 2
      ;;
    --end)
      end="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --target-fs)
      target_fs="$2"
      shift 2
      ;;
    --stop)
      stop="$2"
      shift 2
      ;;
    --stride)
      stride="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$root" || -z "$start" || -z "$end" || -z "$out_dir" ]]; then
  usage
  exit 1
fi

mkdir -p "$out_dir"

for ((idx=start; idx<=end; idx+=stride)); do
  out_path="$out_dir/vis_${idx}.html"
  echo "[${idx}] -> ${out_path}"
  python scripts/legacy/visualize_emg2pose_dataset.py \
    --dataset pimforce_raw \
    --pimforce-root "$root" \
    --index "$idx" \
    --output "$out_path" \
    --target-fs "$target_fs" \
    --stop "$stop"
done
