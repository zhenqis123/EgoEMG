#!/bin/bash
# Re-evaluate EMG models (emg2pose + egoemg) with per-user / per-gesture stats.
#
# Usage:  bash scripts/eval/run_test_analysis_emg.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=============================================="
echo "EMG test_analysis re-evaluation (per-group stats)"
echo "Single GPU forced by evaluate()"
echo "=============================================="

# ── emg2pose dataset (per_user) ──────────────────────────────────────────
echo ""
echo "===== EMG2Pose dataset: small_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_small_aggressive \
  checkpoint="$ROOT/test_results/emg2pose_small_aggressive/checkpoints/best.ckpt" \
  +per_user=true \
  hydra.run.dir="$ROOT/test_results/emg2pose_small_aggressive"

echo ""
echo "===== EMG2Pose dataset: middle_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_middle_aggressive \
  checkpoint="$ROOT/test_results/emg2pose_middle_aggressive/checkpoints/best.ckpt" \
  +per_user=true \
  hydra.run.dir="$ROOT/test_results/emg2pose_middle_aggressive"

echo ""
echo "===== EMG2Pose dataset: large_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_large_aggressive \
  checkpoint="$ROOT/test_results/emg2pose_large_aggressive/checkpoints/best.ckpt" \
  +per_user=true \
  hydra.run.dir="$ROOT/test_results/emg2pose_large_aggressive"

# ── EgoEMG dataset (per_group_stats) ─────────────────────────────────────
echo ""
echo "===== EgoEMG dataset: small_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_small_aggressive_egoemg \
  checkpoint="$ROOT/test_results/egoemg_small_best.ckpt" \
  +per_group_stats=true \
  hydra.run.dir="$ROOT/test_results/egoemg_small_aggressive_with_aug"

echo ""
echo "===== EgoEMG dataset: middle_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_middle_aggressive_egoemg \
  checkpoint="$ROOT/test_results/egoemg_middle_best.ckpt" \
  +per_group_stats=true \
  hydra.run.dir="$ROOT/test_results/egoemg_middle_aggressive_with_aug"

echo ""
echo "===== EgoEMG dataset: large_aggressive ====="
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_large_aggressive_egoemg \
  checkpoint="$ROOT/test_results/egoemg_large_best.ckpt" \
  +per_group_stats=true \
  hydra.run.dir="$ROOT/test_results/egoemg_large_aggressive_with_aug"

echo ""
echo "All EMG evaluations done."
