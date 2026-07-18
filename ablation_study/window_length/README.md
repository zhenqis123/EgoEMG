# Window Length Ablation Study

## Overview

This study investigates the effect of input window length on EMGFormer prediction accuracy for the EgoEMG hand pose estimation task. We sweep window lengths from 7,494 to 31,000 samples (3.7s–15.5s at 2kHz) and measure per-split generalization performance.

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Model | EMGFormer Middle (6.6M params) |
| Attention | Bidirectional (causal=False) |
| Dataset | EgoEMG (41 subjects, filtered EMG, 2kHz) |
| Val splits | gesture (seen user), user (unseen user), both |
| Featurizer | TDS (stride=50, left_context=510) |
| Search method | Optuna TPE (v3: 3k–15k, v4: 14k–25k) + grid sweep (25k–31k) |
| Training epochs | 150 |
| Augmentation | batch_aug.yaml (mag_warping + channel_mask disabled) |

## Key Results

- **Overall MAE**: Monotonically decreases from 0.259 (WL=7.5k) to 0.192 (WL=31k), saturating around WL=29k–31k.
- **Gesture split** (seen user, unseen gesture): Dramatic improvement from 0.226 → 0.102 (55% reduction).
- **User split** (unseen user): Flat at ~0.29, unaffected by window length.
- **Both split** (unseen user + gesture): Flat at ~0.29–0.31.

**Conclusion**: Longer windows help the model exploit temporal context for known users but do not improve cross-user generalization. The bottleneck for unseen users is inter-subject EMG variability (anatomy, electrode placement), not temporal modeling capacity.

## File Manifest

```
ablation_study/window_length/
├── README.md                  # This file
├── run_analysis.py            # Per-timestep MAE evaluation (generates .npy + summary.json)
├── generate_report.py         # HTML report generator (reads results/, outputs report.html)
├── report.html                # Self-contained visual report (dark theme, all charts embedded)
├── results/
│   ├── summary.json           # Trial metadata + per-timestep stats (15 entries)
│   ├── split_mae_results.json # Per-split MAE for all 15 trials
│   └── wl_*.npy              # Per-timestep MAE arrays (one per trial, 15 files)
└── paper_data/
    ├── table_overall.csv      # WL, duration, val_mae, test_mae, min/max/range, T_out
    ├── table_splits.csv       # WL, overall, gesture, user, both
    └── table_timestep_stats.csv # WL, mean/min/max MAE, positions, range
```

## Reproduction

```bash
# 1. Generate per-timestep MAE arrays (requires GPU + checkpoints)
python ablation_study/window_length/run_analysis.py

# 2. Generate HTML report
python ablation_study/window_length/generate_report.py

# 3. Per-split analysis (requires GPU + checkpoints)
# See /tmp/run_split_analysis.py for the batch script pattern
```

## Trials (15 total)

| WL | Duration | Val MAE | Gesture | User | Both | Study |
|----|----------|---------|---------|------|------|-------|
| 7,494 | 3.75s | 0.2586 | 0.2262 | 0.2935 | 0.3043 | optuna-v3 |
| 14,409 | 7.20s | 0.2428 | 0.2023 | 0.2871 | 0.2915 | optuna-v3 |
| 14,638 | 7.32s | 0.2401 | 0.1878 | 0.2996 | 0.3050 | optuna-v4 |
| 15,716 | 7.86s | 0.2330 | 0.1813 | 0.2915 | 0.2986 | optuna-v4 |
| 18,120 | 9.06s | 0.2255 | 0.1657 | 0.2928 | 0.3020 | optuna-v4 |
| 20,585 | 10.29s | 0.2177 | 0.1548 | 0.2924 | 0.2962 | optuna-v4 |
| 22,052 | 11.03s | 0.2156 | 0.1496 | 0.2917 | 0.3017 | optuna-v4 |
| 24,458 | 12.23s | 0.2090 | 0.1323 | 0.2957 | 0.3000 | optuna-v4 |
| 25,000 | 12.50s | 0.2033 | 0.1221 | 0.2960 | 0.3018 | sweep |
| 26,000 | 13.00s | 0.2018 | 0.1221 | 0.2934 | 0.3034 | sweep |
| 27,000 | 13.50s | 0.2057 | 0.1181 | 0.3022 | 0.3162 | sweep |
| 28,000 | 14.00s | 0.1977 | 0.1068 | 0.3004 | 0.3099 | sweep |
| 29,000 | 14.50s | 0.1927 | 0.1028 | 0.2983 | 0.3130 | sweep |
| 30,000 | 15.00s | 0.1927 | 0.1015 | 0.2993 | 0.2930 | sweep |
| 31,000 | 15.50s | 0.1925 | 0.1051 | 0.2926 | 0.2861 | sweep |
