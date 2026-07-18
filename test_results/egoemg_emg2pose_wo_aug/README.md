# egoemg_emg2pose_wo_aug

EMG2Pose baseline (TDS + MLP) on EgoEMG, no augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_emg2pose_wo_aug_mae0.2753_epoch021.ckpt`
- **Config**: `experiment=emg2pose/regression_emg2pose_egoemg`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2695 | 0.2701 ± 0.0230 | 36 |
| user | 0.2813 | 0.2582 ± 0.0507 | 6 |
| both | 0.2858 | 0.2843 ± 0.0103 | 5 |
| **overall** | **0.2753** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2803 | 0.2554 ± 0.0511 (6) | 0.2768 ± 0.0553 (52) |
| user/right | 0.2827 | 0.2610 ± 0.0522 (6) | 0.2795 ± 0.0490 (52) |
| gesture/left | 0.2653 | 0.2657 ± 0.0245 (36) | 0.2392 ± 0.0730 (17) |
| gesture/right | 0.2735 | 0.2745 ± 0.0249 (36) | 0.2492 ± 0.0756 (17) |
| both/left | 0.2860 | 0.2819 ± 0.0210 (5) | 0.2756 ± 0.0732 (16) |
| both/right | 0.2854 | 0.2867 ± 0.0103 (5) | 0.2778 ± 0.0633 (16) |

Simple mean test_mae: **0.2789**
