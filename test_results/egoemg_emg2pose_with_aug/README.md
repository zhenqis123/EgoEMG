# egoemg_emg2pose_with_aug

EMG2Pose baseline (TDS + MLP) on EgoEMG, with rotation augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_emg2pose_with_aug_mae0.2733_epoch053.ckpt`
- **Config**: `experiment=emg2pose/regression_emg2pose_egoemg_with_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2595 | 0.2602 ± 0.0242 | 36 |
| user | 0.2875 | 0.2724 ± 0.0323 | 6 |
| both | 0.2973 | 0.2938 ± 0.0172 | 5 |
| **overall** | **0.2733** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2871 | 0.2696 ± 0.0371 (6) | 0.2851 ± 0.0450 (52) |
| user/right | 0.2881 | 0.2752 ± 0.0299 (6) | 0.2847 ± 0.0592 (52) |
| gesture/left | 0.2545 | 0.2552 ± 0.0262 (36) | 0.2260 ± 0.0694 (17) |
| gesture/right | 0.2646 | 0.2652 ± 0.0282 (36) | 0.2377 ± 0.0768 (17) |
| both/left | 0.2961 | 0.2919 ± 0.0220 (5) | 0.2865 ± 0.0556 (16) |
| both/right | 0.2980 | 0.2957 ± 0.0165 (5) | 0.2856 ± 0.0811 (16) |

Simple mean test_mae: **0.2814**
