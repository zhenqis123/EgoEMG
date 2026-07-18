# egoemg_vemg2pose_with_aug

vEMG2Pose baseline (original architecture) on EgoEMG, with rotation augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_vemg2pose_with_aug_mae0.2740_epoch090.ckpt`
- **Config**: `experiment=emg2pose/regression_vemg2pose_egoemg_with_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2655 | 0.2669 ± 0.0258 | 36 |
| user | 0.2842 | 0.2601 ± 0.0485 | 6 |
| both | 0.2848 | 0.2796 ± 0.0183 | 5 |
| **overall** | **0.2743** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2854 | 0.2593 ± 0.0506 (6) | 0.2825 ± 0.0451 (52) |
| user/right | 0.2834 | 0.2610 ± 0.0484 (6) | 0.2796 ± 0.0625 (52) |
| gesture/left | 0.2612 | 0.2628 ± 0.0280 (36) | 0.2344 ± 0.0744 (17) |
| gesture/right | 0.2698 | 0.2709 ± 0.0284 (36) | 0.2468 ± 0.0793 (17) |
| both/left | 0.2810 | 0.2749 ± 0.0221 (5) | 0.2712 ± 0.0691 (16) |
| both/right | 0.2883 | 0.2841 ± 0.0172 (5) | 0.2788 ± 0.0742 (16) |

Simple mean test_mae: **0.2782**
