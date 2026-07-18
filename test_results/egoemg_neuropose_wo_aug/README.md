# egoemg_neuropose_wo_aug

NeuroPose baseline (original architecture) on EgoEMG, no augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_neuropose_wo_aug_mae0.2784_epoch048.ckpt`
- **Config**: `experiment=emg2pose/regression_neuropose_egoemg`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2778 | 0.2762 ± 0.0205 | 36 |
| user | 0.2838 | 0.2736 ± 0.0227 | 6 |
| both | 0.2877 | 0.2852 ± 0.0117 | 5 |
| **overall** | **0.2809** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2825 | 0.2682 ± 0.0286 (6) | 0.2790 ± 0.0518 (52) |
| user/right | 0.2853 | 0.2790 ± 0.0196 (6) | 0.2824 ± 0.0464 (52) |
| gesture/left | 0.2808 | 0.2777 ± 0.0258 (36) | 0.2653 ± 0.0523 (17) |
| gesture/right | 0.2750 | 0.2747 ± 0.0208 (36) | 0.2552 ± 0.0624 (17) |
| both/left | 0.2804 | 0.2743 ± 0.0218 (5) | 0.2660 ± 0.0780 (16) |
| both/right | 0.2945 | 0.2958 ± 0.0142 (5) | 0.2869 ± 0.0648 (16) |

Simple mean test_mae: **0.2831**
