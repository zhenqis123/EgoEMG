# egoemg_neuropose_with_aug

NeuroPose baseline (original architecture) on EgoEMG, with rotation augmentation.

- **Eval date**: 2026-05-05
- **Checkpoint**: `checkpoints/egoemg_neuropose_with_aug_mae0.2775_epoch091.ckpt`
- **Config**: `experiment=emg2pose/regression_neuropose_egoemg_with_aug`

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.2767 | 0.2760 ± 0.0204 | 36 |
| user | 0.2805 | 0.2637 ± 0.0344 | 6 |
| both | 0.2825 | 0.2784 ± 0.0155 | 5 |
| **overall** | **0.2786** | | |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.2786 | 0.2659 ± 0.0256 (6) | 0.2750 ± 0.0540 (52) |
| user/right | 0.2827 | 0.2616 ± 0.0454 (6) | 0.2799 ± 0.0534 (52) |
| gesture/left | 0.2785 | 0.2766 ± 0.0251 (36) | 0.2599 ± 0.0591 (17) |
| gesture/right | 0.2750 | 0.2753 ± 0.0210 (36) | 0.2525 ± 0.0727 (17) |
| both/left | 0.2781 | 0.2726 ± 0.0202 (5) | 0.2638 ± 0.0703 (16) |
| both/right | 0.2866 | 0.2842 ± 0.0115 (5) | 0.2774 ± 0.0639 (16) |

Simple mean test_mae: **0.2799**
