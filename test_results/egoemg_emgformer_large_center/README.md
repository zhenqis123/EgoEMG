# egoemg_emgformer_large_center

EMGFormer Large (384d, 8 heads, 8 layers) on EgoEMG, center-frame supervised, from pretrained checkpoint.

- **Eval date**: 2026-05-06
- **Checkpoint**: `checkpoints/egoemg_emgformer_large_center_mae0.2556_epoch002.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_large_aggressive_egoemg` + `center_supervised=true` + `center_target_only=true`
- **Pretrained from**: full-window large checkpoint

Note: Only trained for 2 epochs, far from converged.

## Per-hand per-group stats (6 splits)

| split | hand | samples | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|---|---|
| user | left | 830 | 0.2891 | 0.2755 ± 0.0267 (6) | 0.2844 ± 0.0490 (52) |
| user | right | 830 | 0.3002 | 0.2856 ± 0.0317 (6) | 0.2967 ± 0.0587 (52) |
| gesture | left | 1162 | 0.2152 | 0.2161 ± 0.0274 (36) | 0.1544 ± 0.1088 (17) |
| gesture | right | 1162 | 0.2255 | 0.2243 ± 0.0290 (36) | 0.1616 ± 0.1081 (17) |
| both | left | 161 | 0.3026 | 0.2962 ± 0.0266 (5) | 0.2845 ± 0.0799 (16) |
| both | right | 161 | 0.3116 | 0.3114 ± 0.0092 (5) | 0.3020 ± 0.0800 (16) |

Sample-weighted mean test_mae: **0.2555**
