# emg2pose_emgformer_middle_aggressive

EMG2Pose dataset, middle model (256d, 8 heads, 6 layers), aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/emg2pose_emgformer_middle_aggressive_mae0.2269_epoch088.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_middle_aggressive`
- **norm_mode**: per-dataset (matches training config)

## Results (per_user=true, user_stage split)

| generalization | test_mae | per_user_mae ± std | n |
|---|---|---|---|
| stage | 0.175 | 0.178 ± 0.020 | 158 |
| user | 0.215 | 0.216 ± 0.019 | 20 |
| user_stage | 0.215 | 0.216 ± 0.019 | 20 |

## Per-joint MAE (user_stage split, per-user mean ± std, n=20)

| Joint group | overall | per-user mean ± std |
|---|---|---|
| Proximal | 0.176 | 0.176 ± 0.017 |
| Thumb | 0.206 | 0.208 ± 0.026 |
| Index | 0.200 | 0.201 ± 0.025 |
| Middle | 0.207 | 0.209 ± 0.031 |
| Ring | 0.210 | 0.211 ± 0.023 |
| Distal | 0.246 | 0.248 ± 0.028 |
| Pinky | 0.252 | 0.252 ± 0.020 |
| Mid-phalanx | 0.263 | 0.264 ± 0.027 |
