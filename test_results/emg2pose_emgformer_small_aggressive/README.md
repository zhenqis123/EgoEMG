# emg2pose_emgformer_small_aggressive

EMG2Pose dataset, small model (256d, 4 heads, 3 layers), aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/emg2pose_emgformer_small_aggressive_mae0.2264_epoch096.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_small_aggressive`
- **norm_mode**: per-dataset (training used null, but per-dataset is the correct benchmark)

## Results (per_user=true, user_stage split)

| generalization | test_mae | per_user_mae ± std | n |
|---|---|---|---|
| stage | 0.191 | 0.194 ± 0.021 | 158 |
| user | 0.215 | 0.217 ± 0.019 | 20 |
| user_stage | 0.215 | 0.217 ± 0.019 | 20 |

## Per-joint MAE (user_stage split, per-user mean ± std, n=20)

| Joint group | overall | per-user mean ± std |
|---|---|---|
| Proximal | 0.175 | 0.176 ± 0.016 |
| Thumb | 0.203 | 0.206 ± 0.026 |
| Index | 0.200 | 0.202 ± 0.026 |
| Middle | 0.209 | 0.212 ± 0.031 |
| Ring | 0.211 | 0.213 ± 0.023 |
| Distal | 0.246 | 0.249 ± 0.028 |
| Pinky | 0.253 | 0.253 ± 0.019 |
| Mid-phalanx | 0.266 | 0.268 ± 0.028 |
