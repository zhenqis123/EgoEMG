# emg2pose_emgformer_large_aggressive

EMG2Pose dataset, large model (384d, 12 heads, 8 layers), aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/emg2pose_emgformer_large_aggressive_mae0.2267_epoch059.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_large_aggressive`
- **norm_mode**: per-dataset (matches training config)

## Results (per_user=true, user_stage split)

| generalization | test_mae | per_user_mae ± std | n |
|---|---|---|---|
| stage | 0.159 | 0.162 ± 0.019 | 158 |
| user | 0.213 | 0.214 ± 0.019 | 20 |
| user_stage | 0.213 | 0.214 ± 0.019 | 20 |

## Per-joint MAE (user_stage split, per-user mean ± std, n=20)

| Joint group | overall | per-user mean ± std |
|---|---|---|
| Proximal | 0.173 | 0.174 ± 0.017 |
| Thumb | 0.203 | 0.205 ± 0.029 |
| Index | 0.198 | 0.199 ± 0.025 |
| Middle | 0.205 | 0.207 ± 0.031 |
| Ring | 0.207 | 0.209 ± 0.022 |
| Distal | 0.246 | 0.249 ± 0.028 |
| Pinky | 0.250 | 0.249 ± 0.019 |
| Mid-phalanx | 0.258 | 0.259 ± 0.025 |
