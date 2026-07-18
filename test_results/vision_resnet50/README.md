# vision_resnet50

ResNet-50 vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ResNet-50 backbone → MLP head → 22 joint angles
- Checkpoint: epoch 099
- Config: `experiment/fusion/vision_resnet50`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.115 | 14.0 |
| user | right | 0.103 | 13.0 |
| gesture | left | 0.079 | 10.2 |
| gesture | right | 0.077 | 9.9 |
| both | left | 0.113 | 13.2 |
| both | right | 0.103 | 12.6 |

**Simple mean test_mae: 0.098**

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.078 ± 0.022 | 36 |
| User | 0.109 ± 0.039 | 6 |
| Both | 0.108 ± 0.013 | 5 |
