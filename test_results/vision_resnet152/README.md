# vision_resnet152

ResNet-152 vision-only baseline for EgoEMG hand pose estimation (single-frame, no EMG).

- Architecture: ResNet-152 backbone → MLP head → 22 joint angles
- Checkpoint: epoch 089
- Config: `experiment/fusion/vision_resnet152`
- Fusion mode: vision_only

## Results

| split | hand | test_mae | fingertip (mm) |
|-------|------|----------|----------------|
| user | left | 0.111 | 13.4 |
| user | right | 0.101 | 12.6 |
| gesture | left | 0.076 | 9.7 |
| gesture | right | 0.074 | 9.4 |
| both | left | 0.111 | 12.7 |
| both | right | 0.102 | 12.3 |

**Simple mean test_mae: 0.096**

ResNet-152 is the best-performing vision backbone on EgoEMG, outperforming all ViT variants including ViT-Large (0.102).

## Per-split summary (mean of left+right, ± std across users)

| Split | MAE (rad) | n_users |
|-------|-----------|---------|
| Gesture | 0.075 ± 0.021 | 36 |
| User | 0.106 ± 0.041 | 6 |
| Both | 0.106 ± 0.012 | 5 |
