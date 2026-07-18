# Cross-Modal Comparison on EgoEMG

All results on EgoEMG test splits. Latest CSVs as of 2026-05-06.

## Evaluation Protocols

| Modality | Protocol | Description |
|----------|----------|-------------|
| EMG-only | **All-time-step** | MAE averaged over all ~75 time steps per window |
| Vision-only | **Center-frame** | MAE on the single center frame (where vision is available) |
| Fusion | **Center-frame** | MAE on the single center frame (`center_supervised` mode) |

> EMG all-time-step and vision/fusion center-frame numbers are **not directly comparable** due to different evaluation granularity. Center-frame evaluation is generally easier (lower MAE). For protocol-matched comparison, see center-frame rows below.

## Overall Summary (center-frame, protocol-matched)

| Method | Params | Gesture | User | Both | Simple Mean |
|--------|--------|---------|------|------|-------------|
| EMG-Middle (center-frame) | 6.6M | -- | -- | -- | -- |
| Vision-ResNet18 | 11.5M | 0.089\,$\pm$\,0.023 | 0.118\,$\pm$\,0.030 | 0.118\,$\pm$\,0.015 | 0.108 |
| Vision-ResNet50 | 23.5M | 0.078\,$\pm$\,0.022 | 0.109\,$\pm$\,0.039 | 0.108\,$\pm$\,0.013 | 0.098 |
| Vision-ResNet152 | 58.2M | **0.075**\,$\pm$\,**0.021** | **0.106**\,$\pm$\,**0.041** | **0.106**\,$\pm$\,**0.012** | **0.096** |
| Vision-ViT-Small | 21.9M | 0.087\,$\pm$\,0.023 | 0.127\,$\pm$\,0.042 | 0.125\,$\pm$\,0.009 | 0.113 |
| Vision-ViT-Base | 86.2M | 0.084\,$\pm$\,0.023 | 0.122\,$\pm$\,0.040 | 0.120\,$\pm$\,0.011 | 0.109 |
| Vision-ViT-Large | 303.8M | 0.076\,$\pm$\,0.023 | 0.116\,$\pm$\,0.038 | 0.116\,$\pm$\,0.013 | 0.102 |
| Fusion (ResNet18+EMG-Small) | 17.3M | 0.086 | 0.116 | 0.115 | 0.106 |
| **Vision-WiLoR (ViT-Huge)** | **631.6M** | **0.072**\,$\pm$\,**0.022** | **0.098**\,$\pm$\,**0.044** | **0.099**\,$\pm$\,**0.012** | **0.090** |

> ResNet checkpoints: resnet18-011, resnet50-099, resnet152-089. ViT checkpoints: vit-small-179, vit-base-123, vit-large-127. WiLoR checkpoint: wilor-vit-epoch=011-val_mae=0.0840. All vision-only, center-frame evaluation. Metrics are simple mean of left+right per-split MAE. $\pm$ values are standard deviation across users (gesture n=36, user n=6, both n=5).

## Per-Split Detail

### Gesture Split (held-out gestures, seen users)

| Method | Hand | MAE (rad) | Fingertip (mm) |
|--------|------|-----------|-----------------|
| **EMG-only (all-time-step)** | | | |
| EMG-Small | left | 0.234 | 37.9 |
| EMG-Small | right | 0.254 | 42.4 |
| EMG-Middle | left | 0.220 | 35.4 |
| EMG-Middle | right | 0.242 | 39.9 |
| EMG-Large | left | 0.221 | 35.7 |
| EMG-Large | right | 0.242 | 39.9 |
| **Vision-only (center-frame)** | | | |
| ResNet18 | left | 0.090 | 11.7 |
| ResNet18 | right | 0.087 | 11.3 |
| ResNet50 | left | 0.079 | 10.2 |
| ResNet50 | right | 0.077 | 9.9 |
| ResNet152 | left | **0.076** | **9.7** |
| ResNet152 | right | **0.074** | **9.4** |
| ViT-Small | left | 0.088 | 11.6 |
| ViT-Small | right | 0.086 | 11.2 |
| ViT-Base | left | 0.085 | 11.0 |
| ViT-Base | right | 0.083 | 10.8 |
| ViT-Large | left | 0.076 | 9.9 |
| ViT-Large | right | 0.075 | 9.7 |
| **WiLoR (ViT-Huge)** | left | **0.073** | 8.8 |
| WiLoR-Scratch (ViT-Huge) | left | 0.114 | 15.4 |
| **WiLoR (ViT-Huge)** | right | **0.070** | 8.5 |
| WiLoR-Scratch (ViT-Huge) | right | 0.109 | 14.5 |
| **Fusion (center-frame)** | | | |
| ResNet18+EMG-Small | left | 0.088 | 11.3 |
| ResNet18+EMG-Small | right | 0.085 | 11.0 |

### User Split (held-out users, seen gestures)

| Method | Hand | MAE (rad) | Fingertip (mm) |
|--------|------|-----------|-----------------|
| **EMG-only (all-time-step)** | | | |
| EMG-Small | left | 0.275 | 44.6 |
| EMG-Small | right | 0.287 | 49.5 |
| EMG-Middle | left | 0.296 | 47.7 |
| EMG-Middle | right | 0.299 | 51.2 |
| EMG-Large | left | 0.291 | 47.0 |
| EMG-Large | right | 0.297 | 50.7 |
| **Vision-only (center-frame)** | | | |
| ResNet18 | left | 0.122 | 15.3 |
| ResNet18 | right | 0.113 | 14.6 |
| ResNet50 | left | 0.115 | 14.0 |
| ResNet50 | right | 0.103 | 13.0 |
| ResNet152 | left | **0.111** | **13.4** |
| ResNet152 | right | **0.101** | **12.6** |
| ViT-Small | left | 0.133 | 16.4 |
| ViT-Small | right | 0.122 | 15.5 |
| ViT-Base | left | 0.127 | 15.3 |
| ViT-Base | right | 0.116 | 14.8 |
| ViT-Large | left | 0.121 | 14.4 |
| ViT-Large | right | 0.111 | 13.6 |
| **WiLoR (ViT-Huge)** | left | **0.100** | 11.3 |
| WiLoR-Scratch (ViT-Huge) | left | 0.154 | 19.2 |
| **WiLoR (ViT-Huge)** | right | **0.097** | 11.2 |
| WiLoR-Scratch (ViT-Huge) | right | 0.142 | 18.9 |
| **Fusion (center-frame)** | | | |
| ResNet18+EMG-Small | left | 0.124 | 15.3 |
| ResNet18+EMG-Small | right | 0.108 | 14.2 |

### Both Split (held-out users × held-out gestures)

| Method | Hand | MAE (rad) | Fingertip (mm) |
|--------|------|-----------|-----------------|
| **EMG-only (all-time-step)** | | | |
| EMG-Small | left | 0.281 | 45.6 |
| EMG-Small | right | 0.296 | 50.1 |
| EMG-Middle | left | 0.311 | 49.7 |
| EMG-Middle | right | 0.308 | 51.4 |
| EMG-Large | left | 0.311 | 50.5 |
| EMG-Large | right | 0.315 | 52.7 |
| **Vision-only (center-frame)** | | | |
| ResNet18 | left | 0.123 | 14.9 |
| ResNet18 | right | 0.114 | 13.7 |
| ResNet50 | left | 0.113 | 13.2 |
| ResNet50 | right | 0.103 | 12.6 |
| ResNet152 | left | **0.111** | **12.7** |
| ResNet152 | right | **0.102** | **12.3** |
| ViT-Small | left | 0.130 | 15.8 |
| ViT-Small | right | 0.121 | 14.7 |
| ViT-Base | left | 0.126 | 15.2 |
| ViT-Base | right | 0.115 | 14.1 |
| ViT-Large | left | 0.119 | 14.0 |
| ViT-Large | right | 0.112 | 13.4 |
| **WiLoR (ViT-Huge)** | left | **0.101** | 11.2 |
| WiLoR-Scratch (ViT-Huge) | left | 0.153 | 19.0 |
| **WiLoR (ViT-Huge)** | right | **0.096** | 10.5 |
| WiLoR-Scratch (ViT-Huge) | right | 0.141 | 18.1 |
| **Fusion (center-frame)** | | | |
| ResNet18+EMG-Small | left | 0.120 | 14.5 |
| ResNet18+EMG-Small | right | 0.109 | 13.4 |

## Fusion Gain Analysis (center-frame)

| Split | Vision-ResNet18 MAE | Fusion MAE | Δ MAE | Δ% |
|-------|---------------------|-----------|-------|-----|
| gesture | 0.089 | 0.086 | −0.003 | −3.4% |
| user | 0.118 | 0.116 | −0.002 | −1.7% |
| both | 0.118 | 0.115 | −0.003 | −2.5% |
| **Mean** | **0.108** | **0.106** | **−0.002** | **−1.9%** |

Fusion provides modest but consistent improvement over ResNet-18 across all splits. Gains are small because the vision baseline itself is strong on this dataset.

## Vision Backbone Scaling

| Backbone | Params | Gesture | User | Both | Simple Mean | vs ResNet18 |
|----------|--------|---------|------|------|-------------|-------------|
| ResNet18 | 11.5M | 0.089\,$\pm$\,0.023 | 0.118\,$\pm$\,0.030 | 0.118\,$\pm$\,0.015 | 0.108 | -- |
| ViT-Small | 21.9M | 0.087\,$\pm$\,0.023 | 0.127\,$\pm$\,0.042 | 0.125\,$\pm$\,0.009 | 0.113 | +4.6% |
| ResNet50 | 23.5M | 0.078\,$\pm$\,0.022 | 0.109\,$\pm$\,0.039 | 0.108\,$\pm$\,0.013 | 0.098 | −9.3% |
| ResNet152 | 58.2M | **0.075**\,$\pm$\,**0.021** | **0.106**\,$\pm$\,**0.041** | **0.106**\,$\pm$\,**0.012** | **0.096** | **−11.1%** |
| ViT-Base | 86.2M | 0.084\,$\pm$\,0.023 | 0.122\,$\pm$\,0.040 | 0.120\,$\pm$\,0.011 | 0.109 | +0.9% |
| ViT-Large | 303.8M | 0.076\,$\pm$\,0.023 | 0.116\,$\pm$\,0.038 | 0.116\,$\pm$\,0.013 | 0.102 | −5.6% |
| **WiLoR (ViT-Huge)** | **631.6M** | **0.072**\,$\pm$\,**0.022** | **0.098**\,$\pm$\,**0.044** | **0.099**\,$\pm$\,**0.012** | **0.090** | **−16.7%** |
| WiLoR-Scratch (ViT-Huge) | 631.6M | 0.111\,$\pm$\,0.034 | 0.148\,$\pm$\,0.043 | 0.147\,$\pm$\,0.036 | 0.128 | +18.5% |

> **Takeaway**: WiLoR ViT-Huge (hand-specialized, ViTPose→WiLoR pretrained) dominates all generic backbones. ResNet-152 achieves the best overall performance (simple-mean 0.096), outperforming ViT-Large (0.102) despite using 5× fewer parameters. Within each family, scaling up consistently helps. The user split benefits most from scaling: from 0.118 (ResNet18) to 0.106 (ResNet152). $\pm$ values are standard deviation across users (gesture n=36, user n=6, both n=5).

## ResNet152 Vision-Only vs Fusion

| Split | ResNet152 MAE | Fusion (ResNet18+EMG) MAE | Δ |
|-------|---------------|--------------------------|---|
| gesture | **0.075** | 0.086 | −13% |
| user | **0.106** | 0.116 | −9% |
| both | **0.106** | 0.115 | −8% |
| **Mean** | **0.096** | 0.106 | −9% |

ResNet152 vision-only outperforms ResNet18+EMG fusion by 9% overall. This suggests that improving the vision backbone is a more effective path to better center-frame performance than adding EMG to a weak vision model.

## EMG Contribution (Delta Pathway)

Measured on version_9 fusion checkpoint (ResNet18 + EMGFormer Small, `center_supervised`, all components trainable):

| Metric | Value |
|--------|-------|
| ‖Δy_emg‖ / ‖pred‖ mean | 25.3% |
| ‖Δy_emg‖ / ‖pred‖ median | 25.6% |
| ‖Δy_emg‖ / ‖y_v‖ | 31.4% |
| ‖y_v‖ mean | 0.225 |
| ‖Δy_emg‖ mean | 0.065 |

EMG contributes ~25% of final prediction magnitude; vision provides ~75%. EMG residual is a refinement, not the dominant signal.

## Checkpoints

| Experiment | Best Checkpoint | val_mae | Path |
|------------|-----------------|---------|------|
| EMG-Small (EMG2Pose) | best.ckpt | 0.2264 | `test_results/emg2pose_small_aggressive/checkpoints/` |
| EMG-Middle (EMG2Pose) | best.ckpt | 0.2269 | `test_results/emg2pose_middle_aggressive/checkpoints/` |
| EMG-Large (EMG2Pose) | best.ckpt | 0.2267 | `test_results/emg2pose_large_aggressive/checkpoints/` |
| Vision-ResNet18 | epoch=011 | 0.1025 | `logs/fusion/vision_resnet/version_9/checkpoints/` |
| Vision-ResNet50 | epoch=099 | -- | `logs/fusion/vision_resnet50/` |
| Vision-ResNet152 | epoch=089 | -- | `logs/fusion/vision_resnet152/` |
| Vision-ViT-Small | epoch=179 | 0.1053 | `logs/fusion/vision_vit_small/version_0/checkpoints/` |
| Vision-ViT-Base | epoch=123 | 0.1010 | `logs/fusion/vision_vit_base/version_0/checkpoints/` |
| Vision-ViT-Large | epoch=127 | 0.0940 | `logs/fusion/vision_vit_large/version_0/checkpoints/` |
| Fusion-ResNet18+EMG-Small | epoch=198 | 0.0984 | `test_results/fusion/checkpoints/best.ckpt` |
| Vision-WiLoR (ViT-Huge) | epoch=011 | 0.0840 | `logs_temp/fusion/vision_wilor_vit/version_2/checkpoints/wilor-vit-epoch=011-val_mae=0.0840.ckpt` |
| Vision-WiLoR-Scratch (ViT-Huge) | epoch=044 | 0.1281 | `logs_temp/fusion/vision_wilor_vit_scratch/version_0/checkpoints/wilor-vit-scratch-epoch=044-val_mae=0.1281.ckpt` |

## Key Findings

1. **Cross-user generalization is the dominant challenge** for all modalities. EMG degrades from ~0.23 (gesture) to ~0.29 (user), vision from 0.07 to 0.11 (ResNet-152), fusion from 0.09 to 0.12.

2. **ResNet > ViT for this task**: ResNet-152 (58.2M, 0.096) outperforms ViT-Large (303.8M, 0.102) by 6%, and ResNet-50 (23.5M, 0.098) outperforms ViT-Base (86.2M, 0.109) by 10%. This may reflect the limited dataset size (~10 hrs) favoring architectures with stronger inductive biases.

3. **Fusion > Vision > EMG** on center-frame MAE, but EMG operates on all time steps while vision/fusion are center-frame only.

4. **Vision backbone quality dominates fusion performance**: ResNet152 vision-only (0.096) clearly outperforms ResNet18+EMG fusion (0.106), indicating that improving the vision backbone is more impactful than adding EMG to a weak vision model for center-frame prediction.

5. **ViT-Large vision-only matches ResNet18+EMG fusion**: both achieve similar user split MAE (~0.116), but ResNet152 vision-only (0.106) wins convincingly.

6. **Smaller EMG models generalize better** to unseen users: EMG-Small user MAE (0.281) < EMG-Middle (0.298) < EMG-Large (0.294). This capacity-overfitting tradeoff is consistent across splits.

7. **Right hand consistently better** than left for vision and fusion (egocentric camera bias), but EMG is more symmetric.

8. **WiLoR pretraining is critical**: from-scratch WiLoR (7.33° avg) underperforms WiLoR-initialized (4.81°) by +2.5° across all splits. The hand-specialized MANO checkpoint provides a strong inductive prior that 50 epochs of scratch training cannot recover from.
