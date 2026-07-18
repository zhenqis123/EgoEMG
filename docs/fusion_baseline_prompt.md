You are designing a multimodal fusion baseline for a NeurIPS 2026 dataset/benchmark paper. The paper is about EgoEmg, a dataset with synchronized bilateral EMG + egocentric RGB + motion capture hand pose labels.

## Context

We have two existing unimodal baselines and need to design the simplest possible EMG+Vision fusion model. The paper does NOT need architectural innovation for the fusion — it needs a clean, defensible, easy-to-explain baseline that demonstrates the dataset enables multimodal learning.

## EMG Baseline (Emg2PoseFormer)

Architecture:
```
EMG raw (B, 16, 7790)  -- 16 channels, 2000Hz, ~3.895s window
  → TDS Featurizer (2 Conv1dBlocks + 2 TdsStages)
  → features: (B, 256, T') where T' ≈ 75 (80x temporal downsampling)
  → TransformerDecoder (6 layers, d_model=256, 8 heads, RoPE, non-causal)
  → MLP Head (Linear(256,512)→ReLU→Dropout→Linear(512,22))
  → Output: (B, 22, T') -- 20 joint angles + 2 wrist angles, per timestep
```

Key config:
- dropout=0.15, lr=0.0005, batch_size=500, max_epochs=100
- Loss: MAE + 0.01 * FingertipDistance
- Precision: bf16-mixed

## Vision Baseline (Vision2PoseModule)

Architecture:
```
Image raw (B, 3, 256, 256)
  → WiLoR ViT backbone (pretrained, outputs (B, 1280, 16, 12))
  → Global avg pool: (B, 1280)
  → MLP Head (3-layer MLP → 22 joints)
  → Output: (B, 22) -- joint angles, single timestep
```

Key config:
- Backbone frozen for first 10 epochs, then fine-tuned at lr=1e-5
- Head lr=1e-4
- The ViT feature dimension is 1280

## Temporal Mismatch

- EMG input covers a 3.8s window → produces per-timestep predictions (T'≈75)
- Vision input is a single frame snapshot → produces single-timestep prediction
- The EgoEMg dataset has PAIRED EMG + vision data from synchronized capture

## Design Requirements

1. **Simplicity first**: This is a benchmark paper. The fusion should be explainable in 1 paragraph + 1 figure. No complex cross-attention, no new mechanisms. Concat or addition preferred.
2. **Clear temporal handling**: The vision feature is single-frame; EMG feature is a sequence. How to align?
3. **Compatible with existing code**: The EMG baseline's decoder and head should be minimally modified or reused as-is.
4. **Paired data**: EgoEMg dataset provides synchronized EMG + image + pose labels for the same time window. The image corresponds to the center frame of the EMG window (or any fixed position within the window).
5. **Output**: Should match the EMG baseline output: (B, 22, T') — per-timestep joint angles, same format for fair comparison.
6. **Training**: Should be straightforward — ideally a single training loop, not multi-stage pretraining.

## Your Task

Design the fusion module. Specifically provide:

1. **Architecture diagram** (ASCII art): Show the full forward pass from both modalities to output.
2. **Python code**: Write a clean PyTorch module `class EMGVisionFusionFormer(nn.Module)` with a `forward(self, batch)` method. The code should be production-quality with type hints and comments.
3. **Temporal alignment strategy**: Explain how you handle the single-frame vs sequence mismatch, and WHY this choice is appropriate.
4. **Training recipe**: What to freeze, what learning rates, how many epochs. Keep it simple.
5. **Hydra config structure**: Show what the new YAML experiment config should look like (extending the existing EMG config).
6. **Why this is defensible for a benchmark paper**: 2-3 sentences explaining why this simple design is appropriate (not "too simple") for a NeurIPS dataset track paper.

Be specific with tensor shapes at every step. Use the exact dimensions from the baselines above (256 for EMG features, 1280 for ViT features, 22 output channels, T'≈75 timesteps).
