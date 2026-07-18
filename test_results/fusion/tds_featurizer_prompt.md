# TDS Featurizer — Prompt for Pipeline Diagram Generation

Draw a pipeline diagram of the TDS (Time-Depth Separable) featurizer for EMG signal processing. The architecture follows "Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions" (Hannun et al., 2019), adapted for EMG.

INPUT: Raw EMG window (B, 16 channels, 7790 time steps @ 2kHz ≈ 3.9s)

The pipeline consists of 4 stages that progressively downsample the temporal dimension while expanding/processing channel features:

STAGE 1 — Conv1dBlock (initial channel expansion):
- Conv1d: 16 → 256 channels, kernel=11, stride=5
- ReLU activation
- Dropout
- Optional Squeeze-Excitation (global average pooling → FC → ReLU → FC → Sigmoid → channel-wise multiply, with residual connection)
- Output: (B, 256, ~1556)

STAGE 2 — Conv1dBlock (further downsampling):
- Conv1d: 256 → 256 channels, kernel=5, stride=2
- ReLU + Dropout + optional SE
- Output: (B, 256, ~776)

STAGE 3 — TdsStage (time-depth separable processing):
  Step A: Conv1dBlock (256→256, kernel=9, stride=5) → (B, 256, ~154)
  Step B: TDSConvEncoder (1 block pair):
    - TDSConv2dBlock:
      1. Reshape: (B, 256, T) → (B, channels=8, feature_width=32, T) — split 256 = 8×32
      2. Conv2d over temporal dimension: kernel=(1, 5), groups=1 — operates on the 32-wide feature per channel group
      3. ReLU
      4. Reshape back: (B, 8, 32, T) → (B, 256, T')
      5. Residual connection (truncate to match new T')
      6. LayerNorm over channel dimension
      7. Optional SE (global pooling over time)
    - TDSFullyConnectedBlock:
      1. Transpose to (B, T, 256)
      2. MLP: Linear(256→256) → ReLU → Linear(256→256)
      3. Transpose back to (B, 256, T)
      4. Residual connection
      5. LayerNorm over channel
      6. Optional SE
  → Output: (B, 256, ~150)

STAGE 4 — TdsStage (final refinement):
  Step A: Conv1dBlock (256→256, kernel=3, stride=1) → (B, 256, ~148)
  Step B: TDSConvEncoder (1 block pair, kernel_width=3):
    - Same structure as Stage 3 but with smaller temporal kernel
  → Output: (B, 256, ~146)

FINAL OUTPUT: (B, 256, T') where T' is the downsampled temporal dimension

Key design principles to highlight in the diagram:
1. Time-depth separability: the TDSConv2dBlock reshapes channel dimension into (channel_groups × feature_width) so that temporal convolution operates on grouped features independently, then FC blocks mix across all channels
2. Every sub-block has a residual connection + LayerNorm
3. Squeeze-Excitation (SE) is applied after each block for channel-wise attention
4. The stride-5 and stride-2 Conv1d blocks do the heavy temporal downsampling (7790 → ~75-150 steps)
5. The TDS blocks refine features with temporal context (kernel_width=3~5) without further downsampling