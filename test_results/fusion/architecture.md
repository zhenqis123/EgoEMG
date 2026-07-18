# EMG-Vision Fusion 手部姿态估计架构

## 1. 总体设计：残差融合（Residual Fusion）

核心公式：**preds = y_v + Δy_emg**

- **y_v**：纯视觉基线预测（vision-only baseline）
- **Δy_emg**：EMG 信号对视觉预测的残差修正（EMG residual / delta）
- **preds**：最终融合预测

直觉：视觉提供稳健但不完美的姿态估计，EMG 信号（直接测量肌肉电活动）学习预测视觉的误差并加以修正。

## 2. 视觉通路（Vision Pathway）

以 ViT 为例：

```
输入：webcam 手部裁剪图 (B, 3, 256, 256)
  ↓
ViT Backbone（DINOv2 预训练，可冻结/可训练）
  ↓
vision_features (B, 1280)   ← ViT 输出的 spatial feature 经 mean pooling
  ↓
  ├── head_vision: Linear(1280→512) → ReLU → Dropout → Linear(512→22)
  │     ↓
  │   y_v (B, 22)            ← 纯视觉的单帧姿态预测（22个关节角）
  │
  └── vision_proj: Linear(1280→fusion_proj_dim)
        ↓
      vis_feat (B, 256)      ← 投影到融合空间
```

**关键点：**
- `head_vision` 和 `vision_backbone` 可以从预训练的 vision-only checkpoint 加载
- `vision_proj` 将 backbone 特征降维到融合维度，供后续与 EMG 特征拼接

## 3. EMG 通路（EMG Pathway）

```
输入：EMG 时间窗口 (B, 16, 7790)
  ↓  # 16 通道肌电信号 × 7790 时间步（约 3.9 秒 @ 2000Hz）
TDS Featurizer（时间深度可分离卷积）
  ↓
emg_features (B, 256, 75)   ← 时间维度从 7790 压缩到 75
  ↓
Transformer Decoder（causal=False，带位置编码）
  ↓
decoded (B, 256, 75)        ← 每个时间步一个 256-dim 特征
  ↓
Temporal Attention Pooling  ← 仅在 center_supervised 模式下
  ↓
emg_pooled (B, 256)         ← 对 75 个时间步加权求和，学习哪些时刻对中心帧最重要
```

**关键点：**
- EMG 窗口覆盖比标注帧更长的时间范围（7790 点 ≈ 3.9s），提供丰富的运动上下文
- `Temporal Attention Pooling` 是可学习的：`Linear(256→hidden) → Tanh → Linear(hidden→1)`，对 75 个时间步输出 softmax 权重
- Decoder 是多层 Transformer（small: 3层/4头/128dim；middle: 4层/8头/256dim）

## 4. 融合过程（Fusion）

```
emg_pooled (B, 256)     vis_feat (B, 256)
       ↓                      ↓
       └────── concat ────────┘
                ↓
        fused (B, 512)
                ↓ unsqueeze(-1) → (B, 512, 1)
        fusion_proj: Conv1d(512→512)→GELU→Dropout→Conv1d(512→256)
                ↓
        fused_proj (B, 256, 1)
                ↓
        head: MLP → delta (B, 22, 1)
                ↓
        preds = y_v.unsqueeze(-1) + delta  → (B, 22, 1)
```

**关键设计：**
- `fusion_proj` 是 1D 卷积而非全连接，保留通道维度的交互
- `head` 最后一层用**零初始化**（weight≈0, bias=0），确保训练开始时 Δy_emg≈0，模型从纯视觉基线起步
- 拼接发生在 "瓶颈" 层之后，两个模态各自压缩到 256-dim 再融合

## 5. 四种运行模式（fusion_mode）

| 模式 | 行为 | 用途 |
|------|------|------|
| `fusion` | 全窗口监督：decoder 输出每个时间步，vision 特征广播到所有时间步，逐帧融合 | 需要全时间序列预测 |
| `center_supervised` | EMG 全窗口 → attention pool → 仅预测中心帧。preds = y_v + delta（单帧） | **与 vision-only 公平对比**（相同监督目标） |
| `vision_only` | 仅用 vision 通路，忽略 EMG | vision-only baseline |
| `emg_only` | vision_features 置零，仅用 EMG 通路 | EMG-only baseline |

当前主要使用 **`center_supervised`** 模式：EMG 利用了更长的时序上下文（7790 点），但只预测中心一帧的姿态。这与 vision-only baseline（也只看中心帧）的监督目标完全一致，实现了公平的跨模态对比。

## 6. 训练策略

通过 `component_lr_scales` 控制各组件的学习率缩放：

| 组件 | 参数 | 功能 |
|------|------|------|
| featurizer | 1.0 / 0.0 | EMG 时序特征提取 |
| decoder | 1.0 / 0.0 | Transformer 时序建模 |
| vision_backbone | 1.0 / 0.0 | ViT/ResNet 视觉主干 |
| vision_proj | 1.0 / 0.0 | 视觉特征投影到融合空间 |
| fusion_proj | 1.0 / 0.0 | 多模态特征融合 |
| head | 1.0 / 0.0 | 预测 Δy_emg 残差 |
| head_vision | 1.0 / 0.0 | 预测 y_v 视觉基线 |

scale=0.0 表示冻结该组件。三种常用策略：
- **全冻结 baseline**：仅训练 fusion_proj + head + head_vision
- **全开放**：所有组件 1e-4 训练（当前 version_9 策略）
- **冻结 EMG/开放 vision** 或反之

## 7. 其他关键细节

- **moddrop_prob**：训练时以概率 p 将 EMG 输入置零，强制模型不过度依赖 EMG
- **force_zero_emg**：推理时强制 EMG 置零，测量纯视觉下模型表现
- **delta_reg**：对 Δy_emg 的 L2 正则化权重，防止残差过大
- **center_target_only**：数据集只返回中心帧标签，减少 IO
- **零初始化 residual head**：确保训练开始时 preds ≈ y_v，避免融合破坏视觉基线

## 8. 参数量参考

| 组件 | ResNet+Small | ResNet+Middle |
|------|-------------|---------------|
| Vision Backbone (ResNet18) | ~11M | ~11M |
| TDS Featurizer | ~0.5M | ~0.5M |
| Transformer Decoder | ~1.1M (3L/4H/256d) | ~4.5M (4L/8H/256d) |
| Fusion + Heads | ~0.3M | ~0.3M |
| **总计** | **~13M** | **~16M** |

## 9. Delta 贡献率（version_9, epoch 133）

| 指标 | 值 |
|------|-----|
| \|Δy_emg\| / \|pred\| mean | 25.3% |
| \|Δy_emg\| / \|pred\| median | 25.6% |
| \|Δy_emg\| / \|y_v\| | 31.4% |
| P10 - P90 | 16.3% - 33.7% |

EMG 残差贡献约 25%，vision 占 75%。EMG 信号做的是精细化修正而非主导预测。
