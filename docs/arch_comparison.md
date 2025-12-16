# 架构对比：Emg2Pose（TDS + Last-Step XAttn） vs FairEMG（Conv + Wav2Vec2）

本文对比了当前 emg2pose 仓库中使用的 **TDS 前端 + Last-Step Cross-Attention** 架构，与 fairEMG 中的一个典型 **Conv 特征器 + Wav2Vec2 编码器** 方案。便于后续迁移/融合时参考。

## 输入与特征提取
- **Emg2Pose**：原始 EMG 波形（16 通道），无显式归一化；前端 TDS 堆栈（Conv1d + TDS blocks），无 padding，大幅下采样（11790 → ~125），左上下文约 1790。
- **FairEMG Conv+Wav2Vec2**：通常先做 spectrogram/conv 特征器（可配置卷积层数、stride/核宽），得到帧级特征，再喂给 Wav2Vec2。支持输入特征维度配置、可选冻结 conv/feat-projection。

## 时间建模 / 编码器
- **Emg2Pose**：TDS 前端仅作为特征抽取，时间上下文由 TDS 卷积覆盖（固定感受野），后续不再堆叠 Transformer。
- **FairEMG**：Wav2Vec2 Transformer 编码器（多层自注意力），支持冻结/解冻部分层、可配置层数/头数/FFN 宽度，支持掩码（SpecAugment-like）。

## 位置编码
- **Emg2Pose**：Last-step 解码器使用 BERT encoder/decoder（绝对位置嵌入），下游只监督窗口末帧。
- **FairEMG**：Wav2Vec2 自带相对位置（Conv pos）/或绝对位置取决于 HF 配置；不特意添加额外位置编码层。

## 解码 / 读出层
- **Emg2Pose**：自定义 LastStepCrossAttnPoseModule；20 个可学习 query token 在 BERT decoder 中做 self-attn + cross-attn 聚合 encoder 序列，输出单帧 20 维角度（只监督最后一帧）。
- **FairEMG**：通常为 CTC/分类/回归头（Linear/MLP/Pooled MLP），对全序列帧输出 logits；蒸馏时还可有特征/Logit distill heads。无专门 “last-step only” 机制。

## 监督与掩码策略
- **Emg2Pose**：Last-step 数据集预先过滤末帧 IK failure；batch 仅含末帧标签 (C×1)。训练时只在末帧计算 MAE/末端距离等。
- **FairEMG**：标准 CTC/分类序列监督，按帧或 CTC 时间步计算损失；掩码主要用于数据增广（时间/特征掩码）和可选的下游屏蔽。

## 数据管线与窗口
- **Emg2Pose**：滑窗长度固定（11790），stride 常 1000，TDS 左上下文 1790；LastStep dataset 只返回末帧标签。
- **FairEMG**：数据预处理为 sharded dataset，分割/采样可配置（sampler/augmentations），窗口长度由任务和 shard 配置决定（qwerty 任务为字符转录）。

## 训练目标
- **Emg2Pose**：回归 20 维关节角；默认损失 MAE + 指尖距离（权重 0.01）；只评估末帧。
- **FairEMG**：CTC/分类/蒸馏；损失包含 CTC、Logit distill、特征 distill 等，面向 EMG→字符或个性化任务。

## 运行与精度
- **Emg2Pose**：BF16 mixed precision 常用；解码器可选 small/medium/large/xlarge 预设；Last-step 架构显存较低但只监督末帧。
- **FairEMG**：HF Wav2Vec2 + 大 batch 训练，支持 AMP/多卡/分布式；参数规模取决于 Wav2Vec2 配置（可冻结降低显存）。

## 适用场景
- **Emg2Pose TDS+LastStep**：手势/姿态回归，末帧监督、跨窗口下采样，适合 IK 掩码过滤和低延迟读出。
- **FairEMG Conv+Wav2Vec2**：序列建模（如 EMG→文字转录）、CTC/蒸馏任务，重 Transformer 上下文，关注全序列输出。
