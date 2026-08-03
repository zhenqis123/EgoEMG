# EMG2Pose Ablation Study

> 实验启动时间：2026-05-04 ~01:50
> 实验脚本：`bash/run_all_ablation.sh`（15 个实验顺序执行）
> 日志目录：`logs/2026-05-04/`

## 实验设计

消融实验覆盖三个维度：

1. **模型架构**：vemg2pose (LSTM) / emg2pose (MLP) / neuropose / emgformer (Transformer) × small/middle/large
2. **数据增强**：with aug（Rotation + ChannelMask + TimeMask + FreqMask + GaussianNoise） vs wo aug（仅 RotationAugmentation）
3. **归一化策略**：per-dataset norm vs no norm (null)
4. **初始化策略**：from scratch vs pretrained checkpoint（aug 预训练后微调）

实验分为三组，在不同数据集上运行：

| 组别 | 数据集 | 样本量 | 实验编号 |
|------|--------|--------|---------|
| A | EgoEMG (left + right) | ~126K | #1-#9 |
| B | emg2pose_v3 | — | #10-#15 |

---

## 组 A：EgoEMG 数据集（per-dataset norm）

### A1. 传统模型 aug vs wo aug（#1-#6）

**共同配置**：
- 数据集：EgoEMG（left + right）
- 归一化：per-dataset norm
- 初始化：from scratch
- 学习率：lr=0.001
- 最大 epoch：100
- GPU：6 卡 DDP

| # | 实验 | 模型架构 | bs | with aug | epoch | best val_mae | 状态 |
|---|------|---------|-----|----------|-------|-------------|------|
| 1 | vemg2pose with aug | VEMG2PoseWithInitialState + LSTM | 600 | 是 | 100/100 | **0.2740** | 完成 |
| 2 | vemg2pose wo aug | VEMG2PoseWithInitialState + LSTM | 600 | 否 | 100/100 | 0.2750 | 完成 |
| 3 | emg2pose with aug | StatePoseModule + MLP | 900 | 是 | 100/100 | **0.2733** | 完成 |
| 4 | emg2pose wo aug | StatePoseModule + MLP | 900 | 否 | 72/100 | 0.2753 | 提前停止 |
| 5 | neuropose with aug | PoseModule | 256 | 是 | 100/100 | 0.2775 | 完成 |
| 6 | neuropose wo aug | PoseModule | 256 | 否 | 99/100 | 0.2784 | 完成 |

**日志目录映射**（按时间戳）：

| 时间戳 | 对应实验 | 备注 |
|--------|---------|------|
| 01-50-45 | #1 vemg2pose with aug | bs=600 重试成功 |
| 03-11-11 | #2 vemg2pose wo aug | — |
| 04-27-08 | #3 emg2pose with aug | bs=900 重试成功 |
| 05-30-18 | #4 emg2pose wo aug | 提前停止于 epoch 72 |
| 06-10-38 | #5 neuropose with aug | — |
| 07-25-54 | #6 neuropose wo aug | — |
| 01-29-43 | emg2pose with aug (首次) | bs=900 OOM 崩溃 |
| 01-46-23 | vemg2pose with aug (首次) | bs=900 epoch 2 崩溃 |

**观察**：
- 三个模型中 with aug 均略优于 wo aug，但差距很小（0.001-0.002）
- emg2pose (MLP) > vemg2pose (LSTM) > neuropose，均在 0.27-0.28 区间
- bs=900 时 OOM，egoemg/vemg2pose 首次运行崩溃后自动重试成功

### A2. emgformer + pretrained ckpt（#7-#9）

**共同配置**：
- 数据集：EgoEMG（left + right）
- 归一化：per-dataset norm
- 初始化：**pretrained checkpoint（aug 预训练）**
- 训练时数据增强：**wo aug**（仅 RotationAugmentation）
- 最大 epoch：200
- GPU：6 卡 DDP

| # | 实验 | decoder preset | 参数量 | lr | bs | epoch | best val_mae | 状态 |
|---|------|---------------|--------|-----|-----|-------|-------------|------|
| 7 | small aggressive egoemg | small (3L/4H/ffn=512) | ~4.8M | 5e-5 | 600 | 200/200 | **0.2562** | 完成 |
| 8 | middle aggressive egoemg | middle (6L/8H/ffn=1024) | ~6.6M | 5e-4 | 250 | 200/200 | **0.2482** | 完成 |
| 9 | large aggressive egoemg | large_aggressive (8L/12H/ffn=1536) | — | 5e-4 | 200 | epoch 175/200 | **0.2493** | 提前终止（磁盘满） |

**日志目录映射**：

| 时间戳 | 对应实验 | 备注 |
|--------|---------|------|
| 08-35-05 | #7 small egoemg | pretrained ckpt: 04-30/22-36-29, lr=5e-5 |
| 17-05-05 | #8 middle egoemg | pretrained ckpt: 04-29/11-06-44, lr=5e-4, bs 从 550 降至 250 |
| 18-56-28 | #9 large egoemg | pretrained ckpt: 04-29/13-03-41, lr=5e-4, bs 从 400 降至 200 |
| 16-35-18 | #8 首次尝试 | bs=550 OOM 崩溃于 epoch 1 |
| 17-02-13 | #8 二次尝试 | bs=550 再次 OOM（GPU 显存未释放） |
| 10-27-56 | (另见组 B #10) | — |

**故障记录**：
- #8、#9 原始脚本中 `pin_memory=true` 导致 CUDA initialization error，通过命令行 `datamodule.pin_memory=false` 覆盖修复
- #8 首次用 bs=550 OOM，降至 bs=250 后稳定运行（~35s/epoch）
- #9 首次用 bs=400 OOM，降至 bs=200 后稳定运行
- #7 之后的实验因 GPU 显存未释放导致一连串 OOM，清理 GPU 后恢复

**观察**：
- pretrained checkpoint 带来显著提升：从 from scratch 的 ~0.27 降至 ~0.25
- middle (0.2482) > small (0.2562)，差距 0.008，模型增大有效
- 注意：此处 "wo aug" 指微调阶段不用 aug，但 pretrained checkpoint 本身来自 aug 训练，并非真正的 "no augmentation from scratch"

---

## 组 B：emg2pose_v3 数据集（no norm, from scratch）

### B1. no norm + with aug（#10-#12）

**共同配置**：
- 数据集：emg2pose_v3
- 归一化：null（no norm）
- 初始化：from scratch
- 数据增强：with aug（emgformer_regression_aug_best）
- 学习率：lr=1e-4
- 最大 epoch：150
- GPU：6 卡 DDP

| # | 实验 | decoder preset | bs | epoch | best val_mae | 状态 |
|---|------|---------------|-----|-------|-------------|------|
| 10 | small aggressive | small | 650 | 150/150 | **0.2293** | 完成 |
| 11 | middle aggressive | middle | 400 | — | — | 排队 |
| 12 | large aggressive | large_aggressive | 250 | — | — | 排队 |

**日志目录映射**：

| 时间戳 | 对应实验 | 备注 |
|--------|---------|------|
| 10-27-56 | #10 前半段 | 跑至 epoch 57 (best=0.2428) 后被 kill |
| 12-09-28 | #10 续跑 | 从 last.ckpt 恢复，跑完 150 epoch (best=0.2293) |

### B2. no norm + wo aug（#13-#15）

**共同配置**：同 B1，但数据增强替换为 wo aug（`transforms=rotation_augmentation`）

| # | 实验 | decoder preset | bs | epoch | best val_mae | 状态 |
|---|------|---------------|-----|-------|-------------|------|
| 13 | small aggressive wo aug | small | 650 | — | — | 排队 |
| 14 | middle aggressive wo aug | middle | 400 | — | — | 排队 |
| 15 | large aggressive wo aug | large_aggressive | 250 | — | — | 排队 |

**说明**：组 B 是真正的 aug 消融实验。组内 #10 vs #13、#11 vs #14、#12 vs #15 分别在小/中/大模型上对比 with aug vs without aug，控制其他变量不变。

---

## 结果汇总

```
EgoEMG 组（per-dataset norm, from scratch, 100ep）
  vemg2pose:  with aug=0.2740,  wo aug=0.2750  (aug 好 0.001)
  emg2pose:   with aug=0.2733,  wo aug=0.2753  (aug 好 0.002)
  neuropose:  with aug=0.2775,  wo aug=0.2784  (aug 好 0.001)

EgoEMG 组（per-dataset norm, pretrained ckpt, ft wo aug, 200ep）
  emgformer small:  0.2562
  emgformer middle: 0.2482  (vs small: -0.008)
  emgformer large:  0.2493 (epoch 175/200, 提前终止于磁盘满)

emg2pose_v3 组（no norm, from scratch, with aug, 150ep）
  emgformer small:  0.2293 (val) / 0.2562 (test)
  emgformer middle: TBD
  emgformer large:  TBD

emg2pose_v3 组（no norm, from scratch, wo aug, 150ep）
  TBD
```

**注意**：EgoEMG 和 emg2pose_v3 是两个不同数据集，val_mae 不可跨组直接对比。

---

## 最佳 Checkpoint 归档

所有最优 checkpoint 已拷贝到 `ablation_study/checkpoints/`，按实验组别分类：

```
ablation_study/checkpoints/
├── A1_egoemg_traditional/
│   ├── #1_vemg2pose_with_aug_mae0.2740.ckpt       (epoch 82, ~69M)
│   ├── #2_vemg2pose_wo_aug_mae0.2750.ckpt         (epoch 41, ~69M)
│   ├── #3_emg2pose_with_aug_mae0.2733.ckpt        (epoch 53, ~34M)
│   ├── #4_emg2pose_wo_aug_mae0.2753.ckpt          (epoch 21, ~34M)
│   ├── #5_neuropose_with_aug_mae0.2775.ckpt       (epoch 91, ~73M)
│   └── #6_neuropose_wo_aug_mae0.2784.ckpt         (epoch 48, ~73M)
├── A2_egoemg_emgformer/
│   ├── #7_small_egoemg_wo_aug_mae0.2562.ckpt      (epoch 190, ~41M)
│   ├── #8_middle_egoemg_wo_aug_mae0.2482.ckpt     (epoch 108, ~77M)
│   └── #9_large_egoemg_wo_aug_mae0.2493.ckpt      (epoch 99, ~188M)
└── B1_emg2pose_v3/
    └── #10_small_aggressive_with_aug_mae0.2293.ckpt (val_mae=0.2293, test_mae=0.2562, ~41M)
```

> **来源日志目录**：见各实验的「日志目录映射」表。文件名中的 mae 值为 TB 记录的最优 `val_mae`。

---

## 执行脚本

### 原始脚本（bash/run_all_ablation.sh，已被覆盖）

```bash
# A1: 传统模型 aug vs wo aug
python -m egoemg.train experiment=emg2pose/regression_vemg2pose_egoemg_with_aug    # #1
python -m egoemg.train experiment=emg2pose/regression_vemg2pose_egoemg             # #2
python -m egoemg.train experiment=emg2pose/regression_emg2pose_egoemg_with_aug     # #3
python -m egoemg.train experiment=emg2pose/regression_emg2pose_egoemg              # #4
python -m egoemg.train experiment=emg2pose/regression_neuropose_egoemg_with_aug    # #5
python -m egoemg.train experiment=emg2pose/regression_neuropose_egoemg             # #6

# A2: emgformer egoemg wo aug (pretrained ckpt)
python -m egoemg.train experiment=emgformer/regression_emgformer_small_aggressive_egoemg_wo_aug   # #7
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive_egoemg_wo_aug  # #8
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive_egoemg_wo_aug   # #9

# B1: no norm with aug
python -m egoemg.train experiment=emgformer/regression_emgformer_small_aggressive   # #10
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive  # #11
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive   # #12

# B2: no norm without aug
python -m egoemg.train experiment=emgformer/regression_emgformer_small_aggressive transforms=rotation_augmentation   # #13
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive transforms=rotation_augmentation  # #14
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive transforms=rotation_augmentation   # #15
```

### 当前续跑脚本（bash/run_remaining_ablation.sh）

```bash
# retry #8: middle aggressive egoemg wo aug (bs=250, pin_memory=false)
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive_egoemg_wo_aug datamodule.pin_memory=false batch_size=250

# retry #9: large aggressive egoemg wo aug (bs=200, pin_memory=false)
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive_egoemg_wo_aug datamodule.pin_memory=false batch_size=200

# no norm with aug (bs 降低)
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive batch_size=400
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive batch_size=250

# no norm without aug
python -m egoemg.train experiment=emgformer/regression_emgformer_small_aggressive transforms=rotation_augmentation
python -m egoemg.train experiment=emgformer/regression_emgformer_middle_aggressive transforms=rotation_augmentation batch_size=400
python -m egoemg.train experiment=emgformer/regression_emgformer_large_aggressive transforms=rotation_augmentation batch_size=250
```

---

## 配置对照

### 关键 Hydra Config

| 参数 | EgoEMG 组 A1 | EgoEMG 组 A2 | emg2pose_v3 组 B |
|------|-------------|-------------|-----------------|
| 数据集 | EgoEMG left+right | EgoEMG left+right | emg2pose_v3 |
| norm_mode | per-dataset | per-dataset | null |
| per_dataset_norm_stats_path | assets/... | assets/... | assets/... |
| lr | 0.001 | 5e-5 / 5e-4 | 1e-4 |
| max_epochs | 100 | 200 | 150 |
| pretrained_strict | — | true / false | — |
| pin_memory | — | false（覆盖） | false |
| 数据增强 | with aug / wo aug | wo aug（微调阶段） | with aug / wo aug |

### 模型参数量

| decoder preset | 参数量（featurizer + decoder） |
|---------------|------------------------------|
| small | ~4.8M (1.69M + 3.1M) |
| middle | ~6.6M (1.69M + 4.8M) |
| large_aggressive | 更多（8L/12H/ffn=1536） |

### 数据增强 Preset

| Preset | 包含 transforms |
|--------|----------------|
| emgformer_regression_aug_best (with aug) | ExtractToTensor + RotationAugmentation + ChannelMask + TimeMask + FreqMask + GaussianNoise |
| rotation_augmentation (wo aug) | ExtractToTensor + RotationAugmentation（仅旋转增强，无 mask/noise） |
