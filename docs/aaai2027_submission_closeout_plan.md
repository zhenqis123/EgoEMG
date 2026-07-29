# AAAI 2027 投稿收尾计划

## 目标

在不继续扩展新 fusion 架构的前提下，完成现有实验闭环、统一评测口径、
验证微小融合增益的统计可靠性，并消除主文与补充材料中的协议和表述矛盾，
最终生成可以正式提交的 `main_submission.pdf` 与 `supplementary.pdf`。

当前最关键的三个投稿门槛是：

1. 完成 7 个视觉 backbone × frozen/unfrozen 的 14 项 fusion 对比；
2. 为 Table 4 的配对增益提供 participant-clustered 95% 置信区间；
3. 统一代码、Figure 3、正文和表格中的 split 定义与样本统计。

## 执行原则

- 先锁定实验和评测口径，再修改最终数字与论文结论。
- Test set 不参与 frozen/unfrozen 或 checkpoint 选择。
- 所有 Table 4 结果使用同一个 unified center-frame evaluator。
- 对微小差异使用逐样本配对统计，不能用跨用户标准差代替训练或模型不确定性。
- 如果实验无法在投稿前完成，则删除相应“完整对比”声明，不保留 `--` 或
  `experiments still being finalized`。
- 将论文定位为 dataset + benchmark paper；EMGFormer 和 residual fusion 是
  reference baselines，不把简单 fusion 架构包装成主要方法创新。

## 阶段 1：锁定实验协议与结果清单

### 工作项

- [ ] 建立 14 项实验 manifest。
- [ ] 为每项实验记录以下信息：
  - vision backbone；
  - frozen / unfrozen；
  - Hydra experiment config；
  - 数据集范围；
  - EMG window length、stride 和 dataset repeat；
  - augmentation，特别是 Mixup 是否关闭；
  - optimizer、初始 LR、eta min、epoch 和 batch size；
  - vision/EMG 初始化 checkpoint；
  - total parameters 和 trainable parameters；
  - validation metric 和最佳 checkpoint；
  - unified test result 路径。
- [ ] 检查 RN18 和 ViT-S 已有 unfrozen 结果是否与本轮 simple frozen recipe
  构成严格受控对比。
- [ ] 检查所有实验是否使用相同数据、预处理、center-frame target 和
  checkpoint-selection 规则。

### 验收标准

- 14 项实验均有唯一、可追溯的配置与 checkpoint。
- 不混用不同数据集、augmentation、epoch budget 或评测口径的结果。
- 如果历史结果不满足受控对比条件，则明确标记为需要重跑。

### 建议产物

```text
test_results/paper_aaai2027/table4_final/experiment_manifest.json
```

## 阶段 2：补齐 14 项 frozen/unfrozen 实验

### 当前已知待处理项

- [ ] 完成当前 ViT-S + EMGFormer-S frozen 训练。
- [ ] 补齐 RN18 + EMGFormer-S frozen。
- [ ] 将 ViT-B + EMGFormer-S unfrozen 补足到既定 30 epochs，或明确说明统一的
  停止规则。
- [ ] 将 ViT-L + EMGFormer-S unfrozen 从现有 checkpoint 恢复并训练至 30 epochs。
- [ ] 完成 WiLoR + EMGFormer-S unfrozen。
- [ ] 若阶段 1 发现 RN18/ViT-S unfrozen 的历史配置不匹配，则按统一 recipe 重跑。

### 统一实验条件

- 仅使用 EgoEMG；
- EMGFormer-S；
- simple residual fusion；
- center-frame supervision；
- WL12000；
- dataset repeat 2；
- AugBest，但 Mixup 关闭；
- 30 epochs；
- LR 从 `1e-4` cosine anneal 到 `5e-6`；
- frozen/unfrozen 之间只改变视觉分支的可训练状态；
- batch size 可以按显存调整，但需记录有效 global batch size。

### 验收标准

- Supplementary 的 2×7 表中不再存在 `--`。
- 所有实验达到预先规定的训练预算。
- 每项均保留 `last.ckpt`、最佳 validation checkpoint、TensorBoard event 和完整配置。

## 阶段 3：统一 center-frame 评测

### 工作项

- [ ] 用 unified center-frame evaluator 重测 14 项 fusion 模型。
- [ ] 对每个 backbone 同时评测对应 vision-only checkpoint。
- [ ] 强制使用完全相同的 center frames、valid mask 和左右手样本。
- [ ] 保存以下逐样本字段：
  - fusion prediction；
  - vision-only prediction；
  - target；
  - subject ID；
  - split ID；
  - episode/frame index；
  - hand side；
  - validity mask。
- [ ] 单独保存 validation 结果，并只用 validation MAE 选择 frozen/unfrozen。
- [ ] Test 结果只在选择完成后汇总，不允许根据 test MAE 更换变体。

### 建议目录

```text
test_results/paper_aaai2027/table4_final/
├── experiment_manifest.json
├── validation_selection.json
├── unified_center_eval.json
├── sample_counts.json
└── predictions/
```

### 验收标准

- Table 4 和 Supplementary Table 9 的全部数字都能从该目录自动重建。
- Vision/Fusion 的比较严格配对，不存在样本数量或 valid frame 不一致。
- 明确记录左右手样本数和每个 split 的样本数。

## 阶段 4：配对统计与置信区间

### 方法

以 participant 为聚类单位进行 paired bootstrap，而不是独立重采样帧：

1. 在每个 split 内按 participant 有放回采样；
2. 保留每个被采样 participant 的全部有效左右手样本；
3. 分别计算 Vision MAE、Fusion MAE 和
   `gain = MAE_vision - MAE_fusion`；
4. 重复 10,000 次并固定随机种子；
5. 报告 paired gain 的均值、95% CI 和样本数；
6. 对 Overall 使用预先定义的 sample-weighted 聚合方式。

### 结果解释规则

- 95% CI 完全大于 0：可信提升；
- 95% CI 跨过 0：与 vision-only 统计上持平；
- 95% CI 完全小于 0：可信退化。

### 工作项

- [ ] 为 7 个 validation-selected fusion 结果计算 Gesture/User/Both/Overall CI。
- [ ] 为完整 14 项结果保留同样的统计文件。
- [ ] 检查多重比较风险，并避免把单个极小点估计描述为稳定提升。
- [ ] 如果时间允许，为关键 RN18、ViT-S 和 WiLoR 模型增加 3-seed 训练统计。

### 验收标准

- Table 4 的 `0.03–0.09°` 差异不再仅以点估计支撑。
- 正文措辞与 CI 结论一致。

## 阶段 5：同步 EMG 有效性控制实验

### 优先模型

- RN18 + EMGFormer-S：当前融合收益最大的 ResNet；
- 至少一个强视觉模型：优先 WiLoR 或 RN152。

### 对照条件

- [ ] 正确同步 EMG；
- [ ] Zero EMG；
- [ ] Shuffled EMG，优先在 split/subject 内打乱以保留边际分布；
- [ ] Time-misaligned EMG，如 evaluator 可以可靠实现则加入固定偏移条件。

这些实验优先使用已有 checkpoint 做 inference-time control，不重新训练。

### 验收标准

- 同步 EMG 明显优于 zero/shuffled/misaligned 条件，才能把收益归因于跨模态信息。
- 如果控制实验不能支持该结论，则在论文中把 fusion 结果定位为经验 baseline，
  不宣称已经证明 EMG 的独立互补信息。

## 阶段 6：Split 协议审计

### 工作项

- [ ] 从 metadata 和实际 split 文件直接生成权威统计。
- [ ] 明确以下集合：
  - Train；
  - Validation；
  - Gesture：seen users × held-out gestures；
  - User：held-out users × seen gestures；
  - Both：held-out users × held-out gestures。
- [ ] 确认 Gesture/User/Both 是否互斥。
- [ ] 核验 User 是否错误包含了 Both。
- [ ] 核验 Table 2/4 的 Avg 是否重复统计样本。
- [ ] 记录每个集合的 participant 数、gesture 数、episode 数、frame 数和左右手样本数。
- [ ] 明确 validation 的来源及其与 test 的隔离方式。
- [ ] 根据代码事实修改 Figure 3 和正文定义。

### 建议表格

| Split | Participants | Gestures | Episodes | Hand samples | Mutually exclusive |
|---|---:|---:|---:|---:|---|
| Train | TBD | TBD | TBD | TBD | -- |
| Validation | TBD | TBD | TBD | TBD | TBD |
| Gesture | TBD | TBD | TBD | TBD | TBD |
| User | TBD | TBD | TBD | TBD | TBD |
| Both | TBD | TBD | TBD | TBD | TBD |

### 验收标准

- 代码、Figure 3、正文和所有 evaluator 使用完全一致的 split 语义。
- 读者能够判断三个 test split 是否互斥以及 Avg 如何计算。

## 阶段 7：更新 Table 4 与 Supplementary Table 9

### Table 4

- [ ] 每个 backbone 只展示 validation-selected 的 frozen/unfrozen 结果。
- [ ] 保留 Vision → Fusion 的逐 split 两位小数展示。
- [ ] 根据统计结果决定使用“improved”还是“comparable”。
- [ ] 保留 total parameters，并增加 trainable parameters 或在 caption/正文说明。
- [ ] 明确 F/T、selection metric、center-frame protocol 和 paired sample 数。

### Supplementary Table 9

- [ ] 展示完整 7×2 结果。
- [ ] 每行包含 Gesture/User/Both/Avg。
- [ ] 标记 validation-selected variant。
- [ ] 删除所有未完成实验标记。
- [ ] 说明 frozen/unfrozen 使用相同数据、初始化、augmentation 和训练预算。

### 验收标准

- 主表和附录数字一致。
- 选择依据来自 validation，不存在 test-oracle selection。
- 表格 caption 可以独立解释指标、单位、统计量和标记。

## 阶段 8：更新关键论文表述

### Abstract / Introduction

- [ ] 将绝对 novelty claim 改为 `to our knowledge` 并限定为同步双腕 EMG、
  egocentric vision 和 continuous bimanual pose 的组合。
- [ ] 删除或弱化 `consistently improves`。
- [ ] 不再直接声称 fusion 严格超过 full-trajectory EMG-only。
- [ ] 将核心贡献聚焦为 dataset、unified benchmark 和 reference baselines。

推荐的 fusion 总结方向：

> Fusion provides the largest gains for ResNet-18 and ViT-S, while gains over
> stronger visual backbones are modest.

### Evaluation Protocol

- [ ] 明确视觉输入是 mocap-derived oracle hand crop。
- [ ] 明确 fusion 使用 centered、non-causal 6-second EMG window。
- [ ] 将任务限定为 offline, oracle-cropped, center-frame pose regression。
- [ ] 明确 EMG-only full-trajectory 与 vision/fusion center-frame 指标不可直接横比。

### Results / Discussion / Limitations

- [ ] 根据 CI 而不是点估计描述 fusion 收益。
- [ ] 删除“未来评测 WiLoR”的过期表述。
- [ ] 删除“only lightweight generic visual branches”；WiLoR 并不轻量。
- [ ] 将 per-gesture analysis 的 60 类修正为实际分析的 59 类，或补入 Raw 类。
- [ ] 谨慎解释 self-occlusion 的 `rho=0.033`，不依赖 frame-level 小 p 值制造强趋势。
- [ ] 将 Table 3 的 22% 改为相对于论文 reported result 的名义对比。

### Reproducibility / Ethics

- [ ] 使 fusion training details 与实际 30-epoch 配置一致。
- [ ] 逐 backbone 给出 LR、batch、epoch、augmentation、初始化和 trainable params。
- [ ] 补充 split IDs、manifest、checksum 和 reviewer sample/evaluator 信息。
- [ ] 弱化“EMG 不包含敏感健康信息”的绝对表述。
- [ ] 如数据可得，补 participant demographics、handedness 和公平性限制。

## 阶段 9：同步与标签质量证据

### 必要统计

- [ ] camera–mocap residual offset 的均值、中位数和 95th percentile；
- [ ] synchronization jitter；
- [ ] stale-frame 比例；
- [ ] 最大时间偏差；
- [ ] EMG/pose 对齐敏感性或现有同步诊断结果。

### 标签表述

- [ ] 将 markers2mano 与旧 IK 的 `3.5× reduction` 明确标记为跨数据集背景比较，
  或在同一 EgoEMG markers 上重跑旧 IK 后再做直接比较。
- [ ] 区分 marker fitting error、joint-angle accuracy 和 2D reprojection error。
- [ ] 如果可以，补充定量 reprojection error，而不仅是随机可视化检查。

## 阶段 10：最终一致性与投稿检查

### 内容检查

- [ ] Abstract/Introduction 中每个主要 claim 都有明确实验支持。
- [ ] 主文、supplementary、配置和结果文件中的数字一致。
- [ ] 不存在未完成实验、临时目录或未来时承诺与当前结果冲突。
- [ ] 所有外部方法在表格中有引用，并区分 reimplemented/reported results。
- [ ] 所有 metric 的单位、方向、小数位和聚合方法一致。
- [ ] 修复 `Exeternal`、`pretraining remain` 等语言问题。
- [ ] 解释 Figure 2 的 3.9 s 展示窗口与 6 s 训练窗口的区别。

### 编译与视觉检查

- [ ] 运行 `paper/aaai2027/build.sh`。
- [ ] 检查 undefined references、citation warnings 和 overfull boxes。
- [ ] 逐页检查 9 页主稿。
- [ ] 检查 supplementary 的目录、表格分页和引用编号。
- [ ] 最终生成：
  - `paper/aaai2027/main_submission.pdf`；
  - `paper/aaai2027/supplementary.pdf`；
  - `paper/aaai2027/build/main.pdf`。

### 最终独立复审

- [ ] 再调用一个无上下文 reviewer agent，仅基于最终 PDF 给出评分。
- [ ] 对所有 must-fix 意见逐条标记 resolved / intentionally scoped / unresolved。
- [ ] unresolved 的高风险问题必须在提交前解决或明确弱化相关 claim。

## 推荐执行顺序

```text
实验 manifest 与协议审计
→ 补齐 14 项实验
→ unified center-frame evaluation
→ participant-clustered paired bootstrap
→ zero/shuffled/misaligned EMG control
→ split 定义与统计审计
→ 更新 Table 4 和 Supplementary Table 9
→ 重写 Abstract/Results/Limitations/Reproducibility
→ 全文一致性和 PDF 检查
→ 独立 reviewer 复审
```

## 投稿决策门槛

满足以下条件后才进入最终提交：

- [ ] 14 项 frozen/unfrozen 实验全部闭环，或删除完整 2×7 对比声明；
- [ ] Table 4 的微小增益有配对 95% CI；
- [ ] split 定义、样本数和 Avg 计算完全明确；
- [ ] Abstract 与结论不再超出实验支持范围；
- [ ] oracle crop 和 non-causal EMG 的任务范围已显著披露；
- [ ] 主文与 supplementary 不存在数值、实验状态或未来工作矛盾；
- [ ] 最终 reviewer 复审无未解决的直接拒稿风险。
