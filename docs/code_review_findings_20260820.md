# 独立代码审查发现清单（2026-08-20，5 个独立子 agent 并行审查）

> 审查角度：训练复现流 / 评估与可视化流 / 数据管线 / 模型与 Lightning 代码 / 文档配置一致性。
> 状态图例：✅ 本批已修复 | 🔧 待修（下一批，按优先级排序） | 📋 数据级/需决策 | 👁 观察项（不改）
> 每条的详细证据（文件:行号、冒烟复现输出）见当次审查记录；本清单为追踪索引。

## 批次一：已修复（commit 见 git log `Fix documented-command breakage...`）

| # | 来源 | 问题 | 修复 |
|---|---|---|---|
| F1 | 训练B1/文档B1,B2 | `trainer.devices=[0]` 在 emgformer/pretrain 配置链上无该 key，官方命令 Hydra 解析即挂 | `config/base.yaml` trainer 默认 `devices: 1`（README/AGENTS 命令原文可跑；多卡需显式覆盖） |
| F2 | 训练B2 | 默认 stats 文件缺 `egoemg_unified__*` 键 → unified 训练在 dataset 构建时 KeyError | emgformer/fusion lineage 默认指向 `per_dataset_norm_stats_unified.json`（超集；legacy 评估配方保留各自显式覆盖） |
| F3 | 训练M1 | `version_base=1.1` → hydra `job.chdir=True`，相对路径在 run 目录下失效 | `config/base.yaml` 显式 `hydra.job.chdir: false` |
| F4 | 训练M3 | `train_pretrain` 裸入口默认 experiment 指向 `_archive` 配置，compose 失败 | `config/pretrain.yaml` 默认改 `emgformer/pretrain_multitask` |
| F5 | 训练M2/数据M1 | 评估入口 `dataset_name` 硬编码 `egoemg`，unified 训练的检查点评估用错归一化键（右手 std 差 ~17%） | 两个评估入口新增 `eval_dataset_name` 配置旋钮（默认 `egoemg` 保持 legacy 复现；unified 检查点评估传 `egoemg_unified`），README notes 说明 |
| F6 | 评估M2 | `test_analysis_fusion` 读旧键 `egoemg_memmap_dir`、示例配置名不存在 | 改 unified 键优先 + 回退；docstring 待下一批一并更新 |
| F7 | 评估M4 | `create_results_df` 用 `k.split("/")[0]` 压平指标族，多指标互相覆盖 | `k.replace("/", "_")` |
| F8 | 评估M3 | 评估输出单位为弧度、README 表为角度，无换算说明 | README Evaluation notes 增加弧度说明与换算例 |
| F9 | 文档B3 | README 可视化示例缺 `--crops-dir`/`--data-root`，按 ASSET_SETUP 布局必 FileNotFoundError | 示例补全两个参数；ASSET_SETUP canonical 树补 `reprojection_assets/`、`emg_norm_stats.npz`、unified stats |
| F10 | 文档M1 | LICENSE 宣称 "EgoEMG dataset released"，与预发布口径矛盾 | LICENSE 措辞改为"legacy 数据发布可用 CC-BY-NC-4.0；完整数据集仍在准备中" |
| F11 | 评估m1/文档m3 | vision 模式 overlay MP4 在 LMDB/crop key 校验前创建，失败遗留空文件（与 PRERELEASE 契约字面不符） | 三个 writer 统一移到全部校验之后 |
| F12 | 文档m1,m2,m4,m6,m8 | scripts/README 指向错误/缺目录、emgformer_small 头注释 16ch↔8ch、config_architecture 组名过时、VIZ_README 字段名错 | 已逐一修正；m6（README 提及 emg2pose/ 目录）、m7（python badge）下一批 |

## 批次二：修复状态（2026-08-20 第二轮，commit `Apply batch-two review fixes...`）

| # | 状态 | 说明 |
|---|---|---|
| D1 | ✅ 已修 | pretrain_multitask 三头补 `_target_`（keystroke out=98=CharacterSet）；tds_no_out 首 conv 8→16ch（预训练语料统一 16ch）；新增 2 项配置级测试 |
| D2 | 📌 按设计关闭 | 维护者裁决：center-frame 评估纳入缺 crop 黑图为**预期行为**，不改（发布数字口径保持） |
| D3 | ✅ 已修 | SDPA bool mask 进入前取反（MHA↔SDPA 语义差）；3 项回归测试（无未来泄漏/无 NaN/key_padding 隔离） |
| D4 | ⏸ 暂缓 | eval_center_stride 无 split 检索路径不一致——被全部在用配置遮蔽；修复需重构 block 索引，单独一轮处理 |
| D5 | ✅ 已修 | `_build_vision_sample` 缺 crops 时 vision_valid=False（与快速路径对齐） |
| D6 | ✅ 已修 | 批级 MixUp 以同系数混合 label_valid_mask（>0.5 阈值） |
| D7 | ✅ 已修 | ResNet 分支 `head_vision.` 键双重剥前缀；核实 DINOv2/WiLoR 分支本就正确（审查报告对 WiLoR 的指控不成立） |
| D8 | ✅ 已修 | train/test_analysis 的 torch.load 补 weights_only=False；移除 PL2.5 无效 kwarg |
| D9 | ✅ 已修 | center_frame：split id 按 manifest labels 名字解析（回退 [1,2,3]）、设备自适应（CPU 可跑）、EMG preference 配置化（默认 filtered_paper） |
| D10 | ✅ 已修 | collate keystroke 全批探测 + 消费端 None 安全 |
| D11 | ✅ 已修 | EGOEMG_NO_PRETRAINED_DOWNLOAD=1 环境开关（resnet/timm 两构造点），评估加载检查点时自动设置 |
| D12 | ✅ 已修 | `_evaluate_egoemg_pooled` 按 (split,hand) 键配对，缺手跳过并提示 |
| D13 | ⏸ 暂缓 | RotationAugmentation 归一化后 roll 的语义问题需增强策略复核（涉及已训练配方语义），单独立项 |
| D14 | ✅ 部分 | float64 提升修复；EMG variant 缺失改为构造期明确报错；`_prune_hand_fields` 静默丢弃告警暂缓（需模态上下文重构） |

## 批次二原清单（留档）

| # | 来源 | 问题 | 备注 |
|---|---|---|---|
| D1 | 模型B1 | `pretrain_multitask.yaml` recon/gesture/keystroke 头缺 `_target_`（首 forward 即 TypeError）+ featurizer 8ch vs 数据 16ch | README pretrain 命令修复 devices 后仍会挂在此处；需对照 `_archive` 配置与 `lightning_pretrain` 期望补头定义 |
| D2 | 评估M1 | center-frame 评估忽略 `vision_valid_mask`：23+22 个缺 crop 黑图样本计入发布 MAE（~1%） | 修复会改变发布数字复现——需与"重跑评估刷新表值"一起做并记录口径变化 |
| D3 | 模型M1 | `RotaryMultiheadAttention` 把 MHA 语义 bool mask 直传 SDPA（语义相反）+ key_padding 同病；`causal=true` 时反因果+末位 NaN | 当前 active 配置均 `causal: false`（仅默认值危险）；修复=进 SDPA 前取反，配单测 |
| D4 | 数据B1 | `eval_center_stride` 无 frame_split 时索引构建(ecs)与取数(stride)不一致 → 跨 episode/越界 | 现被 val 配置的 split 过滤遮蔽；修=ecs 窗口独立成 block，配回归测试 |
| D5 | 数据M2 | `_build_vision_sample` 缺 crops 时全零图但 valid=True（与两条快速路径不一致） | 一行修复+对齐 |
| D6 | 数据M3 | 批级 MixUp 混合 target 不处理 partner 的 label_valid（~21% 无效左手样本混入零姿态目标） | 修=混合 mask 或 valid 间配对；会影响增强语义，需谨慎+测试 |
| D7 | 模型M2,M3 | WiLoR 分支 head 键未剥前缀 / ResNet 分支 `head_vision.` 双重剥前缀 → 特定 checkpoint 组合构造期 KeyError | 仅 `_archive` 配置触达；修=统一"已剥前缀"形态 |
| D8 | 训练m1,m2 | `torch.load` 三处 `weights_only` 不一致；`load_from_checkpoint(weights_only=)` 在 PL 2.5 是无效参数（声称的 2.6 保护不生效） | 统一处理 |
| D9 | 评估m2,m3,m4 | center_frame 硬编码 `filtered_paper`/split id `[1,2,3]`/强制 `.cuda()`；`center_frame_window_length` 配置键不存在 | 统一默认+按 metadata 解析 split+CPU 回退 |
| D10 | 训练m5 | `_collate_fn` keystroke 字段探测只看 `batch[0]` | 混合数据集路径 |
| D11 | 评估m5 | 评估离线机器可能触发 torchvision/timm ImageNet 下载（`pretrained: true` 未在评估时置空） | 离线可用性 |
| D12 | 评估m9 | `_evaluate_egoemg_pooled` 位置配对假设脆弱 | 空 split 时拿错对 |
| D13 | 数据m1 | RotationAugmentation 在归一化后 roll 通道（sparse16 mask 错位潜伏）；与批级"confirmed harmful"结论不一致 | 需复核增强语义 |
| D14 | 数据m2,m5,m7 + 评估m7 | 杂项：`emg_left_filtered` 缺失时报错点远、`_prune_hand_fields` 静默丢弃显式请求字段、numpy 增强 float64 提升、Avg. 加权需用户自算 | 低风险清理 |

## 数据级 / 需决策

| # | 来源 | 问题 |
|---|---|---|
| P1 | 数据M4 | unified memmap 中 Incre 行 `mocap_{left,right}_valid` 全 True 与零 keypoints/manifest 声明矛盾；仓库 merge 脚本 `VISION_STALE_TRUE_FIELDS` 亦缺 wrist 两个 stale 位（线上数据由旧版脚本产出）。已记入 `docs/data_known_issues.md` #19 |
| P2 | 文档M2 | pretrain "canonical workflow" 依赖 ASSET_SETUP legacy 树中不存在的语料（emg2qwerty/Ninapro 等）→ ASSET_SETUP 需补"pretrain 语料不在 legacy 包内"的说明或补语料获取方式 |
| P3 | 训练m4 | `eval=true` 时 test 与 val 评估同一份数据（`test: ${dataset.val}`），结果键名误导 | 决策：去掉重复评估或改名 |

## 批次三：新鲜克隆复现验证（2026-08-20，验收轮）

模拟"新人克隆 + 按文档放置资产"实跑 README 全部评估与四种可视化，发现并修复：

| # | 问题 | 修复 |
|---|---|---|
| V1 | center-frame 评估读旧 `egoemg_memmap_dir` 别名 → 规范资产树上 FileNotFoundError | `center_frame.py` 改 unified 优先 |
| V2 | README 将 rn50+8ch+WL12000 的 `fusion_resnet_emgfusion_center.ckpt` 配到 rn18/16ch/7790 配置 → 形状不匹配崩溃；且 bundle 并无 rn18-fusion 检查点，覆盖声明失实 | 新增 `fusion_{rn50_m,vits_s}_center_eval_released.yaml`（compose 验证）；README 换正确配对 + 补 ViT vision/fusion 命令 + 修正覆盖声明 |
| V3 | 校准文件在发布资产树无可达位置（只在预览包 meta/ 与未发布的 LeRobot 树里） | 解析器增加发布树候选路径；校准文件上传至云端 `/EgoEMG_release/reprojection_assets/`；ASSET_SETUP 树已含该位置 |
| V4 | 所有 episode 的 crop LMDB 从帧 1 起（帧 0 无 crop），stride 网格必含帧 0 → vision 示例对任何 episode 必失败 | 选择期跳过无 crop 帧并打印数量；缺 LMDB/空选择/保留帧缺 key 仍硬错误；PRERELEASE 措辞同步 |

**实测复现数字**（fresh clone、4090、弧度×57.2958 换算）：
- vision rn18 5.85°（表 5.84）✓；vision ViT-S 6.04°（表 6.02）✓；ViT-S fusion 5.56°（表 5.54）✓
- rn50 fusion（发布检查点）5.36°（表无此行，见 README 覆盖说明）
- EMGFormer-S：gesture 12.21/user 16.15/both 16.70/overall 14.08° vs 表 12.8/15.6/17.4/14.7 —— **新旧提交逐位一致**（f1b575c 对照实验），差异为既有评估口径差，非本轮改动引入；已在 README 注记
- 四种可视化全部实跑通过：vision（29 帧 overlay+双手 crop MP4，mesh/marker/bbox 对齐人工核验）、timeline（PNG）、mesh（8 帧 GLB+markers+occlusion）、fk_vs_mano（GLB 对）

## 观察项（与设计一致或当前不可达，仅记录）

- 训练o：compose 内部自洽（T_max/monitor_metric/loss 键匹配）、make_data_module 装配、split 过滤、入口防护——验证通过。
- 评估o：center-frame 网格数学、三视频同帧序、坐标/手性约定（MANO-right 语义+显示期反射）、MAE 口径（元素级、per-user、样本加权）——验证通过。
- 数据o：窗口/jitter 边界、归一化加载路由、transforms 数值、layout 插值、worker 随机流——验证通过。
- 模型o：RoPE 数值（<1e-7）、TDS 感受野、SMU/低通、val 累加器重置、checkpoint 方向、冻结策略、FK 约定——验证通过。
- 数据o3：`_filtered_paper` 字段未清理 → 训练 dataloader 每样本双份 EMG IO（评估不受影响）。
- 模型o1：掩码重建预训练无 mask token（弱化去噪任务）；o2：`norm_mode: batch` 在验证期用批统计；o3：vision_only 的 DDP unused parameters 风险（active 多卡配置不受影响）。
