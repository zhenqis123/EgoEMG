# EMG+Vision Fusion 实验索引 (2026-07)

本文件归档 7 月fusion 实验的配置、checkpoint 位置与结果，便于后续快速定位。
实验线：`EMG + Vision (ResNet18/DINOv2 ViT-S) → center_supervised fusion`。

## 数据布局说明

- **OLD 布局**：`emg2pose_interpolate16`（8ch→16ch 环形插值）+ `filtered` 字段（当前 memmap 已无此字段）
- **NEW 布局**：`target_hand`（原生 8ch）+ `filtered_paper` 字段 + `per_dataset_norm_stats_repro_filtered_paper_alias.json`
- memmap 帧率 2000Hz，wl=12000=6s，wl=7790≈3.9s

## 实验结果总表

| 实验 | 布局/增强 | wl/stride/repeat | val/test_mae | best ckpt 位置 |
|---|---|---|---|---|
| **OLD RN18-S v14** | interpolate16/aug_best | 7790/400/2 | **0.0945** | `logs/fusion/resnet_small_emgfusion_center/version_14/` |
| OLD RN18-S v15 (多轮续训) | 同上 | 同上 | 0.0944 | `.../version_15/` |
| NEW RN18-S | target_hand/aug_extended | 12000/1200/1 | 0.0992 | 见下方路径 |
| NEW RN18-M | 同上 | 同上 | 0.0989 | 见下方路径 |
| NEW ViT-S+S | target_hand/aug_extended | 12000/1200/1 | 0.1411 | 见下方路径 |
| NEW ViT-S+M | 同上 | 同上 | 0.1621 | 见下方路径 |
| **denalign stage1** (RN18-S, 密度对齐) | target_hand/aug_extended | 12000/400/2 | 0.1022 | `runs_local/denalign/...` |
| **denalign stage2** (+100ep warm restart) | 同上 | 同上 | **0.1004** | `logs/2026-07-18/17-20-17_emg2pose/...` |

## NEW fusion 4 实验 (2026-07-17)

配置: `config/experiment/fusion/fusion_{rn18_s,rn18_m,vits_s,vits_m}_center_8ch.yaml`
共同设置: target_hand + filtered_paper + aug_extended + wl12000/stride1200/rep1 + batch200(4卡) + lr6e-4 + cosine150ep

- **RN18-S** (0.0992): `logs/2026-07-17/00-10-44_emg2pose/logs/fusion/fusion_rn18_s_8ch/version_0/checkpoints/`
- **RN18-M** (0.0989, 最优): `logs/2026-07-17/03-14-22_emg2pose/logs/fusion/fusion_rn18_m_8ch/version_0/checkpoints/`
- **ViT-S+S** (0.1411): `logs/2026-07-17/01-26-15_emg2pose/logs/fusion/fusion_vits_s_8ch/version_0/checkpoints/`
- **ViT-S+M** (0.1621): `logs/2026-07-17/04-36-40_emg2pose/logs/fusion/fusion_vits_m_8ch/version_0/checkpoints/`

## denalign 时序密度对齐实验 (2026-07-17~18)

目的: 在 NEW 布局下把训练密度对齐到 OLD (stride1200→400, rep1→2)，隔离时序变量。
配置: `config/experiment/fusion/fusion_rn18_s_center_8ch_denalign.yaml`

- **stage1** (单卡GPU0, 100ep): best 0.1022 @ epoch100
  - ckpt: `runs_local/denalign/fusion_rn18_s_8ch_denalign/version_0/checkpoints/`
  - 日志: `logs/fusion/rn18_s_8ch_denalign_restart_20260718_003352.log`
- **stage2** (5卡GPU1-5, +100ep warm restart from 6e-4): best **0.1004** @ epoch81
  - 配置: `config/experiment/fusion/fusion_rn18_s_center_8ch_denalign_stage2.yaml`
  - ckpt: `logs/2026-07-18/17-20-17_emg2pose/runs_local/denalign_stage2/fusion_rn18_s_8ch_denalign_stage2/version_0/checkpoints/`
  - 日志: `logs/fusion/rn18_s_8ch_denalign_stage2_20260718_172012.log`

## 关键结论

1. **RN18 完胜 ViT-S**: 0.099 vs 0.14~0.16，DINOv2 ViT-S 这条线明显弱。
2. **新旧差距主因是数据布局/增强，非训练密度**: denalign 密度对齐+200ep+warm restart (0.1004) 仍差于 OLD (0.0945)，且与稀疏密度的 NEW (0.0992) 持平。
3. **OLD 严格复现不可行**: 当前 memmap 无 `filtered` 字段、16ch EMG 预训练 ckpt 丢失。

## 代码改动

- `emg2pose/lightning.py`: `_load_pretrained_backbone` 白名单新增 `temporal_attn.`（两处）。
  原因: 加载 fusion ckpt 时 temporal_attn(EMG时间注意力池化, center_supervised核心)会被漏掉导致随机初始化。
  验证: denalign stage2 加载 240/244 keys 完整覆盖。

## OLD 实验 (历史参考, 不可严格复现)

- 配置: `config/experiment/fusion/vision_resnet_small_emgfusion_center.yaml`
- v14 (单次从零, 0.0945): `logs/fusion/resnet_small_emgfusion_center/version_14/`
- v15 (多轮续训, 0.0944): `logs/fusion/resnet_small_emgfusion_center/version_15/`
- v14 关键超参: lr=6e-4, T_max=150, wl7790/stride400/rep2, decoder 4h/3L/512
- 依赖(已缺失): EMG init ckpt `logs/2026-04-30/.../aggressive_egoemg/...0.2625.ckpt`
