# EgoEMG 数据集已知问题清单

> 本文件记录 EgoEMG 数据集（unified memmap + 视频 + 发布物）在构建与
> 审计过程中发现的**数据问题**，供后续逐步修复。状态图例：
> - ✅ 已修复（含验证方式）
> - 🟡 待补（原始数据存在，可修复）
> - ⚫ 不可补（原始数据缺失/从未录制）
>
> 最后更新：2026-08-20（EgoEMG imu 通道重排修复轮次）

---

## 一、待补问题（🟡）

### 1. EgoEMG 源 imu 为固定模板占位 → ✅ 已修复（2026-08-20，通道重排）

- **状态**：✅ 已修复（原诊断部分有误，见下）
- **真相**：EgoEMG 的 IMU 数据一直是**真实且齐全**的。原始 LeRobot parquet
  （Windows 采集机 `J:\training_dataset_lerobot_full`，41 episode /
  66,161,725 行，100% 非零）的 `observation.imu` 列是 **gyro-first**
  `[gyro×3, acc×3]` 布局（gyro_x 为死轴，恒 0；重力 ~9.2–9.4 m/s² 位于后
  3 通道）。v2 转换将其**逐位照搬**进按 `[acc, gyro]` 语义使用的 unified
  字段 → 误诊为"无重力占位数据"。原诊断中"固定模板值
  [0,-0.093,-0.195]/[-0.15,-2.301,-3.981]"在数据中出现 0 次（当时分析的
  v1/v2 老 memmap 已删除，无法复核该结论的来源）。
- **修复**：`scripts/prepare/fix_egoemg_imu_channel_order.py` 对
  `imu.dat` 的 EgoEMG 行（rows [0, 66,161,725)，已断言连续且 source_id=0）
  原地置换通道 `[3,4,5,0,1,2]`；备份 `imu.dat.bak_prelayout`（3.2 GB）；
  manifest `imu_semantics` 已更新（含 `egoemg_layout_fix` 记录）。
- **验证**：
  - Windows 原始 parquet 全量扫描报告 `scripts/release/imu_verify_report_windows_original.json`
    （41/41 episode，`observation.imu` 100% 非零，重力侧=后 3 通道，死轴=ch0）；
  - 修复后 41/41 episode 的 |acc| p50 与原始报告一致（tol 0.01）；
  - 与未修改的 `/data/xiziheng/EgoEMG_unified_memmap` 副本逐位等价
    （EgoEMG 行 = 旧行置换；ShowEE/Incre 行逐位相同）；
  - 从 Windows 原始 parquet 抽取 ep0 行块（rows 500–600、900000–900100）
    与修复后 memmap **逐位相等**；
  - 数据集类冒烟：`EgoEmgMemmapDataset(modalities=("imu",))` 读取
    EgoEMG/ShowEE 窗口 |acc| 中位数 9.5–9.7，两源语义一致。
- **备注**：11 个 episode（3,4,18,21,22,24,29,30,33,34,35,37,38）|acc|
  中位数偏低（3.0–7.3）**在原始数据中即如此**（非转换伪影），未做幅值
  修正，已在 manifest 注明；ep4 为 5 秒退化片段。gyro_x 死轴如实保留。
- **副本状态**：`/mnt/nvme/xiziheng/EgoEMG_unified_memmap`（活动，已修复；
  `EgoEMG_release/dataset_egoemg_unified` 是它的符号链接，同步修复）；
  `/data/xiziheng/EgoEMG_unified_memmap`（合并期快照，**保持 pre-fix**，
  作为修复前参考，勿再用于训练）。

### 2. emg_left_filtered 缺失（且 emg_right_filtered 管线不可复现）

- **状态**：🟡 待统一重建或文档化
- **现象**：memmap 有 `emg_right_filtered`（**仅 Incre 行有数据，EgoEMG 行实测全零**）但无
  `emg_left_filtered`；`emg_right_filtered` 的滤波管线**无法复现**——
  仓库中三个滤波实现（`scripts/realtime/filter.py`、
  `build_manus_memmap.py`、`filter_emg_into_new_columns.py`）对同一 raw
  段的输出均与存储值不一致（std 1.68 vs 存储 17.7），且 Incre 源 memmap
  已不存在。
- **影响**：左/右手 filtered 变体不对称；`emg_left_filtered` 无法用与
  right 相同的滤波生成（避免左右滤波语义不一致）。
- **待办**：若需要对称的 filtered 字段，用规范滤波
  （`filter_emg_into_new_columns.py`，与 filtered_paper 同管线）为双手
  统一重建 `emg_*_filtered`（注意：会改变 `emg_right_filtered` 的分布，
  影响已训练模型的输入，需谨慎/重训）。

### 3. Incre 3 个会话的 weili_imu.csv 丢失

- **状态**：🟡 待找回原始目录
- **现象**：Incre 共 8 个会话，5 个 `sess_2026*` 有
  `WeiLiEMG_13_COM3/weili_imu.csv`（已补入 `imu_right`）；3 个
  `data_20260526/27*` 会话的原始目录已删除 → `imu_right` 对应行保持 0。
- **待办**：找回 `data_20260526_172725` / `data_20260526_230859` /
  `data_20260527_124150` 的原始目录后补填。

### 4. Incre imu_right 覆盖不完整（58%）

- **状态**：🟡 待补（或标注）
- **现象**：5 个会话的 weili_imu.csv 覆盖会话时长 ~79%，且 csv 内部存在
  全零段（IMU 未开启）→ `imu_right` 的 Incre 行非零率 58%。
- **待办**：如需完整覆盖需原始 CSV（与 #3 同源）；否则在 manifest 注明
  覆盖率。

### 5. 3 个截断的相机 imu json（尾部 ~1s 丢失）

- **状态**：🟡 已容错读取，数据本身缺失
- **现象**：与 #6（损坏 mkv）同一录制中断事件的 3 个右腕相机 IMU 日志
  截断：`20260714_0062_midair_1/{asl_y,ring_bend}`、
  `20260717_0072_handobject_1/checking_for_price_tag` 的
  `showee_right_wrist/imu_*.json`。`fill_showee_incre_imu.py` 按行容错解析
  （保留完整行），尾部 ~1s 的 IMU 数据丢失。
- **待办**：若原始日志有备份可替换完整文件；否则维持现状（缺失 ~1s）。

### 6. README 模态措辞与实际情况不符 → ✅ 仓库 README 已修正（遗留：发布包 README.txt）

- **状态**：✅ 仓库 README 已在 0.1.0rc1 预发布整理中移除 "wrist IMUs" 与
  "external RGB-D" 宣称（IMU 部分另见 #1 的通道重排修复）。剩余工作：legacy
  数据发布包内的 README.txt 如含相同措辞，正式数据发布时一并修正。

### 19. Incre 行 mocap_valid 位与零 keypoints 矛盾

- **状态**：🟡 数据级待决策（重刷位 or 文档标注）
- **现象**：unified memmap 中 Incre 源（ep 63 起）`mocap_{left,right}_valid`
  全为 True，但 `mocap_*_keypoints` 全零、manifest `source_policies` 声明
  "valid=False"。且仓库 `merge_datasets_to_unified_memmap.py` 的
  `VISION_STALE_TRUE_FIELDS` 缺 `image_wrist_left/right_stale` 两个位
  （线上数据为 True，说明由旧版脚本产出；重跑仓库脚本反而会得到 False）。
- **影响**：当前 loss 不消费这些位（只看 label_valid + dataset_name 腕掩码），
  但任何用 `mocap_valid`/wrist stale 做过滤的下游会被误导。
- **待办**：用补丁脚本重刷 Incre 行的 valid/stale 位（或至少在 manifest
  source_policies 中如实标注现状）；同时修 merge 脚本的 stale 位清单。
- **来源**：2026-08-20 数据管线独立审查（见 docs/code_review_findings_20260820.md P1）。

---

## 二、不可补问题（⚫）

### 7. EgoEMG 41 个 episode 的 ZED 视频丢失

- **状态**：⚫ 原始文件已删除，本机无副本；发布仅含 Incre 4 个会话的 ZED
- **现象**：metadata 的 `episode_zed_video_path` 指向
  `videos/observation.images.zed/chunk-000/episode_XXXXXX.mp4`，但原始
  文件不存在（`training_dataset_lerobot_full_NEW` 仅剩 reprojection_assets）。
- **待办**：找回原始 ZED 视频后可重编码发布；否则标注 EgoEMG 无 ZED。

### 8. ZED 深度流未录制

- **状态**：⚫ 录制时未保存深度
- **现象**：所有 `zed_rgbd/` 目录只有 `rgb.mkv` + `rgb_timestamps.jsonl`，
  无深度文件 → "RGB-D" 名不副实，实际为外部 RGB。
- **待办**：修正文档措辞（"external RGB（ZED）"）。

### 9. emg_right_filtered 管线不可复现（源 memmap 丢失）

- **状态**：⚫（与 #2 关联）
- **现象**：Incre 源 memmap（含 filtered 的计算输入/定义）已不存在，
  仓库内滤波实现均不匹配存储值。

---

## 三、已修复问题（✅，留档）

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 10 | **ShowEE 视频偏移**：发布视频为会话级拼接（22 个），但 memmap 帧号为逐动作局部帧号 → 动作越靠后偏移越大 | `fix_showee_session_frame_indices.py` 按精确动作边界 + mkv 帧数重算 22 会话全部 3,750 万行 | 无回绕、max<视频帧数、会话视频与源 mkv 逐帧 diff 0.7–3.0（错位时 31.8） |
| 11 | **3 个损坏右腕 mkv**（moov atom 缺失，录制中断） | `repair_unfinalized_mp4.py`（mdat NAL 提取 → annex-b → MP4 重封装，timescale 6000 与录制器一致）；重建 ep43/ep61 wrist_right 会话视频 + 重算帧号 | 恢复 6,237/1,499/3,479 帧（99%）；会话视频帧数精确匹配 |
| 12 | **episode_end_idx 含端点 off-by-one**（71-ep 布局 start+len-1） | `rebuild_session_metadata.py` 改为独占语义 | 与 928-ep 布局一致 |
| 13 | **episode_beta_idx 全 0**（所有 episode 读 beta 行 0） | 重算为 EgoEMG 自身行 / ShowEE 会话首动作行 / Incre 原行 | — |
| 14 | **metadata 字符串双编码**（/data 旧副本 `b"b'...'"`） | 71-ep 重建为干净单编码；/data 已同步 | 数据集解码正常 |
| 15 | **有 txt 无 mkv 动作帧号错位**（如 0063-ho roll_between_palms） | mkv 缺失即 -1（`build_showee_wrist_zed_indices.py`） | 全量无回绕 |
| 16 | **webcam → head 彻底重命名**（字段/视频/代码/文档） | `rename_webcam_to_head.py` + 163 处代码替换 | pytest 42 通过、四流读取正常 |
| 17 | **ShowEE wrist×2 + ZED 会话视频与帧号** | 66 个会话级视频 + 4 路帧号 + 元数据路径 + 数据集四流支持 | 帧数精确和、逐帧对齐、e2e 读取通过 |
| 18 | **ShowEE/Incre 缺失 IMU 补齐** | `fill_showee_incre_imu.py`：imu_right/imu_head/imu_wrist_left/imu_wrist_right（weili acc 按 9.8/5.4 标定） | ShowEE 96–100%、Incre 5/8 会话 |

---

## 四、工程遗留（非数据问题）

- **未提交 git**：`scripts/prepare/` 下新增脚本（`fix_showee_session_frame_indices`、
  `rebuild_session_metadata`、`rename_webcam_to_head`、`build_showee_session_videos`、
  `build_showee_wrist_zed_indices`、`finalize_release_metadata`、
  `repair_unfinalized_mp4`、`fill_showee_incre_imu`、
  `fix_egoemg_imu_channel_order`、`verify_original_lerobot_imu`）、
  `scripts/release/imu_verify_report_windows_original.json`（原始数据验证报告）、
  `egoemg/tests/test_egoemg_imu_reorder.py`、
  `scripts/viz/visualize_dataset.py`（统一数据集可视化入口）。
- **备份文件清理**：两个 memmap 的 `.bak`/`.orig928`/`.npz.bak2/3`/`.json.bak`、
  `imu.dat.bak_prelayout`（3.2 GB，#1 修复前快照，确认修复稳定后可删）、
  损坏 mkv 的 `.mkv.broken`（3 个，确认修复后可删）。
- **/tmp 旧 readme 草稿**：`videos_readme2.txt`/`crops_readme2.txt` 等描述旧的
  928-per-action 方案，已被实际发布状态取代。
- **root 盘空间**：/ 曾 100% 满（已清理 /tmp 历史垃圾 ~17GB），发布/构建时注意
  `/tmp` 用量。
