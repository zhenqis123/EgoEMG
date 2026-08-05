# EgoEMG 数据集已知问题清单

> 本文件记录 EgoEMG 数据集（unified memmap + 视频 + 发布物）在构建与
> 审计过程中发现的**数据问题**，供后续逐步修复。状态图例：
> - ✅ 已修复（含验证方式）
> - 🟡 待补（原始数据存在，可修复）
> - ⚫ 不可补（原始数据缺失/从未录制）
>
> 最后更新：2026-08-05（ShowEE 视频/IMU 补齐轮次）

---

## 一、待补问题（🟡）

### 1. EgoEMG 源 imu 为固定模板占位

- **状态**：🟡 待确认原始数据后决定（清零 or 补真实数据）
- **现象**：`imu` 字段的 EgoEMG 行（episode 0–40，全部 6,616 万行）为固定占位
  值：`acc=[0, -0.093, -0.195]`, `gyro=[-0.15, -2.301, -3.981]`，无重力分量
  （|acc|≈0.2，真实 IMU 应为 ~9.8）。v1 老 memmap 与 v2 memmap **逐位一致**
  → 固定模板占位，非坏传感器数据。
- **来源**：EgoEMG v2 memmap 的实际构建脚本不在仓库中（仓库内
  `convert_simple_to_egoemg_v2.py`/`build_manus_memmap.py` 写的是全零；
  唯一写真实 IMU 的是 `build_showee_memmap.py`）。原始 LeRobot 数据
  （parquet）本机已丢失，无法确认原始是否有 `observation.imus` 列。
- **影响**：用 `imu` 训练时 EgoEMG 行无效；README 的 "wrist IMUs" 承诺对
  EgoEMG 源不成立。
- **待办**：找到原始 LeRobot 数据（百度网盘/原采集机）→ 检查是否有 IMU
  列；有则补真实 IMU；没有则把 EgoEMG 行清零（与 Incre 的"无数据=0"
  约定一致）并修正 README 措辞。
- **参考**：ShowEE 左腕带 imu 为真实数据（|acc|≈9.8）；新增的
  `imu_right`/`imu_head`/`imu_wrist_left`/`imu_wrist_right` 均为真实数据。

### 2. emg_left_filtered 缺失（且 emg_right_filtered 管线不可复现）

- **状态**：🟡 待统一重建或文档化
- **现象**：memmap 有 `emg_right_filtered`（来自 Incre 源）但无
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

### 6. README 模态措辞与实际情况不符

- **状态**：🟡 待修正
- **现象**：README 宣称 "wrist IMUs"（EgoEMG 源无真实 IMU，见 #1）与
  "external RGB-D"（ZED 录制未保存深度流，见 #8）。
- **待办**：按实际覆盖范围修正措辞（或补数据后保留）。

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
  `repair_unfinalized_mp4`、`fill_showee_incre_imu`）、`scripts/viz/verify_showee_session_alignment.py`。
- **备份文件清理**：两个 memmap 的 `.bak`/`.orig928`/`.npz.bak2/3`/`.json.bak`、
  损坏 mkv 的 `.mkv.broken`（3 个，确认修复后可删）。
- **/tmp 旧 readme 草稿**：`videos_readme2.txt`/`crops_readme2.txt` 等描述旧的
  928-per-action 方案，已被实际发布状态取代。
- **root 盘空间**：/ 曾 100% 满（已清理 /tmp 历史垃圾 ~17GB），发布/构建时注意
  `/tmp` 用量。
