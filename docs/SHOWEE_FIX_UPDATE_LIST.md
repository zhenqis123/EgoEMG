# 数据修复总清单：需更新到百度网盘的文件

（本轮新增第 4 项：全库刚体变换修复+平滑）

修复内容（详见 git log `5c0d46a`、`c3dcd38`、`cc7b5d2`）：
1. ShowEE 左手 MANO 世界变换 180° 翻转修复（z 镜像 → x 镜像约定）
2. ShowEE crops 用各自 session 相机标定重新生成（此前用 EgoEMG 全局标定，
   crop 框心系统性偏移 85–150px）
3. **EgoEMG 右腕 IMU 恢复**：采集器 step1 管线只加载了左腕（primary）设备的
   IMU，右腕带的 `weili_imu.csv` 在 39/41 个采集 session 中都有录制但被丢弃。
   已按 EMG 时间戳对齐重灌入 `imu_band_right`（ep0–40，规范列序 [acc, gyro]）。
   ep29/30 的 session 原始未录右腕，保持零值。

## 网盘上需要替换的文件（对应 /EgoEMG_release/ 下路径）

| 网盘路径 | 大小 | 说明 |
|---|---|---|
| `EgoEMG_full_memmap/mocap_head/mocap_head_transform.dat` | 6.4 GB | 头部变换修复+6Hz 平滑（impossible 步进 全库→0） |
| `EgoEMG_full_memmap/mano/mocap_mano_left_world_transform.dat` | 6.4 GB | 左手世界变换同款处理 |
| `EgoEMG_full_memmap/mano/mocap_mano_right_world_transform.dat` | 6.4 GB | 右手世界变换同款处理 |
| `EgoEMG_full_memmap/checksums.json` | ~10 KB | 含上述三个文件的新 sha256 |

| 网盘路径 | 大小 | 说明 |
|---|---|---|
| `EgoEMG_full_memmap/imu/imu_band_right.dat` | 3.2 GB | 右腕 IMU 恢复（39/41 episodes，ShowEE/Incre 行不变） |
| `EgoEMG_full_memmap/mano/mocap_mano_left_world_transform.dat` | 4.7 GB | 左手变换修正（R ← R·Rot_y(180°)，平移不变） |
| `EgoEMG_crops/episode_000041.lmdb` … `episode_000062.lmdb`（22 个目录） | 共 20.2 GB | 用 session 标定重建的 ShowEE crops |
| `EgoEMG_crops/episode_000041.done` … `episode_000062.done`（22 个小文件） | <1 KB | 对应的完成标记 |

本机源文件位置：
- `/home/xiziheng/develop/emg2pose/data/EgoEMG_full_memmap/mano/mocap_mano_left_world_transform.dat`
- `/home/xiziheng/develop/emg2pose/data/EgoEMG_full_memmap/checksums.json`
- `/home/xiziheng/develop/emg2pose/data/EgoEMG_crops/episode_00004[1-9]…episode_000062.lmdb/.done`

旧版 ShowEE crops 已备份在 `data/EgoEMG_crops_showee_old_backup/`，确认新版无误后可删除。

## 不受影响（无需更新）

- 其余全部 memmap 字段（EMG、关节角、右手变换、head transform 等）
- EgoEMG（ep0–40）与 Incre（ep63–70）的所有数据
- `EgoEMG_videos`、视觉索引、checkpoints
- 仓库脚本修复已直接推送到 GitHub（无需网盘分发）

## 验证

数据侧更新后可运行：
```bash
python scripts/data/validate_memmap.py \
  --memmap-dir "$EGOEMG_ROOT/data/EgoEMG_full_memmap" --checksums
```
