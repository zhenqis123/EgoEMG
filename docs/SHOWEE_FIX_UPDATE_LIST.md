# ShowEE 修复：需更新到百度网盘的文件清单

修复内容（详见 git log `5c0d46a`、`c3dcd38`）：
1. ShowEE 左手 MANO 世界变换 180° 翻转修复（z 镜像 → x 镜像约定）
2. ShowEE crops 用各自 session 相机标定重新生成（此前用 EgoEMG 全局标定，
   crop 框心系统性偏移 85–150px）

## 网盘上需要替换的文件（对应 /EgoEMG_release/ 下路径）

| 网盘路径 | 大小 | 说明 |
|---|---|---|
| `EgoEMG_full_memmap/mano/mocap_mano_left_world_transform.dat` | 4.7 GB | 左手变换修正（R ← R·Rot_y(180°)，平移不变） |
| `EgoEMG_full_memmap/checksums.json` | ~10 KB | 上述文件的 sha256 已同步更新 |
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
