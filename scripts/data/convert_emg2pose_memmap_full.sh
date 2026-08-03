#!/bin/bash
# 一键转换 EMG2Pose Zarr 数据到 Memmap 格式
#
# 用法:
#   bash scripts/data/convert_emg2pose_memmap_full.sh
#
# 磁盘需求: ~465 GB (emg + joint_angles + valid_mask + time)
# 请确保目标磁盘有足够的可用空间

set -e

ZARR_ROOT="${EMG2POSE_ROOT:-$(pwd)}/data/emg_corpus/emg2pose_v3"
MEMMAP_ROOT="${EMG2POSE_ROOT:-$(pwd)}/data/emg_corpus/emg2pose_v3_memmap"
LOG_FILE="/tmp/emg2pose_memmap_convert.log"

echo "========================================"
echo "EMG2Pose Zarr -> Memmap 转换"
echo "========================================"
echo "Zarr源: $ZARR_ROOT"
echo "Memmap目标: $MEMMAP_ROOT"
echo "日志: $LOG_FILE"
echo ""

# 检查磁盘空间
AVAILABLE_GB=$(df -BG ${EMG2POSE_ROOT:-$(pwd)}/data/emg_corpus/ | grep nvme | awk '{print $4}' | sed 's/G//')
REQUIRED_GB=465

echo "可用空间: ${AVAILABLE_GB}GB"
echo "需要空间: ${REQUIRED_GB}GB"

if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo ""
    echo "警告: 磁盘空间不足!"
    echo "建议: 先删除zarr数据后释放空间"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "开始转换..."
echo "预计时间: 3-4小时"
echo ""

# 创建目录
mkdir -p "$MEMMAP_ROOT"

# 执行转换
python scripts/data/convert_emg2pose_zarr_to_memmap.py \
    --zarr-root "$ZARR_ROOT" \
    --out-root "$MEMMAP_ROOT" \
    --fields emg joint_angles valid_mask time \
    --overwrite \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "转换完成!"
echo "========================================"
echo "输出目录: $MEMMAP_ROOT"
echo ""

# 显示结果
ls -lh "$MEMMAP_ROOT"

echo ""
echo "验证数据完整性..."
if [ -f "$MEMMAP_ROOT/manifest.json" ]; then
    python3 -c "
from pathlib import Path
import json

memmap_dir = Path('$MEMMAP_ROOT')
manifest_path = memmap_dir / 'manifest.json'

with open(manifest_path) as f:
    manifest = json.load(f)

print(f'Total rows: {manifest[\"total_rows\"]:,}')
print(f'Fields: {list(manifest[\"fields\"].keys())}')
print(f'Sessions: {len(manifest[\"sessions\"])}')
print(f'Users: {len(manifest[\"users\"])}')

# 验证文件存在
for field, info in manifest['fields'].items():
    fpath = memmap_dir / info['filename']
    size_gb = fpath.stat().st_size / (1024**3)
    print(f'  {field}: {size_gb:.1f} GB')
"
else
    echo "警告: manifest.json不存在，转换可能未完成"
fi

echo ""
echo "如需删除zarr释放空间，运行:"
echo "  rm -rf $ZARR_ROOT"