#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""可视化 HDF5 session 的 IK 插值结果，可批量处理目录。

展示指定关节在原始失败位置（用 NaN 表示缺失）与插值后的轨迹。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import seaborn as sns
import matplotlib

# 无头环境使用非交互后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emg2pose.data import Emg2PoseSessionData
from emg2pose.utils import get_ik_failures_mask, get_contiguous_ones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化一个或多个 HDF5 文件中插值后的关节角轨迹，对比原始失败段。"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="单个 emg2pose HDF5 文件路径，或包含多个 HDF5 的目录。",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=0,
        help="要绘制的关节索引（从 0 开始）。默认 0。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始样本索引（含）。默认 0。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束样本索引（不含）。默认到序列末尾。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="单文件模式下的输出图片路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="目录模式下的输出根目录，保持相对层级并以 .png 输出。",
    )
    parser.add_argument(
        "--pattern",
        default="*.hdf5",
        help="目录模式下匹配 HDF5 文件的模式，默认 *.hdf5。",
    )
    return parser.parse_args()


def load_data(path: Path, joint_idx: int, start: int, end: int):
    with h5py.File(path, "r") as f:
        group = f[Emg2PoseSessionData.HDF5_GROUP]
        ts = group[Emg2PoseSessionData.TIMESERIES]
        end = end if end is not None else len(ts)

        slice_data = ts[start:end]
        times = slice_data[Emg2PoseSessionData.TIMESTAMPS]
        # 用相对时间，避免绝对时间戳过大
        times = times - times[0]

        # 存储为弧度制，绘图前转换成角度制
        joint_angles = np.rad2deg(
            slice_data[Emg2PoseSessionData.JOINT_ANGLES][:, joint_idx]
        )

        if "ik_failure_mask" in group:
            ik_fail_mask = group["ik_failure_mask"][start:end].astype(bool)
        else:
            # 若缺失则即时计算
            full_mask = ~get_ik_failures_mask(ts[Emg2PoseSessionData.JOINT_ANGLES])
            ik_fail_mask = full_mask[start:end]

    return times, joint_angles, ik_fail_mask


def make_plot(times: np.ndarray,
              joint_angles: np.ndarray,
              ik_fail_mask: np.ndarray,
              output: Path):
    # 原始视角：失败段置 NaN，便于看出缺口
    original = joint_angles.copy()
    original[ik_fail_mask] = np.nan

    sns.set_theme(style="whitegrid")

    # 一行两行子图，共用 x 轴
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True
    )

    # === 上面：原始（失败为缺口） ===
    ax1.plot(times, original, label="Original (failures as gaps)")
    ax1.set_ylabel("Joint angle (deg)")
    ax1.set_title("Original joint angle (deg, IK failures as gaps)")
    ax1.legend(loc="upper right")

    # 标出失败区间（连续失败段用阴影表示）
    for start, end in get_contiguous_ones(ik_fail_mask):
        ax1.axvspan(times[start], times[end], color="red", alpha=0.15, linewidth=0)

    # === 下面：插值后的轨迹 ===
    ax2.plot(times, joint_angles, label="Interpolated", color="C1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Joint angle (deg)")
    ax2.set_title("Interpolated joint angle (deg)")
    ax2.legend(loc="upper right")

    fig.suptitle("IK interpolation comparison", y=0.98)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"已保存: {output}")


def process_file(hdf5_path: Path,
                 output_path: Path,
                 joint: int,
                 start: int,
                 end: int) -> None:
    times, joint_angles, ik_fail_mask = load_data(
        hdf5_path, joint, start, end
    )
    make_plot(times, joint_angles, ik_fail_mask, output_path)


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()

    if input_path.is_file():
        if args.output is None:
            raise ValueError("单文件模式需要通过 --output 指定输出图片路径。")
        process_file(
            input_path,
            args.output.expanduser().resolve(),
            args.joint,
            args.start,
            args.end,
        )
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if args.output_dir is None:
        raise ValueError("目录模式需要通过 --output-dir 指定输出根目录。")

    output_root = args.output_dir.expanduser().resolve()
    hdf5_files = sorted(input_path.rglob(args.pattern))
    if not hdf5_files:
        print(f"目录中未找到匹配的 HDF5 文件: {input_path} (模式: {args.pattern})")
        return

    for h5 in hdf5_files:
        rel = h5.relative_to(input_path)
        out_path = (output_root / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_file(h5, out_path, args.joint, args.start, args.end)


if __name__ == "__main__":
    main()
