#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""使用训练好的checkpoint可视化预测结果，对比真实值和重建的手部mesh。

该脚本加载指定的checkpoint和session文件，生成预测结果，
并使用可视化工具对比真实值和预测值。
"""

from __future__ import annotations

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
import hydra

from emg2pose.datasets.emg2pose_dataset import Emg2PoseSessionData
from emg2pose.lightning import EmgPredictionModule
from emg2pose import transforms
from hydra.utils import instantiate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练好的checkpoint可视化预测结果，对比真实值和重建的手部mesh。"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练好的checkpoint文件路径 (.ckpt)",
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="要可视化的session HDF5文件路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="训练时使用的配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径 (可选，如果不提供则显示在浏览器中)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=250,
        help="要可视化的样本数量，默认250",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=67,  # 2000Hz -> ~30Hz
        help="下采样因子，默认67 (2000Hz -> ~30Hz)",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, config_path: Path):
    """从checkpoint加载模型"""
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 加载模型
    module = EmgPredictionModule.load_from_checkpoint(
        str(checkpoint_path),
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        loss_weights=config.loss_weights,
    )
    
    module.eval()
    return module


def prepare_session_data(session_path: Path, window_length: int = 10000):
    """准备session数据用于推理"""
    session_data = Emg2PoseSessionData(session_path)
    
    # 创建窗口化数据集
    from emg2pose.datasets.emg2pose_dataset import WindowedEmgDataset
    
    # 使用基本的变换
    basic_transform = transforms.Compose([
        transforms.ExtractField(field="emg"),
        transforms.ToFloatTensor()
    ])
    
    dataset = WindowedEmgDataset(
        hdf5_path=session_path,
        window_length=window_length,
        stride=window_length,  # 避免重叠
        padding=(0, 0),
        transform=basic_transform,
        skip_ik_failures=False
    )
    
    return dataset, session_data


def run_inference(model: EmgPredictionModule, dataset, device='cpu'):
    """运行推理获取预测结果"""
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i in range(min(len(dataset), 10)):  # 限制处理的数据量
            batch = dataset[i]
            # 将batch转换为模型期望的格式
            batch_tensor = {
                'emg': batch['emg'].unsqueeze(0).to(device),
                'joint_angles': batch['joint_angles'].unsqueeze(0).to(device),
                'label_valid_mask': batch['label_valid_mask'].unsqueeze(0).to(device)
            }
            
            try:
                pred, target, mask = model.forward(batch_tensor)
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
            except Exception as e:
                print(f"推理第{i}个批次时出错: {e}")
                continue
    
    if predictions:
        predictions = np.concatenate(predictions, axis=-1)  # 在时间维度上连接
        targets = np.concatenate(targets, axis=-1)
        return predictions.squeeze(), targets.squeeze()
    else:
        return None, None


def downsample_data(data: np.ndarray, factor: int):
    """下采样数据"""
    return data[:, ::factor]


def visualize_predictions(gt_joint_angles: np.ndarray, pred_joint_angles: np.ndarray, output_path: Path | None = None):
    """可视化真实值和预测值的对比"""
    import emg2pose.visualization as visualization
    
    # 下采样到合适的帧率
    gt_downsampled = downsample_data(gt_joint_angles, 67)  # 2000Hz -> ~30Hz
    pred_downsampled = downsample_data(pred_joint_angles, 67)
    
    # 限制可视化序列长度
    seq_len = min(250, gt_downsampled.shape[1], pred_downsampled.shape[1])
    gt_vis = gt_downsampled[:, :seq_len]
    pred_vis = pred_downsampled[:, :seq_len]
    
    # 创建真实值动画
    print("生成真实值动画...")
    gt_fig = visualization.get_plotly_animation_for_joint_angles(
        gt_vis.T, color="gray", opacity=0.7
    )
    gt_fig.update_layout(title="Ground Truth Hand Mesh")
    
    # 创建预测值动画
    print("生成预测值动画...")
    pred_fig = visualization.get_plotly_animation_for_joint_angles(
        pred_vis.T, color="lightpink", opacity=0.7
    )
    pred_fig.update_layout(title="Predicted Hand Mesh")
    
    # 如果需要保存
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gt_output = output_path.with_name(output_path.stem + "_gt" + output_path.suffix)
        pred_output = output_path.with_name(output_path.stem + "_pred" + output_path.suffix)
        
        gt_fig.write_html(str(gt_output))
        pred_fig.write_html(str(pred_output))
        print(f"真实值动画已保存到: {gt_output}")
        print(f"预测值动画已保存到: {pred_output}")
    
    # 显示动画
    print("显示真实值动画...")
    gt_fig.show()
    
    print("显示预测值动画...")
    pred_fig.show()
    
    # 也可以创建并排对比
    print("生成对比帧...")
    gt_frames = visualization.joint_angles_to_frames_parallel(gt_vis.T, color="gray")
    pred_frames = visualization.joint_angles_to_frames_parallel(pred_vis.T, color="lightpink")
    
    gt_frames = visualization.remove_alpha_channel(gt_frames)
    pred_frames = visualization.remove_alpha_channel(pred_frames)
    
    return gt_frames, pred_frames


def main() -> None:
    args = parse_args()
    
    print(f"加载模型从: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, args.config)
    
    print(f"加载会话数据从: {args.session}")
    dataset, session_data = prepare_session_data(args.session)
    
    print("运行推理...")
    predictions, targets = run_inference(model)
    
    if predictions is not None and targets is not None:
        print(f"预测结果形状: {predictions.shape}")
        print(f"真实值形状: {targets.shape}")
        
        print("生成可视化...")
        gt_frames, pred_frames = visualize_predictions(targets, predictions, args.output)
        
        print("可视化完成!")
    else:
        print("推理失败，无法生成可视化。")


if __name__ == "__main__":
    main()