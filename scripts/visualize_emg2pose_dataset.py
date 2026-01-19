#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.io as pio
import torch

import emg2pose.visualization as visualization
from emg2pose.datasets.emg2pose_dataset import Emg2PoseSessionData
from emg2pose.datasets.pimforce_dataset import WindowedPiMforceDataset
from emg2pose.lightning import EmgPredictionModule
from emg2pose.utils import generate_hydra_config_from_overrides


def _load_session(path: Path, start: int, stop: int):
    session = Emg2PoseSessionData(path)
    window = session[start:stop]
    emg = window["emg"]
    joint_angles = window["joint_angles"]
    no_ik_failure = session.no_ik_failure[start:stop]
    return emg, joint_angles, no_ik_failure


def _load_pimforce_sample(
    root_dir: Path,
    index: int,
    start: int,
    stop: int,
    pose_mode: str,
    pose_in_degrees: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = WindowedPiMforceDataset(
        root_dir=root_dir,
        pose_mode=pose_mode,
        pose_in_degrees=pose_in_degrees,
        window_start=start,
        window_stop=stop,
        clip_to_valid=True,
    )
    sample = dataset[index]
    emg = sample["emg"]
    joint_angles = sample["joint_angles"]
    if torch.is_tensor(emg):
        emg = emg.detach().cpu().numpy()
    if torch.is_tensor(joint_angles):
        joint_angles = joint_angles.detach().cpu().numpy()
    emg = emg.T
    joint_angles = joint_angles.T
    valid_mask = sample.get("label_valid_mask")
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.detach().cpu().numpy()
    if valid_mask is None:
        valid_mask = np.ones(emg.shape[0], dtype=bool)
    no_ik_failure = valid_mask.astype(bool)
    return emg, joint_angles, no_ik_failure


def _interval_sample(array: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return array
    return array[::stride]


def _predict(
    checkpoint: Path, experiment: str, emg: np.ndarray, joint_angles: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    config = generate_hydra_config_from_overrides(
        overrides=[f"experiment={experiment}", f"checkpoint={checkpoint}"]
    )
    module = EmgPredictionModule.load_from_checkpoint(
        str(checkpoint),
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        loss_weights=config.loss_weights,
    )
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    batch = {
        "emg": torch.tensor([emg.T], dtype=torch.float32, device=device),  # BCT
        "joint_angles": torch.tensor([joint_angles.T], dtype=torch.float32, device=device),
        "label_valid_mask": torch.tensor([mask], dtype=torch.bool, device=device),
    }
    with torch.no_grad():
        preds, _, _ = module.forward(batch)
    return preds[0].detach().cpu().T.numpy()


def _plot_mesh_sequence(
    joint_angles: np.ndarray,
    num_frames: int,
    color: str,
    output: Path,
    show: bool,
    frame_duration_ms: int | None = None,
):
    fig = visualization.get_plotly_animation_for_joint_angles(
        joint_angles[:num_frames], color=color
    )
    if frame_duration_ms is not None and frame_duration_ms > 0:
        if fig.layout.updatemenus and fig.layout.updatemenus[0].buttons:
            play_args = {
                "frame": {"duration": frame_duration_ms},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": frame_duration_ms, "easing": "linear"},
            }
            pause_args = {
                "frame": {"duration": 0},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": 0, "easing": "linear"},
            }
            fig.layout.updatemenus[0].buttons[0].args = [None, play_args]
            if len(fig.layout.updatemenus[0].buttons) > 1:
                fig.layout.updatemenus[0].buttons[1].args = [[None], pause_args]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    if show:
        fig.show()
    return fig


def main() -> None:
    pio.renderers.default = "browser"
    parser = argparse.ArgumentParser(description="Visualize emg2pose sessions.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["emg2pose", "pimforce"],
        default="emg2pose",
        help="Dataset type to visualize.",
    )
    parser.add_argument("--session", type=Path, default=None, help="Path to session hdf5.")
    parser.add_argument(
        "--pimforce-root",
        type=Path,
        default=None,
        help="Root directory for PiMforce dataset (contains emg_train.npy).",
    )
    parser.add_argument(
        "--pimforce-pose-mode",
        type=str,
        choices=["last", "sequence"],
        default="sequence",
        help="Pose mode for PiMforce samples.",
    )
    parser.add_argument(
        "--pimforce-pose-unit",
        type=str,
        choices=["deg", "rad"],
        default="rad",
        help="Pose unit for PiMforce samples.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index for PiMforce dataset.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--stop", type=int, default=10_000, help="Stop index.")
    parser.add_argument("--num-frames", type=int, default=10000, help="Frames to visualize.")
    parser.add_argument(
        "--native-fs",
        type=int,
        default=2000,
        help="Native sample rate for joint angles (default: 2000 Hz).",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=2000,
        help="Target frame rate for visualization (interval sampling).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Optional checkpoint path."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="tracking_vemg2pose",
        help="Experiment name for checkpoint config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path for saving animations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plotly animations in a browser.",
    )
    args = parser.parse_args()

    if args.dataset == "emg2pose":
        if args.session is None:
            raise ValueError("--session is required for dataset=emg2pose")
        emg, joint_angles, no_ik_failure = _load_session(
            args.session, args.start, args.stop
        )
        visualization.ik_failure_plot(Emg2PoseSessionData(args.session))
    else:
        if args.pimforce_root is None:
            raise ValueError("--pimforce-root is required for dataset=pimforce")
        emg, joint_angles, no_ik_failure = _load_pimforce_sample(
            args.pimforce_root,
            args.index,
            args.start,
            args.stop,
            args.pimforce_pose_mode,
            args.pimforce_pose_unit == "deg",
        )

    stride = max(int(round(args.native_fs / args.target_fs)), 1)
    effective_fs = args.native_fs / stride
    joint_angles_vis = _interval_sample(joint_angles, stride)
    frame_duration_ms = max(int(round(1000.0 / effective_fs)), 1)
    output_base = args.output or Path("/tmp/emg2pose_vis.html")
    gt_output = output_base.with_name(output_base.stem + "_gt.html")
    _plot_mesh_sequence(
        joint_angles_vis,
        args.num_frames,
        color="gray",
        output=gt_output,
        show=args.show,
        frame_duration_ms=frame_duration_ms,
    )

    if args.checkpoint is not None:
        preds = _predict(args.checkpoint, args.experiment, emg, joint_angles, no_ik_failure)
        preds_vis = _interval_sample(preds, stride)
        pred_output = output_base.with_name(output_base.stem + "_pred.html")
        _plot_mesh_sequence(
            preds_vis,
            args.num_frames,
            color="lightpink",
            output=pred_output,
            show=args.show,
            frame_duration_ms=frame_duration_ms,
        )

        gt_frames = visualization.joint_angles_to_frames_parallel(
            joint_angles_vis[: args.num_frames], color="gray"
        )
        pred_frames = visualization.joint_angles_to_frames_parallel(
            preds_vis[: args.num_frames], color="lightpink"
        )
        gt_frames = visualization.remove_alpha_channel(gt_frames)
        pred_frames = visualization.remove_alpha_channel(pred_frames)
        try:
            import mediapy

            mediapy.show_videos(
                {"gt": gt_frames, "pred": pred_frames},
                width=400,
                fps=30,
                downsample=True,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
