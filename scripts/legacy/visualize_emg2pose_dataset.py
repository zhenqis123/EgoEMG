#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np
import plotly.io as pio
import torch

import egoemg.visualization as visualization
from egoemg.datasets.emg2pose_dataset_legacy import Emg2PoseSessionData
from egoemg.datasets.pimforce_dataset import _pimforce_to_emg2pose_angles
from egoemg.lightning import EmgPredictionModule
from egoemg.utils import generate_hydra_config_from_overrides


def _load_emg2pose_session(
    path: Path,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    session = Emg2PoseSessionData(path)
    if stop <= 0:
        stop = len(session)
    window = session[start:stop]
    emg = window["emg"]
    joint_angles = window["joint_angles"]
    no_ik_failure = session.no_ik_failure[start:stop]
    return emg, joint_angles, no_ik_failure


def _load_pimforce_raw(
    root_dir: Path,
    index: int,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata_path = root_dir / "pimforce_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    if index < 0:
        raise IndexError("Index must be non-negative.")

    row = None
    with metadata_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, record in enumerate(reader):
            if i == index:
                row = record
                break
    if row is None:
        raise IndexError(f"Index {index} out of range for {metadata_path}")

    output_rel = Path(row["output_path"])
    sample_path = root_dir / output_rel
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample file: {sample_path}")

    combined = np.load(sample_path, mmap_mode="r")
    num_emg = int(row["num_emg_channels"])
    num_joint = int(row["num_joint_channels"])

    if combined.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {combined.shape} in {sample_path}")

    emg = combined[:num_emg, :].T
    joint_angles = combined[num_emg : num_emg + num_joint, :].T
    if stop <= 0:
        stop = emg.shape[0]
    emg = emg[start:stop]
    joint_angles = joint_angles[start:stop].copy()
    if joint_angles.shape[1] != 20:
        raise ValueError(
            "Expected 20 PiMforce joint angles, got "
            f"{joint_angles.shape[1]} in {sample_path}"
        )
    # Match PiMforce kinematics: thumb CMC angles are offset in the raw data.
    joint_angles[:, 0] -= 20.0  # thumb spread/AA
    joint_angles[:, 1] -= 60.0  # thumb flexion/FE
    joint_angles = _pimforce_to_emg2pose_angles(
        joint_angles.T, pose_in_degrees=True
    ).T
    no_ik_failure = np.ones(emg.shape[0], dtype=bool)
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
    # pio.renderers.default = "browser"
    parser = argparse.ArgumentParser(
        description="Visualize emg2pose sessions or PiMforce processed_raw samples."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["emg2pose", "pimforce_raw"],
        default="emg2pose",
        help="Dataset type to visualize.",
    )
    parser.add_argument("--session", type=Path, default=None, help="Path to session hdf5.")
    parser.add_argument(
        "--pimforce-root",
        type=Path,
        default=None,
        help="Root directory for PiMforce processed_raw dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index for PiMforce dataset.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--stop", type=int, default=-1, help="Stop index.")
    parser.add_argument("--native-fs", type=int, default=2000, help="Native sample rate.")
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
    parser.add_argument(
        "--show-compare",
        action="store_true",
        help="Show GT/pred comparison videos inline (requires display).",
    )
    parser.add_argument(
        "--ik-failure-plot",
        action="store_true",
        help="Render the IK failure plot for emg2pose sessions.",
    )
    args = parser.parse_args()

    if args.dataset == "emg2pose":
        if args.session is None:
            raise ValueError("--session is required for dataset=emg2pose")
        emg, joint_angles, no_ik_failure = _load_emg2pose_session(
            args.session, args.start, args.stop
        )
        if args.ik_failure_plot:
            visualization.ik_failure_plot(Emg2PoseSessionData(args.session))
    else:
        if args.pimforce_root is None:
            raise ValueError("--pimforce-root is required for dataset=pimforce_raw")
        emg, joint_angles, no_ik_failure = _load_pimforce_raw(
            args.pimforce_root,
            args.index,
            args.start,
            args.stop,
        )
    stride = max(int(round(args.native_fs / args.target_fs)), 1)
    effective_fs = args.native_fs / stride
    joint_angles_vis = _interval_sample(joint_angles, stride)
    num_frames = joint_angles_vis.shape[0]
    frame_duration_ms = max(int(round(1000.0 / effective_fs)), 1)
    output_base = args.output or Path("/tmp/emg2pose_vis.html")
    gt_output = output_base.with_name(output_base.stem + "_gt.html")
    _plot_mesh_sequence(
        joint_angles_vis,
        num_frames,
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
            num_frames,
            color="lightpink",
            output=pred_output,
            show=args.show,
            frame_duration_ms=frame_duration_ms,
        )

        if args.show_compare:
            gt_frames = visualization.joint_angles_to_frames_parallel(
                joint_angles_vis[:num_frames], color="gray"
            )
            pred_frames = visualization.joint_angles_to_frames_parallel(
                preds_vis[:num_frames], color="lightpink"
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
