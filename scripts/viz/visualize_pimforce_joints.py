#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to visualize PiMforce dataset joint angles over time.
Each joint gets its own plot with time on x-axis and angle in degrees on y-axis.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from egoemg.constants import JOINTS
from egoemg.datasets.pimforce_dataset import _pimforce_to_emg2pose_angles

matplotlib.use("Agg")


def _load_processed_raw_sample(
    root_dir: Path,
    index: int,
    start: int,
    stop: int | None,
) -> np.ndarray:
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
    joint_angles = combined[num_emg : num_emg + num_joint, :].T
    if stop is None or stop <= 0:
        stop = joint_angles.shape[0]
    joint_angles = joint_angles[start:stop].copy()
    if joint_angles.shape[1] != 20:
        raise ValueError(
            "Expected 20 PiMforce joint angles, got "
            f"{joint_angles.shape[1]} in {sample_path}"
        )
    # Match PiMforce kinematics: thumb CMC angles are offset in the raw data.
    joint_angles[:, 0] -= 20.0  # thumb spread/AA
    joint_angles[:, 1] -= 60.0  # thumb flexion/FE
    joint_angles = np.rad2deg(
        _pimforce_to_emg2pose_angles(joint_angles.T, pose_in_degrees=True)
    ).T
    return joint_angles


def visualize_pimforce_session(
    root_dir: Path,
    output_dir: Path,
    session_index: int = 0,
    start: int = 0,
    stop: int | None = None,
    fs: int = 2000,  # Sampling frequency in Hz
):
    """
    Visualize joint angles for a specific session in the PiMforce dataset.
    
    Args:
        root_dir: Root directory of the PiMforce dataset
        output_dir: Directory to save the plots
        session_index: Index of the session to visualize
        start: Start index for the session
        stop: Stop index for the session
        fs: Sampling frequency in Hz (default 2000 Hz)
        pose_mode: Pose mode for PiMforce samples ('last' or 'sequence')
        pose_in_degrees: Whether poses are in degrees (default True)
    """
    print(f"Loading PiMforce processed_raw sample {session_index}...")

    joint_angles = _load_processed_raw_sample(
        root_dir=root_dir,
        index=session_index,
        start=start,
        stop=stop,
    )
    joint_angles = joint_angles.T  # (num_joints, time_steps)
    
    # Calculate time vector
    time_steps = joint_angles.shape[1]
    time_vector = np.arange(time_steps) / fs  # Time in seconds
    
    print(f"Joint angles shape: {joint_angles.shape}")
    print(f"Time vector shape: {time_vector.shape}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each joint separately
    for joint_idx, joint_info in enumerate(JOINTS):
        if joint_idx >= joint_angles.shape[0]:
            print(f"Warning: Joint index {joint_idx} exceeds available joints ({joint_angles.shape[0]})")
            continue
            
        joint_name = joint_info.name
        joint_data = joint_angles[joint_idx, :]  # Shape: (time_steps,)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector, joint_data, linewidth=1.0)
        plt.title(f"Joint Angle Over Time - {joint_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the plot
        output_path = output_dir / f"{joint_name}_session_{session_index}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Saved plot for {joint_name} to {output_path}")


def visualize_pimforce_multiple_sessions(
    root_dir: Path,
    output_dir: Path,
    session_indices: list = None,
    start: int = 0,
    stop: int | None = None,
    fs: int = 2000,
):
    """
    Visualize joint angles for multiple sessions in the PiMforce dataset.
    
    Args:
        root_dir: Root directory of the PiMforce dataset
        output_dir: Directory to save the plots
        session_indices: List of session indices to visualize (default: [0])
        start: Start index for each session
        stop: Stop index for each session
        fs: Sampling frequency in Hz (default 2000 Hz)
        pose_mode: Pose mode for PiMforce samples ('last' or 'sequence')
        pose_in_degrees: Whether poses are in degrees (default True)
    """
    if session_indices is None:
        session_indices = [0]
    
    for session_idx in session_indices:
        session_output_dir = output_dir / f"session_{session_idx}"
        visualize_pimforce_session(
            root_dir=root_dir,
            output_dir=session_output_dir,
            session_index=session_idx,
            start=start,
            stop=stop,
            fs=fs,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize PiMforce dataset joint angles over time.")
    parser.add_argument(
        "--pimforce-root",
        type=Path,
        required=True,
        help="Root directory for PiMforce processed_raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pimforce_visualizations"),
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--session-indices",
        type=str,
        default="0",
        help="Comma-separated list of session indices to visualize (default: 0).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for the session.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop index for the session.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=2000,
        help="Sampling frequency in Hz (default: 2000).",
    )
    
    args = parser.parse_args()
    
    # Parse session indices
    session_indices = [int(x.strip()) for x in args.session_indices.split(',')]
    
    print(f"Visualizing PiMforce dataset from {args.pimforce_root}")
    print(f"Session indices: {session_indices}")
    print(f"Output directory: {args.output_dir}")
    
    visualize_pimforce_multiple_sessions(
        root_dir=args.pimforce_root,
        output_dir=args.output_dir,
        session_indices=session_indices,
        start=args.start,
        stop=args.stop,
        fs=args.fs,
    )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
