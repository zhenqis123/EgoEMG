#!/usr/bin/env python

"""
Build PiMforce raw EMG and joint angle data from join_*.csv files.

This script processes raw join_*.csv files to extract EMG and joint angle data,
performing the same preprocessing and time alignment as the original pipeline
but without windowing. Each join_*.csv file is saved as a separate .npy file,
and metadata is recorded in a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def _get_session_dirs(root: Path) -> list[Path]:
    """Get all session directories."""
    session_re = re.compile(r"Session(\d+)$")
    session_dirs = []

    for path in root.iterdir():
        if path.is_dir():
            match = session_re.match(path.name)
            if match:
                session_dirs.append(path)

    # Sort by session number
    session_dirs.sort(key=lambda p: int(re.search(r"Session(\d+)", p.name).group(1)))
    return session_dirs


def _infer_user_id(session_num: int) -> int:
    """
    Infer user ID from session number.
    Every 3 sessions belong to one user (Session 1-3 -> User 1, Session 4-6 -> User 2, etc.)
    """
    return ((session_num - 1) // 3) + 1


def _get_join_files(session_dir: Path) -> list[Path]:
    """Get all join_*.csv files in a session directory."""
    join_re = re.compile(r"join_(\d+)\.csv$")
    join_files = []

    for path in session_dir.iterdir():
        if path.is_file() and join_re.match(path.name):
            join_files.append(path)

    # Sort by file number
    join_files.sort(key=lambda p: int(re.search(r"join_(\d+)\.csv$", p.name).group(1)))
    return join_files


def _get_force_files(session_dir: Path) -> list[Path]:
    """Get all fsr_*.csv files in a session directory."""
    fsr_re = re.compile(r"fsr_(\d+)\.csv$")
    fsr_files = []

    for path in session_dir.iterdir():
        if path.is_file() and fsr_re.match(path.name):
            fsr_files.append(path)

    # Sort by file number
    fsr_files.sort(key=lambda p: int(re.search(r"fsr_(\d+)\.csv$", p.name).group(1)))
    return fsr_files


def _get_pps_files(session_dir: Path) -> list[Path]:
    """Get all pps_*.csv files in a session directory."""
    pps_re = re.compile(r"pps_(\d+)\.csv$")
    pps_files = []

    for path in session_dir.iterdir():
        if path.is_file() and pps_re.match(path.name):
            pps_files.append(path)

    # Sort by file number
    pps_files.sort(key=lambda p: int(re.search(r"pps_(\d+)\.csv$", p.name).group(1)))
    return pps_files


def process_join_file(
    join_file_path: Path,
    fsr_file_path: Path,
    pps_file_path: Path,
    regression_model_path: Path,
    start_time: float = 0.0,
    duration: float = 28.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Process a single join_*.csv file to extract EMG and joint angle data,
    and corresponding fsr_*.csv and pps_*.csv files to extract force data.

    Args:
        join_file_path: Path to the join_*.csv file
        fsr_file_path: Path to the fsr_*.csv file
        pps_file_path: Path to the pps_*.csv file
        regression_model_path: Path to the regression model used for FSR calibration
        start_time: Start time for the segment (default 0.0)
        duration: Duration of the segment (default 28.0 seconds)

    Returns:
        Tuple of (emg_data, joint_angle_data, force_data, original_length).
        Data are sample-aligned at 2000 Hz and force is scaled to match the baseline script.
    """
    emg_fps = 2000

    # Process EMG and joint angle data from join file
    emg = pd.read_csv(join_file_path, dtype=np.float64, float_precision="round_trip")
    emg = emg.iloc[:, 1:]  # Match baseline: drop the first column.

    # Identify EMG signal columns (contain 'sEMG')
    emg_signal_names = [col for col in emg.columns if "sEMG" in col]
    # Identify joint angle columns (contain '[')
    manus_signal_names = [col for col in emg.columns if "[" in col]

    emg_remove = pd.DataFrame(np.array(emg[emg_signal_names]))
    emg = pd.concat([emg.iloc[:, 0], emg_remove, emg[manus_signal_names]], axis=1)

    emg = np.array(emg.transpose())
    emg[0] = np.linspace(start=0, stop=emg[0][-1] - emg[0][0], num=len(emg[0]))
    emg = emg[:, emg[0] != 0.0]  # Remove trailing empty entries.

    begin = 0
    end = 0
    for timestamp in emg[0]:
        if timestamp < start_time:
            begin += 1
        else:
            break
    for timestamp in emg[0][::-1]:
        if timestamp > start_time + duration:
            end -= 1
        else:
            break

    emg = emg[1:, begin:end] if end < 0 else emg[1:, begin:]

    num_emg_channels = len(emg_signal_names)
    emg_data = emg[:num_emg_channels]
    joint_angle_data = emg[num_emg_channels:]
    sample_times = emg[0].astype(np.float32)

    # Process force data from fsr file
    force = pd.read_csv(fsr_file_path, dtype=np.float64, float_precision="round_trip")
    force = force.drop_duplicates(["Timestamp"])  # Remove duplicate timestamps

    marble_names = [column for column in force.columns if "FSR" in column]
    marble = np.array(force[marble_names])

    force = np.concatenate([np.array(force.iloc[:, 0]).reshape(-1, 1), marble], 1)

    force = np.array(force.transpose())
    force[0] = np.linspace(start=0, stop=force[0][-1] - force[0][0], num=len(force[0]))
    force = np.array(force.transpose())

    force_times = force[:, 0]
    force_fsr = force[:, 1:]

    model_path = Path(regression_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Regression model not found: {model_path}")
    with open(model_path, "rb") as model_file:
        predict_model = pickle.load(model_file)

    # Align FSR to EMG samples, then calibrate to Newtons.
    force_fsr_aligned = np.zeros((emg_data.shape[1], force_fsr.shape[1]))
    for i in range(force_fsr.shape[1]):
        force_fsr_aligned[:, i] = np.interp(
            sample_times,
            force_times,
            force_fsr[:, i],
            left=0.0,
            right=0.0,
        )

    poly_features = PolynomialFeatures(degree=5, include_bias=False)
    force_newton = []
    for i in range(force_fsr_aligned.shape[1]):
        force_poly = poly_features.fit_transform(force_fsr_aligned[:, i].reshape(-1, 1))
        force_newton.append(predict_model.predict(force_poly))

    force_newton = np.hstack(force_newton) * 0.5

    # Process PPS data
    pps_force = pd.read_csv(pps_file_path, dtype=np.float64, float_precision="round_trip")
    pps_force = pps_force.drop_duplicates(["Time [ms]"])

    ref_time = np.linspace(start=pps_force.iloc[0, 0], stop=pps_force.iloc[-1, 0], num=2975)
    pps_time = pps_force.iloc[:, 0]

    ref_time = pd.to_datetime(ref_time, unit="s")
    ref_time = pd.DataFrame(ref_time)
    ref_time.columns = ["Time [ms]"]
    pps_force.iloc[:, 0] = pd.to_datetime(pps_time, unit="s")

    force_interpolation = pd.merge_asof(
        ref_time, pps_force, on="Time [ms]", direction="nearest", tolerance=pd.Timedelta("0.01s")
    )

    for column in force_interpolation.columns:
        if column != "Time [ms]":
            force_interpolation[column] = force_interpolation[column].interpolate(
                method="linear"
            )

    force_interpolation.iloc[:, 0] = force_interpolation.iloc[:, 0].values.astype("float64")
    pps_force = force_interpolation

    palm_rightup = ["Elem15", "Elem16"]
    palm_leftup = ["Elem17", "Elem18"]
    palm_rightdown = ["Elem0", "Elem1", "Elem5", "Elem6", "Elem9", "Elem10"]
    palm_leftdown = ["Elem2", "Elem3", "Elem7", "Elem8", "Elem11", "Elem12"]

    # Definition of fingers and palm (match baseline).
    pps_force["V5"] = pps_force[palm_rightup].max(axis=1)
    pps_force["V6"] = pps_force[palm_rightdown].max(axis=1)
    pps_force["V7"] = pps_force[palm_leftup].max(axis=1)
    pps_force["V8"] = pps_force[palm_leftdown].max(axis=1)

    pps_names = [column for column in pps_force.columns if "V" in column]

    force_pps = np.array(pps_force[pps_names])

    force_pps[force_pps < 0] = 0

    force_pps = np.concatenate([np.array(pps_force.iloc[:, 0]).reshape(-1, 1), force_pps], 1)

    force_pps = np.array(force_pps.transpose())
    force_pps[0] = np.linspace(start=0, stop=30, num=len(force_pps[0]))

    force_pps = np.array(force_pps.transpose())

    pps_times = force_pps[:, 0]
    force_pps_values = force_pps[:, 1:]

    # Align PPS to EMG samples, then apply calibration.
    force_pps_aligned = np.zeros((emg_data.shape[1], force_pps_values.shape[1]))
    for i in range(force_pps_values.shape[1]):
        force_pps_aligned[:, i] = np.interp(
            sample_times,
            pps_times,
            force_pps_values[:, i],
            left=0.0,
            right=0.0,
        )

    force_pps = force_pps_aligned / 2
    force_pps[force_pps < 0.8] = 0
    force_pps = force_pps * 2
    force_pps = force_pps * 0.689009

    force_newton = np.concatenate([force_newton, force_pps], 1)

    force_newton[force_newton < 0.2] = 0
    force_newton[force_newton > 20] = 20

    scale = 8
    force_newton = force_newton / scale

    original_length = emg_data.shape[1]

    return emg_data, joint_angle_data, force_newton.T, original_length


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build PiMforce raw EMG, joint angle, and force data from join_*.csv files. "
            "Each join_*.csv file is saved as a separate .npy file with metadata in a CSV."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/Dataset_NeurIPS2024/Dataset"),
        help="Root directory containing Session*/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/processed_raw"),
        help="Output directory for processed .npy files and metadata CSV.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("pimforce_metadata.csv"),
        help="Name of the metadata CSV file.",
    )
    parser.add_argument(
        "--regression-model",
        type=Path,
        default=Path("Checkpoints/regression_model.sav"),
        help="Path to the regression model used for FSR calibration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time for data segment (default 0.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=28.0,
        help="Duration of data segment (default 28.0 seconds)",
    )
    args = parser.parse_args()

    # Validate input
    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

    session_dirs = _get_session_dirs(data_root)
    if not session_dirs:
        raise FileNotFoundError(f"No Session*/ folders found in {data_root}")

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata file
    metadata_path = output_dir / args.metadata_file
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} exists (use --overwrite to replace).")

    # Prepare metadata CSV
    metadata_rows = []

    # Process each session and its join files
    for session_dir in session_dirs:
        session_name = session_dir.name
        session_num = int(re.search(r"Session(\d+)", session_name).group(1))

        # Infer user ID based on session number (every 3 sessions belong to one user)
        user_id = _infer_user_id(session_num)

        join_files = _get_join_files(session_dir)
        fsr_files = _get_force_files(session_dir)
        pps_files = _get_pps_files(session_dir)

        for i, join_file in enumerate(join_files):
            if i >= len(fsr_files) or i >= len(pps_files):
                print(
                    f"Warning: Not enough force or PPS files for {session_name}. Skipping remaining files."
                )
                break

            file_name = join_file.name
            file_num = int(re.search(r"join_(\d+)\.csv", file_name).group(1))

            fsr_file = fsr_files[i]
            pps_file = pps_files[i]

            print(f"Processing {session_name}/{file_name}...")

            # Process the join file along with corresponding force files
            emg_data, joint_angle_data, force_data, original_length = process_join_file(
                join_file,
                fsr_file,
                pps_file,
                args.regression_model,
                start_time=args.start_time,
                duration=args.duration,
            )

            # Create combined data array (EMG + joint angles + force)
            combined_data = np.concatenate([emg_data, joint_angle_data, force_data], axis=0)

            # Generate output filename
            output_filename = f"{session_name}_{file_name.replace('.csv', '.npy')}"
            output_path = output_dir / output_filename

            # Save the combined data
            np.save(output_path, combined_data)

            # Record metadata
            metadata_rows.append(
                {
                    "user_id": user_id,
                    "session_id": session_num,
                    "file_id": file_num,
                    "session_name": session_name,
                    "filename": file_name,
                    "output_path": str(output_path.relative_to(output_dir)),
                    "original_length": original_length,
                    "num_emg_channels": emg_data.shape[0],
                    "num_joint_channels": joint_angle_data.shape[0],
                    "num_force_channels": force_data.shape[0],
                    "total_channels": combined_data.shape[0],
                    "shape": str(combined_data.shape),
                    "start_time": args.start_time,
                    "duration": args.duration,
                }
            )

    # Write metadata CSV
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "user_id",
            "session_id",
            "file_id",
            "session_name",
            "filename",
            "output_path",
            "original_length",
            "num_emg_channels",
            "num_joint_channels",
            "num_force_channels",
            "total_channels",
            "shape",
            "start_time",
            "duration",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)

    print("\nProcessing complete!")
    print(f"Processed {len(metadata_rows)} files.")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Data files saved to: {output_dir}")


if __name__ == "__main__":
    main()
