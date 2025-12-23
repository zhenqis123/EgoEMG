#!/usr/bin/env python
#
# Compute z-score stats for EMG normalization (global/user/channel/user_channel).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EMG normalization stats (global, user, channel, user-channel)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/emg2pose_data"),
        help="Root directory containing HDF5 session files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.csv (defaults to data-root/metadata.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/emg_norm_stats.npz"),
        help="Output npz for normalization stats.",
    )
    return parser.parse_args()


def _init_channel_stats(n_channels: int) -> dict[str, np.ndarray]:
    return {
        "sum": np.zeros(n_channels, dtype=np.float64),
        "sumsq": np.zeros(n_channels, dtype=np.float64),
        "count": np.zeros(n_channels, dtype=np.int64),
    }


def _update_channel_stats(stats: dict[str, np.ndarray], data: np.ndarray) -> None:
    # data: (T, C)
    stats["sum"] += data.sum(axis=0)
    stats["sumsq"] += (data ** 2).sum(axis=0)
    stats["count"] += data.shape[0]


def _stats_from_accum(stats: dict[str, np.ndarray], eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = stats["sum"] / np.maximum(stats["count"], 1)
    var = (stats["sumsq"] / np.maximum(stats["count"], 1)) - mean ** 2
    std = np.sqrt(np.maximum(var, eps))
    return mean, std


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else data_root.joinpath("metadata.csv")
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(metadata_path)

    # Accumulators
    global_sum = 0.0
    global_sumsq = 0.0
    global_count = 0

    channel_stats = _init_channel_stats(32)  # left+right -> 32 channels
    user_stats: dict[str, dict[str, float]] = {}
    user_channel_stats: dict[str, dict[str, np.ndarray]] = {}

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Files"):
        filename = f"{row['filename']}.hdf5" if not str(row["filename"]).endswith(".hdf5") else str(row["filename"])
        path = data_root.joinpath(filename)
        if not path.exists():
            tqdm.write(f"[WARN] Missing file: {path}")
            continue

        user = str(row.get("user", "unknown"))
        side = str(row.get("side", "unknown")).lower()
        side_offset = 0 if side == "left" else 16

        with h5py.File(path, "r") as f:
            ts = f["emg2pose"]["timeseries"]
            emg = np.asarray(ts["emg"][...], dtype=np.float32)  # (T, 16)

        # Global
        global_sum += float(emg.sum())
        global_sumsq += float((emg ** 2).sum())
        global_count += int(emg.size)

        # Channel (32)
        ch_data = np.zeros((emg.shape[0], 32), dtype=np.float32)
        ch_data[:, side_offset:side_offset + 16] = emg
        _update_channel_stats(channel_stats, ch_data)

        # User global (all channels)
        if user not in user_stats:
            user_stats[user] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
        user_stats[user]["sum"] += float(emg.sum())
        user_stats[user]["sumsq"] += float((emg ** 2).sum())
        user_stats[user]["count"] += int(emg.size)

        # User × channel (32)
        if user not in user_channel_stats:
            user_channel_stats[user] = _init_channel_stats(32)
        _update_channel_stats(user_channel_stats[user], ch_data)

    # Finalize stats
    global_mean = global_sum / max(global_count, 1)
    global_var = (global_sumsq / max(global_count, 1)) - global_mean ** 2
    global_std = float(np.sqrt(max(global_var, 1e-6)))

    channel_mean, channel_std = _stats_from_accum(channel_stats)

    user_mean = {}
    user_std = {}
    for user, acc in user_stats.items():
        mean = acc["sum"] / max(acc["count"], 1)
        var = (acc["sumsq"] / max(acc["count"], 1)) - mean ** 2
        user_mean[user] = mean
        user_std[user] = float(np.sqrt(max(var, 1e-6)))

    user_channel_mean = {}
    user_channel_std = {}
    for user, acc in user_channel_stats.items():
        mean, std = _stats_from_accum(acc)
        user_channel_mean[user] = mean
        user_channel_std[user] = std

    np.savez(
        output_path,
        global_mean=global_mean,
        global_std=global_std,
        channel_mean=channel_mean,
        channel_std=channel_std,
        user_mean=user_mean,
        user_std=user_std,
        user_channel_mean=user_channel_mean,
        user_channel_std=user_channel_std,
    )

    print(f"Wrote normalization stats to {output_path}")


if __name__ == "__main__":
    main()
