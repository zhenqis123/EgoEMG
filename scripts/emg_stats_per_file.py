#!/usr/bin/env python
#
# Compute per-file, per-channel EMG statistics over the full dataset.
# Outputs a CSV with one row per (file, channel).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import multiprocessing as mp

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-file EMG channel statistics (full pass)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/emg2pose_data"),
        help="Root directory containing session HDF5 files.",
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
        default=Path("logs/emg_stats_per_file.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of worker processes for HDF5 reading/stats (default: cpu_count-1).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Chunk size for multiprocessing imap (tune for throughput/memory).",
    )
    return parser.parse_args()


def compute_channel_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    total = values.shape[0]
    n_finite = int(finite.sum())
    finite_frac = float(n_finite) / float(total) if total > 0 else 0.0

    if n_finite == 0:
        return {
            "n_total": total,
            "n_finite": n_finite,
            "finite_frac": finite_frac,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "rms": np.nan,
            "energy": np.nan,
            "sum_abs": np.nan,
        }

    v = values[finite].astype(np.float64, copy=False)
    energy = float(np.sum(v ** 2))
    mean = float(np.mean(v))
    std = float(np.std(v))
    p05 = float(np.percentile(v, 5))
    p95 = float(np.percentile(v, 95))
    return {
        "n_total": total,
        "n_finite": n_finite,
        "finite_frac": finite_frac,
        "mean": mean,
        "std": std,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p05": p05,
        "p95": p95,
        "rms": float(np.sqrt(np.mean(v ** 2))),
        "energy": energy,
        "sum_abs": float(np.sum(np.abs(v))),
    }


def _process_file(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker helper to compute stats for one file."""
    path: Path = entry["path"]
    row = entry["meta"]
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with h5py.File(path, "r") as f:
        ts = f["emg2pose"]["timeseries"]
        emg = np.asarray(ts["emg"][...], dtype=np.float32)  # (T, 16)

    for ch in range(emg.shape[1]):
        stats = compute_channel_stats(emg[:, ch])
        rec = {
            "file": path.name,
            "user": row.get("user", None),
            "stage": row.get("stage", None),
            "side": row.get("side", None),
            "channel": ch,
        }
        rec.update(stats)
        records.append(rec)
    return records


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root.expanduser().resolve()
    metadata_path: Path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else data_root.joinpath("metadata.csv")
    )
    output_path: Path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(metadata_path)
    tasks = []
    for _, row in df_meta.iterrows():
        filename = (
            f"{row['filename']}.hdf5"
            if not str(row["filename"]).endswith(".hdf5")
            else str(row["filename"])
        )
        h5_path = data_root.joinpath(filename)
        tasks.append({"path": h5_path, "meta": row.to_dict()})

    records: list[dict[str, Any]] = []
    if args.num_workers and args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            for recs in tqdm(
                pool.imap_unordered(_process_file, tasks, chunksize=args.chunksize),
                total=len(tasks),
                desc="Files",
            ):
                records.extend(recs)
    else:
        for task in tqdm(tasks, total=len(tasks), desc="Files"):
            records.extend(_process_file(task))

    if not records:
        print("No records generated; check dataset paths.")
        return

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
