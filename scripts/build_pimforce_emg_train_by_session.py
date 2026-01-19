#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SESSION_RE = re.compile(r"Session(\d+)$")
EMG_RE = re.compile(r"emg_(\d+)\.npy$")


def _sorted_session_dirs(root: Path) -> list[Path]:
    sessions: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = SESSION_RE.match(path.name)
        if match:
            sessions.append((int(match.group(1)), path))
    sessions.sort(key=lambda item: item[0])
    return [path for _, path in sessions]


def _sorted_emg_files(session_dir: Path) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in session_dir.iterdir():
        match = EMG_RE.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def _scan_sessions(
    session_dirs: list[Path],
) -> tuple[list[int], int, np.dtype]:
    lengths: list[int] = []
    channels: int | None = None
    dtype: np.dtype | None = None

    for session_dir in session_dirs:
        emg_files = _sorted_emg_files(session_dir)
        if not emg_files:
            raise FileNotFoundError(f"No emg_*.npy found in {session_dir}")

        total_len = 0
        for emg_path in emg_files:
            arr = np.load(emg_path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array in {emg_path}, got {arr.shape}")
            if channels is None:
                channels = int(arr.shape[1])
                dtype = arr.dtype
            elif int(arr.shape[1]) != channels:
                raise ValueError(
                    f"Channel mismatch in {emg_path}: {arr.shape[1]} vs {channels}"
                )
            total_len += int(arr.shape[0])
        lengths.append(total_len)

    if channels is None or dtype is None:
        raise RuntimeError("Failed to infer channels/dtype from sessions.")

    return lengths, channels, dtype


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build emg_train.npy by concatenating emg_*.npy per session in order."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/Dataset_NeurIPS2024/Dataset"),
        help="Root directory containing Session*/ folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for emg_train.npy (defaults to <data-root>/emg_train.npy).",
    )
    parser.add_argument(
        "--lengths-output",
        type=Path,
        default=None,
        help="Optional output for session lengths (defaults to <output>_lengths.npy).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional dtype override (e.g., float32, float64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    session_dirs = _sorted_session_dirs(data_root)
    if not session_dirs:
        raise FileNotFoundError(f"No Session*/ folders found in {data_root}")

    lengths, channels, src_dtype = _scan_sessions(session_dirs)
    out_dtype = np.dtype(args.dtype) if args.dtype else src_dtype
    max_len = max(lengths)

    output_path = args.output or (data_root / "emg_train.npy")
    lengths_path = (
        args.lengths_output
        or output_path.with_name(output_path.stem + "_lengths.npy")
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists (use --overwrite to replace).")
    if lengths_path.exists() and not args.overwrite:
        raise FileExistsError(f"{lengths_path} exists (use --overwrite to replace).")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    emg_mem = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=out_dtype,
        shape=(len(session_dirs), channels, max_len),
    )
    emg_mem[:] = 0

    for session_idx, session_dir in enumerate(session_dirs):
        emg_files = _sorted_emg_files(session_dir)
        offset = 0
        for emg_path in emg_files:
            arr = np.load(emg_path, mmap_mode="r")
            seg_len = int(arr.shape[0])
            seg = np.ascontiguousarray(arr.T, dtype=out_dtype)
            emg_mem[session_idx, :, offset : offset + seg_len] = seg
            offset += seg_len
        if offset != lengths[session_idx]:
            raise RuntimeError(
                f"Length mismatch in {session_dir}: {offset} vs {lengths[session_idx]}"
            )

    emg_mem.flush()
    np.save(lengths_path, np.asarray(lengths, dtype=np.int64))


if __name__ == "__main__":
    main()
