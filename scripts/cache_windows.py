#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from emg2pose.data import Emg2PoseSessionData, WindowedEmgDataset
from emg2pose.utils import generate_hydra_config_from_overrides
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute EMG windows into memory-mapped arrays for fast training/eval."
        )
    )
    parser.add_argument(
        "--config-name",
        default="base",
        help="Hydra config name to load (default: base).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Optional path to the Hydra config directory. Defaults to the project's "
            "config/ folder."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional Hydra overrides, e.g. experiment=tracking_vemg2pose.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory where cached arrays and manifests will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Dataset splits to cache (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files for the selected splits.",
    )
    parser.add_argument(
        "--version-base",
        default="1.1",
        help="Hydra version_base forwarded to config loading (default: 1.1).",
    )
    return parser.parse_args()


def resolve_config_path(config_path: str | None) -> str:
    if config_path is not None:
        return str(Path(config_path).expanduser().resolve())
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root.joinpath("config"))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_sessions(data_root: Path, session_names: Iterator[str]) -> list[Path]:
    paths = []
    if session_names is None:
        return paths
    for name in session_names:
        candidate = data_root.joinpath(f"{name}.hdf5")
        if not candidate.exists():
            raise FileNotFoundError(f"Missing session file: {candidate}")
        paths.append(candidate)
    return paths


def compute_total_windows(
    session_paths: list[Path],
    *,
    window_length: int,
    padding: tuple[int, int],
    skip_ik_failures: bool,
) -> tuple[int, list[int]]:
    counts: list[int] = []
    total = 0
    for session_path in tqdm(
        session_paths,
        desc="Counting windows",
        unit="session",
    ):
        dataset = WindowedEmgDataset(
            session_path,
            window_length=window_length,
            stride=None,
            padding=padding,
            jitter=False,
            skip_ik_failures=skip_ik_failures,
            allow_mask_recompute=bool(datamodule_cfg.get("allow_mask_recompute", True)),
            treat_interpolated_as_valid=bool(
                datamodule_cfg.get("treat_interpolated_as_valid", True)
            ),
        )
        count = len(dataset)
        counts.append(count)
        total += count
        dataset.session._file.close()  # close HDF5 handle
    return total, counts


def create_memmaps(
    split_dir: Path,
    total_windows: int,
    *,
    effective_length: int,
    emg_channels: int,
    joint_dims: int,
    overwrite: bool,
) -> dict[str, np.memmap]:
    if split_dir.exists() and overwrite:
        for child in split_dir.iterdir():
            if child.is_file():
                child.unlink()
    split_dir.mkdir(parents=True, exist_ok=True)

    emg_path = split_dir.joinpath("emg.f32")
    angles_path = split_dir.joinpath("joint_angles.f32")
    mask_path = split_dir.joinpath("no_ik_mask.u1")

    if not overwrite and any(path.exists() for path in (emg_path, angles_path, mask_path)):
        raise FileExistsError(
            f"Cache files already exist in {split_dir}. Use --overwrite to replace them."
        )

    emg_map = np.memmap(
        emg_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, effective_length, emg_channels),
    )
    angles_map = np.memmap(
        angles_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_windows, effective_length, joint_dims),
    )
    mask_map = np.memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_windows, effective_length),
    )
    return {"emg": emg_map, "angles": angles_map, "mask": mask_map}


def write_split_cache(
    split_name: str,
    session_paths: list[Path],
    window_length: int,
    padding: tuple[int, int],
    skip_ik_failures: bool,
    output_dir: Path,
    overwrite: bool,
) -> None:
    if not session_paths:
        print(f"[{split_name}] No sessions configured. Skipping.")
        return

    effective_length = window_length + padding[0] + padding[1]
    print(
        f"[{split_name}] Preparing cache "
        f"(window={window_length}, padding={padding}, effective_len={effective_length})"
    )

    total_windows, per_session_counts = compute_total_windows(
        session_paths,
        window_length=window_length,
        padding=padding,
        skip_ik_failures=skip_ik_failures,
    )

    if total_windows == 0:
        print(f"[{split_name}] No valid windows found. Nothing to cache.")
        return

    with Emg2PoseSessionData(session_paths[0]) as example_session:
        emg_info = example_session.timeseries.dtype.fields[
            Emg2PoseSessionData.EMG
        ][0]
        angles_info = example_session.timeseries.dtype.fields[
            Emg2PoseSessionData.JOINT_ANGLES
        ][0]
        emg_channels = int(emg_info.shape[0])
        joint_dims = int(angles_info.shape[0])

    split_dir = output_dir.joinpath(split_name)
    maps = create_memmaps(
        split_dir,
        total_windows,
        effective_length=effective_length,
        emg_channels=emg_channels,
        joint_dims=joint_dims,
        overwrite=overwrite,
    )

    manifest_rows = []
    write_index = 0

    for session_idx, session_path in enumerate(
        tqdm(session_paths, desc=f"[{split_name}] caching", unit="session")
    ):
        if per_session_counts[session_idx] == 0:
            continue

        dataset = WindowedEmgDataset(
            session_path,
            window_length=window_length,
            stride=None,
            padding=padding,
            jitter=False,
            skip_ik_failures=skip_ik_failures,
        )
        session = dataset.session
        metadata = dict(session.metadata)

        for local_idx, (offset, _) in enumerate(dataset.windows):
            window_start = max(offset - dataset.left_padding, 0)
            window_end = offset + dataset.window_length + dataset.right_padding
            window = session[window_start:window_end]

            emg = window[Emg2PoseSessionData.EMG]
            joint_angles = window[Emg2PoseSessionData.JOINT_ANGLES]
            mask = session.no_ik_failure[window_start:window_end]

            if emg.shape[0] != effective_length:
                dataset.session._file.close()
                raise RuntimeError(
                    "Encountered window with unexpected length "
                    f"(expected {effective_length}, got {emg.shape[0]}). "
                    "Consider adjusting padding or window parameters."
                )

            maps["emg"][write_index] = emg.astype(np.float32, copy=False)
            maps["angles"][write_index] = joint_angles.astype(np.float32, copy=False)
            maps["mask"][write_index] = mask.astype(np.uint8, copy=False)

            manifest_rows.append(
                {
                    "split": split_name,
                    "global_index": write_index,
                    "session": metadata.get("session"),
                    "file_name": session_path.name,
                    "user": metadata.get("user"),
                    "stage": metadata.get("stage"),
                    "generalization": metadata.get("generalization"),
                    "side": metadata.get("side"),
                    "window_start_idx": window_start,
                    "window_end_idx": window_end,
                    "window_offset": offset,
                    "window_length": dataset.window_length,
                    "effective_length": effective_length,
                    "emg_channels": emg_channels,
                    "joint_dims": joint_dims,
                }
            )

            write_index += 1

        dataset.session._file.close()

    for mmap in maps.values():
        mmap.flush()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = split_dir.joinpath("manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(
        f"[{split_name}] Cached {write_index} windows across "
        f"{len(session_paths)} sessions into {split_dir}"
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config_path)
    ensure_output_dir(args.output_dir)

    config = generate_hydra_config_from_overrides(
        config_path=config_path,
        version_base=args.version_base,
        config_name=args.config_name,
        overrides=args.overrides,
    )

    data_root = Path(config.data_location).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(f"Configured data_location does not exist: {data_root}")

    datamodule_cfg = config.datamodule
    window_length = int(datamodule_cfg.window_length)
    padding = tuple(int(x) for x in datamodule_cfg.get("padding", [0, 0]))
    skip_ik_failures = bool(datamodule_cfg.get("skip_ik_failures", False))
    val_test_window_length = int(
        datamodule_cfg.get("val_test_window_length", window_length)
    )

    split_cfg = instantiate(config.data_split)
    if isinstance(split_cfg, dict):
        splits = split_cfg
    else:
        splits = OmegaConf.to_container(split_cfg, resolve=True) or {}

    split_to_sessions = {
        split: list_sessions(data_root, splits.get(split, []))
        for split in ("train", "val", "test")
    }

    for split in args.splits:
        if split == "train":
            split_window = window_length
            split_padding = padding
        elif split == "val":
            split_window = val_test_window_length
            split_padding = padding
        else:  # test
            split_window = val_test_window_length
            split_padding = (0, 0)

        write_split_cache(
            split,
            split_to_sessions.get(split, []),
            split_window,
            split_padding,
            skip_ik_failures,
            args.output_dir,
            args.overwrite,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
