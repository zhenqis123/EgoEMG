from __future__ import annotations

"""
Convert EMG2Pose HDF5 sessions into a single global Zarr store.

Layout:
- emg             (T, C_emg)
- joint_angles    (T, C_joint)
- time            (T,)
- valid_mask      (T,)
- ik_failure_mask (T,)
- sessions/*      per-session index + metadata
- blocks/*        contiguous valid segments in global coordinates
- stats/*         per-session sums/sumsq/count for EMG
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
import sys
from typing import Sequence

import h5py
import numpy as np
from tqdm import tqdm

from egoemg.utils import get_ik_failures_mask

try:
    import zarr
    from zarr import codecs as zarr_codecs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies for Zarr conversion. Please install `zarr` in "
        "your environment."
    ) from exc


EMG2POSE_GROUP = "emg2pose"
TIMESERIES = "timeseries"
IK_FAILURE_MASK = "ik_failure_mask"


def _field_channels(ts_dtype: np.dtype, field: str) -> int:
    if ts_dtype.fields is None or field not in ts_dtype.fields:
        raise KeyError(f"Field '{field}' not present in timeseries dtype.")
    field_dtype = ts_dtype.fields[field][0]
    if field_dtype.subdtype is None:
        return 1
    _, shape = field_dtype.subdtype
    if not shape:
        return 1
    return int(shape[0])


def _encode_fixed_bytes(values: Sequence[str], *, min_width: int = 1) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=f"S{min_width}")
    encoded = [v.encode("utf-8", errors="replace") for v in values]
    max_len = max(max(len(b), min_width) for b in encoded)
    return np.asarray(encoded, dtype=f"S{max_len}")


def _blosc_shuffle_v3(mode: str) -> zarr_codecs.BloscShuffle:
    mode = mode.lower()
    if mode == "bitshuffle":
        return zarr_codecs.BloscShuffle.bitshuffle
    if mode == "shuffle":
        return zarr_codecs.BloscShuffle.shuffle
    if mode in {"noshuffle", "none"}:
        return zarr_codecs.BloscShuffle.noshuffle
    raise ValueError(f"Unsupported Blosc shuffle mode: {mode}")


def _make_compressor(
    *,
    blosc_cname: str,
    blosc_clevel: int,
    blosc_shuffle: str,
) -> zarr_codecs.BloscCodec:
    return zarr_codecs.BloscCodec(
        cname=blosc_cname,
        clevel=int(blosc_clevel),
        shuffle=_blosc_shuffle_v3(blosc_shuffle),
    )


def _compression_kwargs(compressor: object) -> dict[str, object]:
    return {"compressors": [compressor]}


@dataclass(slots=True)
class SessionInfo:
    session_idx: int
    path: Path
    session_id: str
    filename: str
    user: str
    stage: str
    split: str
    side: str
    moving_hand: str
    generalization: str
    held_out_stage: bool
    held_out_user: bool
    sample_rate: float
    length: int
    emg_channels: int
    joint_channels: int
    start_idx: int = 0


def discover_sessions(paths: Sequence[Path]) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for session_idx, path in enumerate(tqdm(paths, desc="Discover sessions")):
        with h5py.File(path, "r") as f:
            if EMG2POSE_GROUP not in f:
                raise KeyError(f"{path} missing group '{EMG2POSE_GROUP}'.")
            g = f[EMG2POSE_GROUP]
            if TIMESERIES not in g:
                raise KeyError(f"{path} missing dataset '{TIMESERIES}'.")
            ts = g[TIMESERIES]

            sessions.append(
                SessionInfo(
                    session_idx=session_idx,
                    path=path,
                    session_id=str(g.attrs.get("session", path.stem)),
                    filename=str(g.attrs.get("filename", path.stem)),
                    user=str(g.attrs.get("user", "unknown")),
                    stage=str(g.attrs.get("stage", "unknown")),
                    split=str(g.attrs.get("split", "unknown")),
                    side=str(g.attrs.get("side", "unknown")).lower(),
                    moving_hand=str(g.attrs.get("moving_hand", "unknown")).lower(),
                    generalization=str(g.attrs.get("generalization", "unknown")).lower(),
                    held_out_stage=bool(g.attrs.get("held_out_stage", False)),
                    held_out_user=bool(g.attrs.get("held_out_user", False)),
                    sample_rate=float(g.attrs.get("sample_rate", np.nan)),
                    length=int(len(ts)),
                    emg_channels=_field_channels(ts.dtype, "emg"),
                    joint_channels=_field_channels(ts.dtype, "joint_angles"),
                )
            )
    return sessions


def assign_session_offsets(sessions: list[SessionInfo]) -> int:
    offset = 0
    for info in sessions:
        info.start_idx = int(offset)
        offset += int(info.length)
    return int(offset)


def _normalize_split(name: str) -> str:
    n = name.strip().lower()
    if n in {"val", "valid", "validation"}:
        return "val"
    if n in {"train", "training"}:
        return "train"
    if n in {"test", "testing"}:
        return "test"
    return n or "unknown"


def order_sessions_by_split(sessions: list[SessionInfo]) -> list[SessionInfo]:
    split_order = {"train": 0, "val": 1, "test": 2}

    def key(info: SessionInfo) -> tuple[int, str, str]:
        split = _normalize_split(info.split)
        return (split_order.get(split, 99), split, info.session_id)

    ordered = sorted(sessions, key=key)
    for idx, info in enumerate(ordered):
        info.session_idx = int(idx)
    return ordered


def _update_blocks_from_valid_chunk(
    valid_chunk: np.ndarray,
    *,
    global_start: int,
    open_block_start: int | None,
    blocks: list[tuple[int, int]],
) -> int | None:
    if valid_chunk.size == 0:
        return open_block_start

    if open_block_start is not None and not bool(valid_chunk[0]):
        blocks.append((open_block_start, global_start))
        open_block_start = None

    prev_open = open_block_start is not None
    v = valid_chunk.astype(np.int8, copy=False)
    diffs = np.diff(v)

    start_positions = (np.where(diffs == 1)[0] + 1).tolist()
    end_positions = (np.where(diffs == -1)[0] + 1).tolist()

    if bool(valid_chunk[0]) and not prev_open:
        start_positions.insert(0, 0)

    starts = deque(start_positions)

    for end_pos in end_positions:
        if open_block_start is None:
            if starts:
                open_block_start = global_start + int(starts.popleft())
            else:
                open_block_start = global_start
        blocks.append((open_block_start, global_start + int(end_pos)))
        open_block_start = None

    if bool(valid_chunk[-1]) and open_block_start is None and starts:
        open_block_start = global_start + int(starts.popleft())

    return open_block_start


def _list_hdf5_paths(input_root: Path, glob_pattern: str) -> list[Path]:
    paths = sorted(input_root.glob(glob_pattern))
    return [p for p in paths if p.is_file()]


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output root already exists: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_zarr_dataset(
    *,
    input_root: Path,
    output_root: Path,
    glob_pattern: str,
    chunk_t: int,
    shard_t: int,
    blosc_cname: str,
    blosc_clevel: int,
    blosc_shuffle: str,
    min_block_len: int,
    consolidate_metadata: bool,
    overwrite: bool,
    limit: int | None,
    dry_run: bool,
) -> None:
    paths = _list_hdf5_paths(input_root, glob_pattern)
    if not paths:
        raise SystemExit(f"No HDF5 files found under {input_root} with {glob_pattern}.")
    if limit is not None:
        paths = paths[:limit]

    print(f"Found {len(paths)} sessions.")
    sessions = discover_sessions(paths)
    sessions = order_sessions_by_split(sessions)

    emg_channels = {s.emg_channels for s in sessions}
    joint_channels = {s.joint_channels for s in sessions}
    if len(emg_channels) != 1:
        raise SystemExit(f"Inconsistent EMG channels across sessions: {emg_channels}")
    if len(joint_channels) != 1:
        raise SystemExit(
            f"Inconsistent joint angle dims across sessions: {joint_channels}"
        )
    emg_channels_val = int(next(iter(emg_channels)))
    joint_channels_val = int(next(iter(joint_channels)))

    total_length = assign_session_offsets(sessions)

    print(f"Total frames: {total_length}")
    if dry_run:
        print("Dry run complete. No data was written.")
        return

    _prepare_output_root(output_root, overwrite=overwrite)

    compressor = _make_compressor(
        blosc_cname=blosc_cname,
        blosc_clevel=int(blosc_clevel),
        blosc_shuffle=blosc_shuffle,
    )
    compression = _compression_kwargs(compressor)

    if shard_t <= 0:
        raise SystemExit("--shard-t must be positive.")
    if shard_t % chunk_t != 0:
        raise SystemExit(
            f"--shard-t ({shard_t}) must be a multiple of --chunk-t ({chunk_t})."
        )

    users = sorted({s.user for s in sessions})
    stages = sorted({s.stage for s in sessions})
    sides = sorted({s.side for s in sessions})
    raw_splits = {_normalize_split(s.split) for s in sessions}
    split_order = ["train", "val", "test"]
    splits = [s for s in split_order if s in raw_splits] + sorted(
        [s for s in raw_splits if s not in split_order]
    )

    user_to_id = {u: i for i, u in enumerate(users)}
    stage_to_id = {st: i for i, st in enumerate(stages)}
    side_to_id = {sd: i for i, sd in enumerate(sides)}
    split_to_id = {sp: i for i, sp in enumerate(splits)}

    S = len(sessions)
    stats_sum = np.zeros((S, emg_channels_val), dtype=np.float64)
    stats_sumsq = np.zeros((S, emg_channels_val), dtype=np.float64)
    stats_count = np.zeros((S,), dtype=np.int64)

    blocks_out: dict[str, list[int]] = defaultdict(list)

    root = zarr.open_group(str(output_root), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 3,
            "layout": "global_concat_v1",
            "zarr_format": 3,
            "chunk_t": int(chunk_t),
            "shard_t": int(shard_t),
            "num_sessions": int(S),
            "total_length": int(total_length),
            "emg_channels": int(emg_channels_val),
            "joint_channels": int(joint_channels_val),
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_root),
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, emg_channels_val),
        chunks=(chunk_t, emg_channels_val),
        shards=(shard_t, emg_channels_val),
        dtype="f4",
        **compression,
    )
    joint_arr = root.create_array(
        "joint_angles",
        shape=(total_length, joint_channels_val),
        chunks=(chunk_t, joint_channels_val),
        shards=(shard_t, joint_channels_val),
        dtype="f4",
        **compression,
    )
    time_arr = root.create_array(
        "time",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f8",
        **compression,
    )
    valid_arr = root.create_array(
        "valid_mask",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="bool",
        **compression,
    )
    ik_arr = root.create_array(
        IK_FAILURE_MASK,
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="bool",
        **compression,
    )

    for info in tqdm(sessions, desc="Convert sessions"):
        with h5py.File(info.path, "r") as f:
            g = f[EMG2POSE_GROUP]
            ts = g[TIMESERIES]
            T = int(len(ts))
            ik_ds = g[IK_FAILURE_MASK] if IK_FAILURE_MASK in g else None

            sum_c = np.zeros((info.emg_channels,), dtype=np.float64)
            sumsq_c = np.zeros((info.emg_channels,), dtype=np.float64)
            count_c = np.int64(0)

            blocks: list[tuple[int, int]] = []
            open_block_start: int | None = None

            for start in range(0, T, chunk_t):
                end = min(T, start + chunk_t)
                global_start = info.start_idx + start
                global_end = info.start_idx + end

                ts_chunk = ts[start:end]

                emg_chunk = np.asarray(ts_chunk["emg"], dtype=np.float32)
                joint_chunk = np.asarray(ts_chunk["joint_angles"], dtype=np.float32)
                time_chunk = np.asarray(ts_chunk["time"], dtype=np.float64)

                if ik_ds is not None:
                    ik_chunk = np.asarray(ik_ds[start:end], dtype=bool)
                    no_fail = ~ik_chunk
                else:
                    no_fail = get_ik_failures_mask(joint_chunk)
                    ik_chunk = ~no_fail

                finite_mask = np.isfinite(joint_chunk).all(axis=1)
                valid_chunk = np.asarray(no_fail & finite_mask, dtype=bool)

                emg_arr[global_start:global_end] = emg_chunk
                joint_arr[global_start:global_end] = joint_chunk
                time_arr[global_start:global_end] = time_chunk
                valid_arr[global_start:global_end] = valid_chunk
                ik_arr[global_start:global_end] = ik_chunk

                if valid_chunk.any():
                    emg_valid = emg_chunk[valid_chunk]
                    sum_c += emg_valid.sum(axis=0, dtype=np.float64)
                    sumsq_c += np.square(emg_valid, dtype=np.float64).sum(
                        axis=0, dtype=np.float64
                    )
                    count_c += np.int64(valid_chunk.sum())

                open_block_start = _update_blocks_from_valid_chunk(
                    valid_chunk,
                    global_start=global_start,
                    open_block_start=open_block_start,
                    blocks=blocks,
                )

            if open_block_start is not None:
                blocks.append((open_block_start, info.start_idx + T))

        stats_sum[info.session_idx, :] = sum_c
        stats_sumsq[info.session_idx, :] = sumsq_c
        stats_count[info.session_idx] = count_c

        for block_start, block_end in blocks:
            seg_len = block_end - block_start
            if seg_len < min_block_len:
                continue
            blocks_out["session_idx"].append(info.session_idx)
            blocks_out["start"].append(int(block_start))
            blocks_out["end"].append(int(block_end))
            blocks_out["length"].append(int(seg_len))

    sessions_sorted = sorted(sessions, key=lambda s: s.session_idx)
    session_ids = [s.session_id for s in sessions_sorted]
    filenames = [s.filename for s in sessions_sorted]
    start_idx_arr = np.asarray([s.start_idx for s in sessions_sorted], dtype=np.int64)
    length_arr = np.asarray([s.length for s in sessions_sorted], dtype=np.int64)
    end_idx_arr = start_idx_arr + length_arr

    user_id_arr = np.asarray([user_to_id[s.user] for s in sessions_sorted], dtype=np.int32)
    stage_id_arr = np.asarray(
        [stage_to_id[s.stage] for s in sessions_sorted], dtype=np.int32
    )
    side_id_arr = np.asarray([side_to_id[s.side] for s in sessions_sorted], dtype=np.int8)
    split_id_arr = np.asarray(
        [split_to_id[_normalize_split(s.split)] for s in sessions_sorted],
        dtype=np.int32,
    )

    sample_rate_arr = np.asarray([s.sample_rate for s in sessions_sorted], dtype=np.float32)
    moving_hand_arr = _encode_fixed_bytes([s.moving_hand for s in sessions_sorted])
    generalization_arr = _encode_fixed_bytes(
        [s.generalization for s in sessions_sorted]
    )
    held_out_stage_arr = np.asarray(
        [s.held_out_stage for s in sessions_sorted], dtype=bool
    )
    held_out_user_arr = np.asarray(
        [s.held_out_user for s in sessions_sorted], dtype=bool
    )

    sessions_group = root.require_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("filename", data=_encode_fixed_bytes(filenames))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("user_id", data=user_id_arr)
    sessions_group.create_array("stage_id", data=stage_id_arr)
    sessions_group.create_array("side_id", data=side_id_arr)
    sessions_group.create_array("split_id", data=split_id_arr)
    sessions_group.create_array("sample_rate", data=sample_rate_arr)
    sessions_group.create_array("moving_hand", data=moving_hand_arr)
    sessions_group.create_array("generalization", data=generalization_arr)
    sessions_group.create_array("held_out_stage", data=held_out_stage_arr)
    sessions_group.create_array("held_out_user", data=held_out_user_arr)

    users_group = root.require_group("users")
    users_group.create_array("user", data=_encode_fixed_bytes(users))
    users_group.create_array("user_id", data=np.arange(len(users), dtype=np.int32))

    stages_group = root.require_group("stages")
    stages_group.create_array("stage", data=_encode_fixed_bytes(stages))
    stages_group.create_array("stage_id", data=np.arange(len(stages), dtype=np.int32))

    sides_group = root.require_group("sides")
    sides_group.create_array("side", data=_encode_fixed_bytes(sides))
    sides_group.create_array("side_id", data=np.arange(len(sides), dtype=np.int8))

    splits_group = root.require_group("splits")
    splits_group.create_array("split", data=_encode_fixed_bytes(splits))
    splits_group.create_array("split_id", data=np.arange(len(splits), dtype=np.int32))

    stats_group = root.require_group("stats")
    stats_group.create_array("session_sum", data=stats_sum)
    stats_group.create_array("session_sumsq", data=stats_sumsq)
    stats_group.create_array("session_count", data=stats_count)

    blocks_group = root.require_group("blocks")
    blocks_group.create_array(
        "session_idx", data=np.asarray(blocks_out["session_idx"], dtype=np.int32)
    )
    blocks_group.create_array(
        "start", data=np.asarray(blocks_out["start"], dtype=np.int64)
    )
    blocks_group.create_array(
        "end", data=np.asarray(blocks_out["end"], dtype=np.int64)
    )
    blocks_group.create_array(
        "length", data=np.asarray(blocks_out["length"], dtype=np.int32)
    )

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg2pose_data"),
        help="Root directory containing HDF5 sessions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/emg2pose_v3"),
        help="Output directory for the global Zarr store.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="**/*.hdf5",
        help="Glob pattern relative to input root for discovering sessions.",
    )
    parser.add_argument(
        "--chunk-t",
        type=int,
        default=12000,
        help="Chunk size along time dimension.",
    )
    parser.add_argument(
        "--shard-t",
        type=int,
        default=120000,
        help="Shard size along time dimension (multiple of chunk-t).",
    )
    parser.add_argument(
        "--blosc-cname",
        type=str,
        default="lz4",
        help="Blosc compressor name (e.g., zstd, lz4).",
    )
    parser.add_argument(
        "--blosc-clevel",
        type=int,
        default=5,
        help="Blosc compression level.",
    )
    parser.add_argument(
        "--blosc-shuffle",
        type=str,
        default="bitshuffle",
        choices=["bitshuffle", "shuffle", "noshuffle", "none"],
        help="Blosc shuffle mode.",
    )
    parser.add_argument(
        "--min-block-len",
        type=int,
        default=1,
        help="Minimum valid block length to keep.",
    )
    parser.add_argument(
        "--no-consolidate-metadata",
        action="store_true",
        help="Disable consolidated metadata creation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output root if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N discovered sessions (for debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the conversion but do not write any data.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    build_zarr_dataset(
        input_root=args.input_root.expanduser(),
        output_root=args.output_root.expanduser(),
        glob_pattern=args.glob_pattern,
        chunk_t=int(args.chunk_t),
        shard_t=int(args.shard_t),
        blosc_cname=str(args.blosc_cname),
        blosc_clevel=int(args.blosc_clevel),
        blosc_shuffle=str(args.blosc_shuffle),
        min_block_len=int(args.min_block_len),
        consolidate_metadata=not bool(args.no_consolidate_metadata),
        overwrite=bool(args.overwrite),
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
