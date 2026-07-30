from __future__ import annotations

"""
Convert EMG2QWERTY HDF5 sessions into a single global Zarr v3 store.

Layout:
- emg_left         (T, 16)
- emg_right        (T, 16)
- time             (T,)
- valid_mask       (T,)  # all True
- sessions/*       per-session index + metadata
- users/*          user id mapping
- conditions/*     condition id mapping
- keystrokes/*     global event table + per-session offsets
- prompts/*        global event table + per-session offsets
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys
from typing import Any, Sequence

import h5py
import numpy as np
from tqdm import tqdm

try:
    import zarr
    from zarr import codecs as zarr_codecs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies for Zarr conversion. Please install `zarr` in "
        "your environment."
    ) from exc


EMG2QWERTY_GROUP = "emg2qwerty"
TIMESERIES = "timeseries"


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
    user: str
    condition: str
    sample_rate: float
    length: int
    start_idx: int = 0


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


def discover_sessions(paths: Sequence[Path]) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for session_idx, path in enumerate(tqdm(paths, desc="Discover sessions")):
        with h5py.File(path, "r") as f:
            if EMG2QWERTY_GROUP not in f:
                raise KeyError(f"{path} missing group '{EMG2QWERTY_GROUP}'.")
            g = f[EMG2QWERTY_GROUP]
            if TIMESERIES not in g:
                raise KeyError(f"{path} missing dataset '{TIMESERIES}'.")
            ts = g[TIMESERIES]

            session_id = str(g.attrs.get("session_name", path.stem))
            user = str(g.attrs.get("user", "unknown"))
            condition = str(g.attrs.get("condition", "unknown"))
            sample_rate = float(g.attrs.get("daq_sample_rate", np.nan))
            length = int(len(ts))

        sessions.append(
            SessionInfo(
                session_idx=session_idx,
                path=path,
                session_id=session_id,
                user=user,
                condition=condition,
                sample_rate=sample_rate,
                length=length,
            )
        )
    return sessions


def order_sessions(sessions: list[SessionInfo]) -> list[SessionInfo]:
    ordered = sorted(sessions, key=lambda s: s.session_id)
    for idx, info in enumerate(ordered):
        info.session_idx = int(idx)
    return ordered


def assign_session_offsets(sessions: list[SessionInfo]) -> int:
    offset = 0
    for info in sessions:
        info.start_idx = int(offset)
        offset += int(info.length)
    return int(offset)


def _parse_json_attr(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, list):
        return value
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
    sessions = order_sessions(sessions)

    total_length = assign_session_offsets(sessions)

    print(f"Total frames: {total_length}")
    if dry_run:
        print("Dry run complete. No data was written.")
        return

    if shard_t <= 0:
        raise SystemExit("--shard-t must be positive.")
    if shard_t % chunk_t != 0:
        raise SystemExit(
            f"--shard-t ({shard_t}) must be a multiple of --chunk-t ({chunk_t})."
        )

    _prepare_output_root(output_root, overwrite=overwrite)

    compressor = _make_compressor(
        blosc_cname=blosc_cname,
        blosc_clevel=int(blosc_clevel),
        blosc_shuffle=blosc_shuffle,
    )
    compression = _compression_kwargs(compressor)

    users = sorted({s.user for s in sessions})
    conditions = sorted({s.condition for s in sessions})
    user_to_id = {u: i for i, u in enumerate(users)}
    condition_to_id = {c: i for i, c in enumerate(conditions)}

    root = zarr.open_group(str(output_root), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 1,
            "layout": "emg2qwerty_global_v1",
            "zarr_format": 3,
            "chunk_t": int(chunk_t),
            "shard_t": int(shard_t),
            "num_sessions": int(len(sessions)),
            "total_length": int(total_length),
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_root),
        }
    )

    emg_left = root.create_array(
        "emg_left",
        shape=(total_length, 16),
        chunks=(chunk_t, 16),
        shards=(shard_t, 16),
        dtype="f4",
        **compression,
    )
    emg_right = root.create_array(
        "emg_right",
        shape=(total_length, 16),
        chunks=(chunk_t, 16),
        shards=(shard_t, 16),
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

    keystroke_start: list[float] = []
    keystroke_end: list[float] = []
    keystroke_ascii: list[int] = []
    keystroke_key: list[str] = []
    keystroke_offset = np.zeros((len(sessions),), dtype=np.int64)
    keystroke_length = np.zeros((len(sessions),), dtype=np.int64)

    prompt_start: list[float] = []
    prompt_end: list[float] = []
    prompt_num: list[int] = []
    prompt_text: list[str] = []
    prompt_offset = np.zeros((len(sessions),), dtype=np.int64)
    prompt_length = np.zeros((len(sessions),), dtype=np.int64)

    for info in tqdm(sessions, desc="Convert sessions"):
        with h5py.File(info.path, "r") as f:
            g = f[EMG2QWERTY_GROUP]
            ts = g[TIMESERIES]
            T = int(len(ts))

            if ts.dtype.fields is None:
                raise ValueError(f"Timeseries dtype invalid in {info.path}")
            if "emg_left" not in ts.dtype.fields or "emg_right" not in ts.dtype.fields:
                raise KeyError(f"Missing emg_left/emg_right in {info.path}")

            for start in range(0, T, chunk_t):
                end = min(T, start + chunk_t)
                global_start = info.start_idx + start
                global_end = info.start_idx + end

                ts_chunk = ts[start:end]

                emg_left[global_start:global_end] = np.asarray(
                    ts_chunk["emg_left"], dtype=np.float32
                )
                emg_right[global_start:global_end] = np.asarray(
                    ts_chunk["emg_right"], dtype=np.float32
                )
                time_arr[global_start:global_end] = np.asarray(
                    ts_chunk["time"], dtype=np.float64
                )
                valid_arr[global_start:global_end] = True

            # Keystrokes
            ks_list = _parse_json_attr(g.attrs.get("keystrokes"))
            keystroke_offset[info.session_idx] = len(keystroke_start)
            keystroke_length[info.session_idx] = len(ks_list)
            for k in ks_list:
                keystroke_start.append(_safe_float(k.get("start", 0.0)))
                keystroke_end.append(_safe_float(k.get("end", 0.0)))
                keystroke_ascii.append(_safe_int(k.get("ascii", -1)))
                keystroke_key.append(str(k.get("key", "")))

            # Prompts
            prompts_list = _parse_json_attr(g.attrs.get("prompts"))
            prompt_offset[info.session_idx] = len(prompt_start)
            prompt_length[info.session_idx] = len(prompts_list)
            for p in prompts_list:
                prompt_start.append(_safe_float(p.get("start", 0.0)))
                prompt_end.append(_safe_float(p.get("end", 0.0)))
                payload = p.get("payload") if isinstance(p, dict) else None
                if isinstance(payload, dict):
                    prompt_num.append(_safe_int(payload.get("prompt_num", -1)))
                    prompt_text.append(str(payload.get("text", "")))
                else:
                    prompt_num.append(-1)
                    prompt_text.append("")

    sessions_sorted = sorted(sessions, key=lambda s: s.session_idx)
    session_ids = [s.session_id for s in sessions_sorted]
    start_idx_arr = np.asarray([s.start_idx for s in sessions_sorted], dtype=np.int64)
    length_arr = np.asarray([s.length for s in sessions_sorted], dtype=np.int64)
    end_idx_arr = start_idx_arr + length_arr
    user_id_arr = np.asarray([user_to_id[s.user] for s in sessions_sorted], dtype=np.int32)
    condition_id_arr = np.asarray(
        [condition_to_id[s.condition] for s in sessions_sorted], dtype=np.int32
    )
    sample_rate_arr = np.asarray([s.sample_rate for s in sessions_sorted], dtype=np.float32)

    sessions_group = root.require_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("user_id", data=user_id_arr)
    sessions_group.create_array("condition_id", data=condition_id_arr)
    sessions_group.create_array("sample_rate", data=sample_rate_arr)

    users_group = root.require_group("users")
    users_group.create_array("user", data=_encode_fixed_bytes(users))
    users_group.create_array("user_id", data=np.arange(len(users), dtype=np.int32))

    conditions_group = root.require_group("conditions")
    conditions_group.create_array(
        "condition", data=_encode_fixed_bytes(conditions)
    )
    conditions_group.create_array(
        "condition_id", data=np.arange(len(conditions), dtype=np.int32)
    )

    keystrokes_group = root.require_group("keystrokes")
    keystrokes_group.create_array(
        "start", data=np.asarray(keystroke_start, dtype=np.float64)
    )
    keystrokes_group.create_array(
        "end", data=np.asarray(keystroke_end, dtype=np.float64)
    )
    keystrokes_group.create_array(
        "ascii", data=np.asarray(keystroke_ascii, dtype=np.int32)
    )
    keystrokes_group.create_array(
        "key", data=_encode_fixed_bytes(keystroke_key)
    )
    keystrokes_group.create_array("session_offset", data=keystroke_offset)
    keystrokes_group.create_array("session_length", data=keystroke_length)

    prompts_group = root.require_group("prompts")
    prompts_group.create_array(
        "start", data=np.asarray(prompt_start, dtype=np.float64)
    )
    prompts_group.create_array(
        "end", data=np.asarray(prompt_end, dtype=np.float64)
    )
    prompts_group.create_array(
        "prompt_num", data=np.asarray(prompt_num, dtype=np.int32)
    )
    prompts_group.create_array(
        "text", data=_encode_fixed_bytes(prompt_text)
    )
    prompts_group.create_array("session_offset", data=prompt_offset)
    prompts_group.create_array("session_length", data=prompt_length)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/emg2qwerty_data"),
        help="Root directory containing EMG2QWERTY HDF5 sessions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/emg2qwerty_v3"),
        help="Output directory for the global Zarr store.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="*.hdf5",
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
        consolidate_metadata=not bool(args.no_consolidate_metadata),
        overwrite=bool(args.overwrite),
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
