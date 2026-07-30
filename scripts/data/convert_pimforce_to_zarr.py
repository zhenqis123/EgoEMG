from __future__ import annotations

"""
Convert PiMforce processed_raw .npy files into a single global Zarr v3 store.

Layout:
- emg          (T, 8)
- joint_angles (T, 20)
- force        (T, 9)
- valid_mask   (T,)  # all True
- sessions/*   per-file metadata + index
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import zarr
    from zarr import codecs as zarr_codecs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies for Zarr conversion. Please install `zarr` in "
        "your environment."
    ) from exc


EMG_CHANNELS = 8
JOINT_CHANNELS = 20
FORCE_CHANNELS = 9
TOTAL_CHANNELS = EMG_CHANNELS + JOINT_CHANNELS + FORCE_CHANNELS


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
    user_id: int
    session_id: int
    file_id: int
    session_name: str
    filename: str
    original_length: int
    duration: float
    length: int
    start_idx: int = 0


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output root already exists: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def load_sessions(input_root: Path, metadata_path: Path) -> list[SessionInfo]:
    df = pd.read_csv(metadata_path)
    required = {
        "user_id",
        "session_id",
        "file_id",
        "session_name",
        "filename",
        "output_path",
        "original_length",
        "duration",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    sessions: list[SessionInfo] = []
    df = df.sort_values(["session_id", "file_id"]).reset_index(drop=True)
    for idx, row in df.iterrows():
        rel = Path(str(row["output_path"]))
        path = rel if rel.is_absolute() else input_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing .npy file: {path}")

        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Unexpected shape {arr.shape} in {path}")
        if arr.shape[0] == TOTAL_CHANNELS:
            length = int(arr.shape[1])
        elif arr.shape[1] == TOTAL_CHANNELS:
            length = int(arr.shape[0])
        else:
            raise ValueError(f"Unexpected channels in {path}: {arr.shape}")

        sessions.append(
            SessionInfo(
                session_idx=int(idx),
                path=path,
                user_id=int(row["user_id"]),
                session_id=int(row["session_id"]),
                file_id=int(row["file_id"]),
                session_name=str(row["session_name"]),
                filename=str(row["filename"]),
                original_length=int(row["original_length"]),
                duration=float(row["duration"]),
                length=length,
            )
        )
    return sessions


def assign_session_offsets(sessions: list[SessionInfo]) -> int:
    offset = 0
    for info in sessions:
        info.start_idx = int(offset)
        offset += int(info.length)
    return int(offset)


def build_zarr_dataset(
    *,
    input_root: Path,
    output_root: Path,
    metadata_path: Path,
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
    sessions = load_sessions(input_root, metadata_path)
    if limit is not None:
        sessions = sessions[:limit]

    total_length = assign_session_offsets(sessions)
    print(f"Found {len(sessions)} sessions. Total frames: {total_length}")
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

    root = zarr.open_group(str(output_root), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_version": 1,
            "layout": "pimforce_global_v1",
            "zarr_format": 3,
            "chunk_t": int(chunk_t),
            "shard_t": int(shard_t),
            "num_sessions": int(len(sessions)),
            "total_length": int(total_length),
            "emg_channels": EMG_CHANNELS,
            "joint_channels": JOINT_CHANNELS,
            "force_channels": FORCE_CHANNELS,
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_root),
            "metadata_csv": str(metadata_path),
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, EMG_CHANNELS),
        chunks=(chunk_t, EMG_CHANNELS),
        shards=(shard_t, EMG_CHANNELS),
        dtype="f4",
        **compression,
    )
    joint_arr = root.create_array(
        "joint_angles",
        shape=(total_length, JOINT_CHANNELS),
        chunks=(chunk_t, JOINT_CHANNELS),
        shards=(shard_t, JOINT_CHANNELS),
        dtype="f4",
        **compression,
    )
    force_arr = root.create_array(
        "force",
        shape=(total_length, FORCE_CHANNELS),
        chunks=(chunk_t, FORCE_CHANNELS),
        shards=(shard_t, FORCE_CHANNELS),
        dtype="f4",
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

    for info in tqdm(sessions, desc="Convert sessions"):
        arr = np.load(info.path, mmap_mode="r")
        if arr.shape[0] == TOTAL_CHANNELS:
            channel_first = True
            length = int(arr.shape[1])
        elif arr.shape[1] == TOTAL_CHANNELS:
            channel_first = False
            length = int(arr.shape[0])
        else:
            raise ValueError(f"Unexpected shape {arr.shape} in {info.path}")

        for start in range(0, length, chunk_t):
            end = min(length, start + chunk_t)
            global_start = info.start_idx + start
            global_end = info.start_idx + end

            if channel_first:
                chunk = arr[:, start:end].T
            else:
                chunk = arr[start:end, :]
            chunk = np.asarray(chunk, dtype=np.float32)

            emg_arr[global_start:global_end] = chunk[:, :EMG_CHANNELS]
            joint_arr[global_start:global_end] = chunk[
                :, EMG_CHANNELS : EMG_CHANNELS + JOINT_CHANNELS
            ]
            force_arr[global_start:global_end] = chunk[
                :, EMG_CHANNELS + JOINT_CHANNELS : TOTAL_CHANNELS
            ]
            valid_arr[global_start:global_end] = True

    sessions_sorted = sorted(sessions, key=lambda s: s.session_idx)
    session_ids = np.asarray([s.session_id for s in sessions_sorted], dtype=np.int32)
    user_ids = np.asarray([s.user_id for s in sessions_sorted], dtype=np.int32)
    file_ids = np.asarray([s.file_id for s in sessions_sorted], dtype=np.int32)
    session_names = _encode_fixed_bytes([s.session_name for s in sessions_sorted])
    filenames = _encode_fixed_bytes([s.filename for s in sessions_sorted])
    start_idx_arr = np.asarray([s.start_idx for s in sessions_sorted], dtype=np.int64)
    length_arr = np.asarray([s.length for s in sessions_sorted], dtype=np.int64)
    end_idx_arr = start_idx_arr + length_arr
    original_length_arr = np.asarray(
        [s.original_length for s in sessions_sorted], dtype=np.int64
    )
    duration_arr = np.asarray([s.duration for s in sessions_sorted], dtype=np.float32)

    sessions_group = root.require_group("sessions")
    sessions_group.create_array("session_id", data=session_ids)
    sessions_group.create_array("user_id", data=user_ids)
    sessions_group.create_array("file_id", data=file_ids)
    sessions_group.create_array("session_name", data=session_names)
    sessions_group.create_array("filename", data=filenames)
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("original_length", data=original_length_arr)
    sessions_group.create_array("duration", data=duration_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/processed_raw"),
        help="Root directory containing processed_raw .npy files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/processed_raw/pimforce_metadata.csv"),
        help="Path to pimforce_metadata.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/PiMforce/pimforce_v3"),
        help="Output directory for the global Zarr store.",
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
        help="Process only the first N sessions (for debugging).",
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
        metadata_path=args.metadata.expanduser(),
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
