from __future__ import annotations

"""
Convert Ninapro_relabeled HDF5 sessions into Zarr v3 stores.

One Zarr store is generated per DB (DB1..DB8), each with a single
global concatenated layout.
"""

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

try:
    import zarr
    from zarr import codecs as zarr_codecs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies for Zarr conversion. Please install `zarr` in "
        "your environment."
    ) from exc


EMG2POSE_GROUP = "emg2pose"


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
    user: int
    exercise: int
    dataset: str
    emg_channels: int
    joint_channels: int | None
    length: int
    start_idx: int = 0


def _list_hdf5_paths(input_root: Path) -> list[Path]:
    paths = sorted(input_root.glob("*.hdf5"))
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
            if EMG2POSE_GROUP not in f:
                raise KeyError(f"{path} missing group '{EMG2POSE_GROUP}'.")
            g = f[EMG2POSE_GROUP]
            if "emg" not in g or "gesture_id" not in g:
                raise KeyError(f"{path} missing emg/gesture_id datasets.")
            emg = g["emg"]
            joint = g["joint_angles"] if "joint_angles" in g else None

            session_id = str(g.attrs.get("session", path.stem))
            user = int(g.attrs.get("user", -1))
            exercise = int(g.attrs.get("exercise", -1))
            dataset = str(g.attrs.get("dataset", "unknown"))
            length = int(emg.shape[0])

            joint_channels = int(joint.shape[1]) if joint is not None else None

        sessions.append(
            SessionInfo(
                session_idx=session_idx,
                path=path,
                session_id=session_id,
                user=user,
                exercise=exercise,
                dataset=dataset,
                emg_channels=int(emg.shape[1]),
                joint_channels=joint_channels,
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


def _read_gesture_labels(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        g = f[EMG2POSE_GROUP]
        if "gesture_labels" not in g:
            return []
        labels = np.asarray(g["gesture_labels"]).tolist()
        return [str(x) for x in labels]


def build_db(
    *,
    input_root: Path,
    output_root: Path,
    chunk_t: int,
    shard_t: int,
    blosc_cname: str,
    blosc_clevel: int,
    blosc_shuffle: str,
    consolidate_metadata: bool,
    overwrite: bool,
    limit: int | None,
) -> None:
    paths = _list_hdf5_paths(input_root)
    if not paths:
        raise SystemExit(f"No HDF5 files found under {input_root}.")

    sessions = discover_sessions(paths)
    sessions = order_sessions(sessions)
    if limit is not None:
        sessions = sessions[:limit]

    emg_channels = {s.emg_channels for s in sessions}
    if len(emg_channels) != 1:
        raise SystemExit(f"Inconsistent EMG channels: {emg_channels}")
    emg_channels_val = int(next(iter(emg_channels)))

    joint_channels = {s.joint_channels for s in sessions}
    has_joint = None not in joint_channels
    joint_channels_val = None
    if has_joint:
        if len(joint_channels) != 1:
            raise SystemExit(f"Inconsistent joint channels: {joint_channels}")
        joint_channels_val = int(next(iter(joint_channels)))

    total_length = assign_session_offsets(sessions)

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
            "layout": "ninapro_global_v1",
            "zarr_format": 3,
            "chunk_t": int(chunk_t),
            "shard_t": int(shard_t),
            "num_sessions": int(len(sessions)),
            "total_length": int(total_length),
            "emg_channels": int(emg_channels_val),
            "joint_channels": int(joint_channels_val) if joint_channels_val else 0,
            "has_joint_angles": bool(has_joint),
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
    gesture_arr = root.create_array(
        "gesture_id",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i4",
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

    joint_arr = None
    if has_joint:
        joint_arr = root.create_array(
            "joint_angles",
            shape=(total_length, int(joint_channels_val)),
            chunks=(chunk_t, int(joint_channels_val)),
            shards=(shard_t, int(joint_channels_val)),
            dtype="f4",
            **compression,
        )

    gesture_labels = _read_gesture_labels(sessions[0].path)
    if gesture_labels:
        root.create_array("gesture_labels", data=_encode_fixed_bytes(gesture_labels))

    for info in tqdm(sessions, desc=f"Convert {input_root.name}"):
        with h5py.File(info.path, "r") as f:
            g = f[EMG2POSE_GROUP]
            emg = g["emg"]
            gesture_id = g["gesture_id"]
            joint = g["joint_angles"] if "joint_angles" in g else None
            T = int(emg.shape[0])
            if T != int(gesture_id.shape[0]):
                raise ValueError(f"Length mismatch in {info.path}")

            for start in range(0, T, chunk_t):
                end = min(T, start + chunk_t)
                global_start = info.start_idx + start
                global_end = info.start_idx + end

                emg_arr[global_start:global_end] = np.asarray(
                    emg[start:end], dtype=np.float32
                )
                gesture_arr[global_start:global_end] = np.asarray(
                    gesture_id[start:end], dtype=np.int32
                )
                valid_arr[global_start:global_end] = True

                if has_joint:
                    if joint is None:
                        raise ValueError(f"Missing joint_angles in {info.path}")
                    if joint_arr is None:
                        raise RuntimeError("joint_arr not initialized")
                    joint_arr[global_start:global_end] = np.asarray(
                        joint[start:end], dtype=np.float32
                    )

    sessions_sorted = sorted(sessions, key=lambda s: s.session_idx)
    session_ids = _encode_fixed_bytes([s.session_id for s in sessions_sorted])
    user_ids = np.asarray([s.user for s in sessions_sorted], dtype=np.int32)
    exercise_ids = np.asarray([s.exercise for s in sessions_sorted], dtype=np.int16)
    dataset_names = _encode_fixed_bytes([s.dataset for s in sessions_sorted])
    start_idx_arr = np.asarray([s.start_idx for s in sessions_sorted], dtype=np.int64)
    length_arr = np.asarray([s.length for s in sessions_sorted], dtype=np.int64)
    end_idx_arr = start_idx_arr + length_arr

    sessions_group = root.require_group("sessions")
    sessions_group.create_array("session_id", data=session_ids)
    sessions_group.create_array("user", data=user_ids)
    sessions_group.create_array("exercise", data=exercise_ids)
    sessions_group.create_array("dataset", data=dataset_names)
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro_relabeled"),
        help="Root directory containing DB1..DB8 folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/Ninapro_relabeled_zarr"),
        help="Output directory for Zarr stores (one per DB).",
    )
    parser.add_argument(
        "--dbs",
        type=str,
        nargs="*",
        default=None,
        help="Specific DB folders to convert (e.g., DB1 DB2).",
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
        help="Overwrite existing output root if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sessions per DB (for debugging).",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_root = args.input_root.expanduser()
    output_root = args.output_root.expanduser()

    db_dirs = [
        d for d in sorted(input_root.iterdir()) if d.is_dir() and d.name.startswith("DB")
    ]
    if args.dbs:
        allowed = set(args.dbs)
        db_dirs = [d for d in db_dirs if d.name in allowed]

    if not db_dirs:
        raise SystemExit("No DB folders found to convert.")

    for db_dir in db_dirs:
        out_dir = output_root / db_dir.name
        build_db(
            input_root=db_dir,
            output_root=out_dir,
            chunk_t=int(args.chunk_t),
            shard_t=int(args.shard_t),
            blosc_cname=str(args.blosc_cname),
            blosc_clevel=int(args.blosc_clevel),
            blosc_shuffle=str(args.blosc_shuffle),
            consolidate_metadata=not bool(args.no_consolidate_metadata),
            overwrite=bool(args.overwrite),
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
