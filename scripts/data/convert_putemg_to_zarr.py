from __future__ import annotations

"""
Convert putEMG HDF5 sessions into a single global Zarr v3 store.

Layout:
- emg               (T, 24)
- time              (T,)
- force             (T, 10)          [NaN for gesture-only sessions]
- force_mvc         (T,)             [NaN for gesture-only sessions]
- traj              (T, K)           K=max available traj channels
- gesture_gt        (T,)             [NaN for non-gesture sessions]
- gesture_gt_nf     (T,)             [NaN for non-gesture sessions]
- video_stamp       (T,)             [NaN for non-gesture sessions]
- valid_mask        (T,)
- sessions/*        per-session index + metadata
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
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


try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None


@dataclass(slots=True)
class SessionInfo:
    session_idx: int
    path: Path
    session_id: str
    filename: str
    subject_id: int
    protocol: str
    file_type: str
    length: int
    source_length: int
    emg_channels: int
    force_channels: int
    traj_channels: int
    has_force: bool
    has_gesture: bool


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


def _parse_filename(stem: str) -> tuple[int, str]:
    parts = stem.split("-")
    subject_id = -1
    protocol = "unknown"
    if len(parts) >= 2:
        try:
            subject_id = int(parts[1])
        except ValueError:
            subject_id = -1
    if len(parts) >= 3:
        protocol = parts[2]
    return subject_id, protocol


def _resampled_length(n_samples: int, *, up: int, down: int) -> int:
    if up == down:
        return int(n_samples)
    return int(math.ceil(n_samples * up / down))


def discover_sessions(
    input_root: Path, *, source_fs: int, target_fs: int
) -> list[SessionInfo]:
    paths = sorted(input_root.glob("*.h5"))
    sessions: list[SessionInfo] = []
    up = target_fs
    down = source_fs
    for idx, path in enumerate(tqdm(paths, desc="Discover sessions")):
        with h5py.File(path, "r") as f:
            emg = f["emg"]
            source_length = int(emg.shape[1])
            length = _resampled_length(source_length, up=up, down=down)
            emg_channels = int(emg.shape[0])
            file_type = str(f.attrs.get("file_type", "unknown"))
            force_channels = int(f.attrs.get("force_channels", 0))

            has_force = "force" in f
            has_gesture = "gesture_optional" in f

            traj_channels = 0
            if "traj" in f:
                traj_keys = [k for k in f["traj"].keys() if k.startswith("TRAJ_")]
                traj_channels = len(traj_keys)

        subject_id, protocol = _parse_filename(path.stem)
        sessions.append(
            SessionInfo(
                session_idx=idx,
                path=path,
                session_id=path.stem,
                filename=path.name,
                subject_id=subject_id,
                protocol=protocol,
                file_type=file_type,
                length=length,
                source_length=source_length,
                emg_channels=emg_channels,
                force_channels=force_channels,
                traj_channels=traj_channels,
                has_force=has_force,
                has_gesture=has_gesture,
            )
        )
    return sessions


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass --overwrite to replace."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_zarr_dataset(
    *,
    input_root: Path,
    output_root: Path,
    source_fs: int,
    target_fs: int,
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
    if shard_t % chunk_t != 0:
        raise ValueError("shard_t must be a multiple of chunk_t")

    sessions = discover_sessions(input_root, source_fs=source_fs, target_fs=target_fs)
    if limit is not None:
        sessions = sessions[: int(limit)]

    if not sessions:
        raise ValueError(f"No .h5 files found under {input_root}")

    emg_channels = sessions[0].emg_channels
    if any(s.emg_channels != emg_channels for s in sessions):
        raise ValueError("Inconsistent emg_channels across sessions")

    max_force_channels = max(s.force_channels for s in sessions)
    max_traj_channels = max(s.traj_channels for s in sessions)
    total_length = int(sum(s.length for s in sessions))

    print(
        f"Found {len(sessions)} sessions, total frames={total_length}, "
        f"emg_channels={emg_channels}, force_channels={max_force_channels}, "
        f"traj_channels={max_traj_channels}, resample {source_fs}->{target_fs} Hz"
    )

    if dry_run:
        print("Dry run: no data written.")
        return

    _prepare_output_root(output_root, overwrite=overwrite)
    root = zarr.open_group(str(output_root), mode="w", zarr_format=3)

    compressor = _make_compressor(
        blosc_cname=blosc_cname,
        blosc_clevel=blosc_clevel,
        blosc_shuffle=blosc_shuffle,
    )
    compression = _compression_kwargs(compressor)

    root.attrs.update(
        {
            "dataset": "putEMG",
            "source_root": str(input_root),
            "emg_channels": int(emg_channels),
            "force_channels": int(max_force_channels),
            "traj_channels": int(max_traj_channels),
            "sample_rate": int(target_fs),
            "source_sample_rate": int(source_fs),
            "resample_ratio": f"{target_fs}/{source_fs}",
            "time_unit": "seconds",
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, emg_channels),
        chunks=(chunk_t, emg_channels),
        shards=(shard_t, emg_channels),
        dtype="f4",
        **compression,
    )
    time_arr = root.create_array(
        "time",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f4",
        **compression,
    )
    valid_arr = root.create_array(
        "valid_mask",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="bool",
        fill_value=True,
        **compression,
    )

    force_arr = root.create_array(
        "force",
        shape=(total_length, max_force_channels),
        chunks=(chunk_t, max_force_channels),
        shards=(shard_t, max_force_channels),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )
    force_mvc_arr = root.create_array(
        "force_mvc",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )
    traj_arr = root.create_array(
        "traj",
        shape=(total_length, max_traj_channels),
        chunks=(chunk_t, max_traj_channels),
        shards=(shard_t, max_traj_channels),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )
    gesture_gt_arr = root.create_array(
        "gesture_gt",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )
    gesture_gt_nf_arr = root.create_array(
        "gesture_gt_no_filter",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )
    video_stamp_arr = root.create_array(
        "video_stamp",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="f4",
        fill_value=np.nan,
        **compression,
    )

    start_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    length_arr = np.zeros((len(sessions),), dtype=np.int64)
    end_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    subject_arr = np.zeros((len(sessions),), dtype=np.int32)
    force_channels_arr = np.zeros((len(sessions),), dtype=np.int16)
    traj_channels_arr = np.zeros((len(sessions),), dtype=np.int16)
    has_force_arr = np.zeros((len(sessions),), dtype=bool)
    has_gesture_arr = np.zeros((len(sessions),), dtype=bool)

    session_ids = []
    filenames = []
    protocols = []
    file_types = []

    if target_fs <= 0 or source_fs <= 0:
        raise ValueError("sample rates must be positive")
    up = target_fs
    down = source_fs
    if up != down and resample_poly is None:
        raise RuntimeError(
            "scipy is required for resampling. Install scipy or set target_fs=source_fs."
        )

    def _resample(data: np.ndarray) -> np.ndarray:
        if up == down:
            return data
        return resample_poly(data, up, down, axis=0)

    offset = 0
    for s in tqdm(sessions, desc="Convert sessions"):
        start = offset
        end = start + s.length

        with h5py.File(s.path, "r") as f:
            emg = np.asarray(f["emg"], dtype=np.float32).T
            time = np.asarray(f["timestamp"], dtype=np.float32)

            if emg.shape[0] != s.source_length or time.shape[0] != s.source_length:
                raise ValueError(f"Length mismatch in {s.filename}")

            emg_ds = _resample(emg)
            time_ds = time[0] + np.arange(emg_ds.shape[0], dtype=np.float32) / float(
                target_fs
            )

            if emg_ds.shape[0] != s.length:
                raise ValueError(f"Resample length mismatch in {s.filename}")

            emg_arr[start:end] = np.ascontiguousarray(emg_ds)
            time_arr[start:end] = time_ds
            valid_arr[start:end] = True

            if "force" in f:
                force = np.asarray(f["force/force_1_to_10"], dtype=np.float32).T
                force_mvc = np.asarray(f["force/force_mvc"], dtype=np.float32)
                force_ds = _resample(force)
                force_mvc_ds = _resample(force_mvc[:, None]).squeeze(-1)
                force_arr[start:end, : force_ds.shape[1]] = force_ds
                force_mvc_arr[start:end] = force_mvc_ds

            if "traj" in f:
                traj_keys = sorted(
                    [k for k in f["traj"].keys() if k.startswith("TRAJ_")],
                    key=lambda k: int(k.split("_")[1]),
                )
                if traj_keys:
                    traj = np.stack(
                        [np.asarray(f["traj"][k], dtype=np.float32) for k in traj_keys],
                        axis=1,
                    )
                    traj_ds = _resample(traj)
                    traj_arr[start:end, : traj_ds.shape[1]] = traj_ds

            if "gesture_optional" in f:
                gt = np.asarray(f["gesture_optional/TRAJ_GT"], dtype=np.float32)
                gt_nf = np.asarray(
                    f["gesture_optional/TRAJ_GT_NO_FILTER"], dtype=np.float32
                )
                vs = np.asarray(f["gesture_optional/VIDEO_STAMP"], dtype=np.float32)
                gesture_gt_arr[start:end] = _resample(gt[:, None]).squeeze(-1)
                gesture_gt_nf_arr[start:end] = _resample(gt_nf[:, None]).squeeze(-1)
                video_stamp_arr[start:end] = _resample(vs[:, None]).squeeze(-1)

        start_idx_arr[s.session_idx] = start
        length_arr[s.session_idx] = s.length
        end_idx_arr[s.session_idx] = end
        subject_arr[s.session_idx] = s.subject_id
        force_channels_arr[s.session_idx] = s.force_channels
        traj_channels_arr[s.session_idx] = s.traj_channels
        has_force_arr[s.session_idx] = s.has_force
        has_gesture_arr[s.session_idx] = s.has_gesture
        session_ids.append(s.session_id)
        filenames.append(s.filename)
        protocols.append(s.protocol)
        file_types.append(s.file_type)

        offset = end

    sessions_group = root.create_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("filename", data=_encode_fixed_bytes(filenames))
    sessions_group.create_array("protocol", data=_encode_fixed_bytes(protocols))
    sessions_group.create_array("file_type", data=_encode_fixed_bytes(file_types))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("subject_id", data=subject_arr)
    sessions_group.create_array("force_channels", data=force_channels_arr)
    sessions_group.create_array("traj_channels", data=traj_channels_arr)
    sessions_group.create_array("has_force", data=has_force_arr)
    sessions_group.create_array("has_gesture", data=has_gesture_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/putEMG/Data-HDF5"),
        help="Root directory containing putEMG .h5 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/putEMG/putemg_v3"),
        help="Output directory for the global Zarr store.",
    )
    parser.add_argument(
        "--source-fs",
        type=int,
        default=5120,
        help="Source sampling rate (Hz).",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=2000,
        help="Target sampling rate (Hz).",
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
        source_fs=int(args.source_fs),
        target_fs=int(args.target_fs),
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
