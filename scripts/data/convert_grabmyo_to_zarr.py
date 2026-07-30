from __future__ import annotations

"""
Convert GrabMyo (Output BM/Session*_converted) .mat files into a global Zarr v3 store.

Layout:
- emg            (T, 28)  [forearm 16 + wrist 12]
- time           (T,)     sample index (seconds if sample_rate known)
- gesture_id     (T,)     0-based gesture index
- valid_mask     (T,)
- sessions/*     per-trial index + metadata
- gesture_labels (17,)
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import shutil
import sys
from typing import Sequence

import numpy as np
import scipy.io
from tqdm import tqdm

try:
    import zarr
    from zarr import codecs as zarr_codecs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies for Zarr conversion. Please install `zarr` in "
        "your environment."
    ) from exc


@dataclass(slots=True)
class SessionInfo:
    session_idx: int
    path: Path
    session_id: str
    filename: str
    participant_id: int
    session_number: int
    repetition: int
    gesture_id: int
    length: int


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


def _parse_motion_sequence(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text()
    names = re.findall(r"'([^']+)'", text)
    return names


def _parse_filename(stem: str) -> tuple[int, int]:
    match = re.search(r"session(\\d+)_participant(\\d+)", stem)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def discover_files(input_root: Path) -> list[Path]:
    files = sorted(input_root.glob("Output BM/Session*_converted/*.mat"))
    def key(p: Path):
        session_num, participant_id = _parse_filename(p.stem)
        return (session_num, participant_id, p.name)
    return sorted(files, key=key)


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

    motion_labels = _parse_motion_sequence(input_root / "MotionSequence.txt")
    if motion_labels and len(motion_labels) != 17:
        print(f"Warning: expected 17 gesture labels, found {len(motion_labels)}")

    files = discover_files(input_root)
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise ValueError(f"No .mat files found under {input_root}")

    sample = scipy.io.loadmat(files[0])
    forearm = sample["DATA_FOREARM"]
    wrist = sample["DATA_WRIST"]
    n_reps, n_gestures = forearm.shape
    sample_len, forearm_channels = forearm[0, 0].shape
    sample_len_wrist, wrist_channels = wrist[0, 0].shape
    if sample_len != sample_len_wrist:
        raise ValueError("Forearm and wrist sample lengths differ in sample file.")

    total_sessions = len(files) * n_reps * n_gestures
    total_length = total_sessions * sample_len

    print(
        f"Found {len(files)} files, {total_sessions} trials, total frames={total_length}, "
        f"channels={forearm_channels + wrist_channels}"
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
            "dataset": "grabmyo",
            "source_root": str(input_root),
            "forearm_channels": int(forearm_channels),
            "wrist_channels": int(wrist_channels),
            "sample_rate": 1.0,
            "time_unit": "sample_index",
            "gesture_id_base": 0,
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, forearm_channels + wrist_channels),
        chunks=(chunk_t, forearm_channels + wrist_channels),
        shards=(shard_t, forearm_channels + wrist_channels),
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
    gesture_arr = root.create_array(
        "gesture_id",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i2",
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

    if motion_labels:
        root.create_array("gesture_labels", data=_encode_fixed_bytes(motion_labels))

    start_idx_arr = np.zeros((total_sessions,), dtype=np.int64)
    length_arr = np.zeros((total_sessions,), dtype=np.int64)
    end_idx_arr = np.zeros((total_sessions,), dtype=np.int64)
    participant_arr = np.zeros((total_sessions,), dtype=np.int16)
    session_num_arr = np.zeros((total_sessions,), dtype=np.int16)
    repetition_arr = np.zeros((total_sessions,), dtype=np.int8)
    gesture_id_arr = np.zeros((total_sessions,), dtype=np.int16)
    session_ids = []
    filenames = []

    offset = 0
    sidx = 0
    for path in tqdm(files, desc="Convert files"):
        mat = scipy.io.loadmat(path)
        forearm = mat["DATA_FOREARM"]
        wrist = mat["DATA_WRIST"]
        session_num, participant_id = _parse_filename(path.stem)

        for rep in range(n_reps):
            for gest in range(n_gestures):
                fa = forearm[rep, gest]
                wr = wrist[rep, gest]
                if fa.shape[0] != sample_len or wr.shape[0] != sample_len:
                    raise ValueError(
                        f"Unexpected length in {path.name} rep={rep} gest={gest}"
                    )
                emg = np.concatenate([fa, wr], axis=1).astype(np.float32, copy=False)

                start = offset
                end = start + sample_len
                emg_arr[start:end] = emg
                time_arr[start:end] = np.arange(sample_len, dtype=np.float32)
                gesture_arr[start:end] = int(gest)
                valid_arr[start:end] = True

                session_id = (
                    f"session{session_num}_participant{participant_id}_"
                    f"rep{rep}_gesture{gest}"
                )
                session_ids.append(session_id)
                filenames.append(path.name)
                start_idx_arr[sidx] = start
                length_arr[sidx] = sample_len
                end_idx_arr[sidx] = end
                participant_arr[sidx] = participant_id
                session_num_arr[sidx] = session_num
                repetition_arr[sidx] = rep
                gesture_id_arr[sidx] = gest
                sidx += 1
                offset = end

    sessions_group = root.create_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("filename", data=_encode_fixed_bytes(filenames))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("participant_id", data=participant_arr)
    sessions_group.create_array("session_number", data=session_num_arr)
    sessions_group.create_array("repetition", data=repetition_arr)
    sessions_group.create_array("gesture_id", data=gesture_id_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/grabmyo/1.1.0"),
        help="Root directory containing grabmyo data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/grabmyo/grabmyo_v3"),
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
        help="Process only the first N .mat files (for debugging).",
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
