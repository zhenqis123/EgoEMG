from __future__ import annotations

"""
Convert MyoKi .mat files into a single global Zarr v3 store.

Layout:
- emg               (T, 12)
- gyro              (T, 27)
- acc               (T, 27)
- glove             (T, 18)
- glove_calibrated  (T, 18)
- time              (T,)
- task              (T,) int16
- grasp             (T,) int16
- repetition        (T,) int16
- valid_mask        (T,)
- sessions/*        per-session index + metadata
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
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
    file_index: int
    session_id: str
    filename: str
    participant_id: int
    local_start: int
    local_end: int
    length: int
    task: int
    grasp: int
    repetition: int


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


def _discover_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("P*.mat"))


def _parse_participant_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("P"):
        try:
            return int(stem[1:])
        except ValueError:
            return -1
    return -1


def _find_segments(task: np.ndarray, grasp: np.ndarray, repetition: np.ndarray) -> list[tuple[int, int]]:
    if task.size == 0:
        return []
    change = np.ones(task.shape[0], dtype=bool)
    change[1:] = (
        (task[1:] != task[:-1])
        | (grasp[1:] != grasp[:-1])
        | (repetition[1:] != repetition[:-1])
    )
    starts = np.nonzero(change)[0]
    ends = np.concatenate([starts[1:], np.array([task.shape[0]], dtype=np.int64)])
    return list(zip(starts.tolist(), ends.tolist()))


def discover_sessions(files: list[Path]) -> tuple[list[SessionInfo], list[int], int, int]:
    sessions: list[SessionInfo] = []
    file_lengths: list[int] = []
    sample_rate = None
    for file_index, path in enumerate(tqdm(files, desc="Discover files")):
        mat = scipy.io.loadmat(path)
        if sample_rate is None:
            sample_rate = int(np.asarray(mat["frequency"]).squeeze())
        task = np.asarray(mat["task"]).squeeze().astype(np.int16, copy=False)
        grasp = np.asarray(mat["grasp"]).squeeze().astype(np.int16, copy=False)
        repetition = np.asarray(mat["repetition"]).squeeze().astype(np.int16, copy=False)

        length = int(task.shape[0])
        file_lengths.append(length)
        participant_id = _parse_participant_id(path)
        segments = _find_segments(task, grasp, repetition)
        for start, end in segments:
            tval = int(task[start])
            gval = int(grasp[start])
            rval = int(repetition[start])
            session_id = f"{path.stem}_task{tval}_grasp{gval}_rep{rval}_start{start}"
            sessions.append(
                SessionInfo(
                    session_idx=len(sessions),
                    file_index=file_index,
                    session_id=session_id,
                    filename=path.name,
                    participant_id=participant_id,
                    local_start=start,
                    local_end=end,
                    length=end - start,
                    task=tval,
                    grasp=gval,
                    repetition=rval,
                )
            )
    if sample_rate is None:
        sample_rate = -1
    return sessions, file_lengths, int(sample_rate), int(sum(file_lengths))


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

    files = _discover_files(input_root)
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise ValueError(f"No .mat files found under {input_root}")

    sessions, file_lengths, sample_rate, total_length = discover_sessions(files)
    print(
        f"Found {len(files)} files, {len(sessions)} sessions, total frames={total_length}"
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
            "dataset": "MyoKi",
            "source_root": str(input_root),
            "sample_rate": int(sample_rate),
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, 12),
        chunks=(chunk_t, 12),
        shards=(shard_t, 12),
        dtype="f4",
        **compression,
    )
    gyro_arr = root.create_array(
        "gyro",
        shape=(total_length, 27),
        chunks=(chunk_t, 27),
        shards=(shard_t, 27),
        dtype="f4",
        **compression,
    )
    acc_arr = root.create_array(
        "acc",
        shape=(total_length, 27),
        chunks=(chunk_t, 27),
        shards=(shard_t, 27),
        dtype="f4",
        **compression,
    )
    glove_arr = root.create_array(
        "glove",
        shape=(total_length, 18),
        chunks=(chunk_t, 18),
        shards=(shard_t, 18),
        dtype="f4",
        **compression,
    )
    glove_cal_arr = root.create_array(
        "glove_calibrated",
        shape=(total_length, 18),
        chunks=(chunk_t, 18),
        shards=(shard_t, 18),
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
    task_arr = root.create_array(
        "task",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i2",
        **compression,
    )
    grasp_arr = root.create_array(
        "grasp",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i2",
        **compression,
    )
    rep_arr = root.create_array(
        "repetition",
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

    file_offsets = np.cumsum([0] + file_lengths[:-1]).astype(np.int64)

    for file_index, path in enumerate(tqdm(files, desc="Write files")):
        mat = scipy.io.loadmat(path)
        start = int(file_offsets[file_index])
        end = start + int(file_lengths[file_index])

        emg = np.asarray(mat["emg"], dtype=np.float32)
        gyro = np.asarray(mat["gyro"], dtype=np.float32)
        acc = np.asarray(mat["acc"], dtype=np.float32)
        glove = np.asarray(mat["glove"], dtype=np.float32)
        glove_cal = np.asarray(mat["glove_calibrated"], dtype=np.float32)
        time = np.asarray(mat["timestamp"], dtype=np.float32).squeeze()
        task = np.asarray(mat["task"], dtype=np.int16).squeeze()
        grasp = np.asarray(mat["grasp"], dtype=np.int16).squeeze()
        repetition = np.asarray(mat["repetition"], dtype=np.int16).squeeze()

        emg_arr[start:end] = emg
        gyro_arr[start:end] = gyro
        acc_arr[start:end] = acc
        glove_arr[start:end] = glove
        glove_cal_arr[start:end] = glove_cal
        time_arr[start:end] = time
        task_arr[start:end] = task
        grasp_arr[start:end] = grasp
        rep_arr[start:end] = repetition
        valid_arr[start:end] = True

    start_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    length_arr = np.zeros((len(sessions),), dtype=np.int64)
    end_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    participant_arr = np.zeros((len(sessions),), dtype=np.int16)
    task_meta_arr = np.zeros((len(sessions),), dtype=np.int16)
    grasp_meta_arr = np.zeros((len(sessions),), dtype=np.int16)
    rep_meta_arr = np.zeros((len(sessions),), dtype=np.int16)
    file_index_arr = np.zeros((len(sessions),), dtype=np.int16)
    filenames = []
    session_ids = []

    for s in sessions:
        file_offset = int(file_offsets[s.file_index])
        start = file_offset + s.local_start
        end = file_offset + s.local_end
        start_idx_arr[s.session_idx] = start
        length_arr[s.session_idx] = s.length
        end_idx_arr[s.session_idx] = end
        participant_arr[s.session_idx] = s.participant_id
        task_meta_arr[s.session_idx] = s.task
        grasp_meta_arr[s.session_idx] = s.grasp
        rep_meta_arr[s.session_idx] = s.repetition
        file_index_arr[s.session_idx] = s.file_index
        filenames.append(s.filename)
        session_ids.append(s.session_id)

    sessions_group = root.create_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("filename", data=_encode_fixed_bytes(filenames))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("participant_id", data=participant_arr)
    sessions_group.create_array("task", data=task_meta_arr)
    sessions_group.create_array("grasp", data=grasp_meta_arr)
    sessions_group.create_array("repetition", data=rep_meta_arr)
    sessions_group.create_array("file_index", data=file_index_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/MyoKi"),
        help="Root directory containing MyoKi .mat files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/MyoKi/myoki_v3"),
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
