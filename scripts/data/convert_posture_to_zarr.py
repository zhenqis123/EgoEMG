from __future__ import annotations

"""
Convert posture dataset blocks into a single global Zarr v3 store.

Layout:
- emg            (T, 16)
- glove          (T, 18)
- finger         (T, 5)
- time           (T,)
- target_pos     (T,)   int16, per-trial constant (or -1 if missing)
- grasp          (T,)   int16, per-trial constant (or -1 if missing)
- valid_mask     (T,)
- sessions/*     per-trial index + metadata
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import shutil
import sys
from typing import Sequence

import h5py
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


@dataclass(slots=True)
class SessionInfo:
    session_idx: int
    block_dir: Path
    session_id: str
    filename: str
    participant_id: int
    day: int
    block: int
    trial_id: int
    trial_no: int
    target_position: int
    grasp: int
    length: int
    emg_len: int
    glove_len: int
    finger_len: int
    sample_rate: float | None


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


def _parse_block_dir(name: str) -> tuple[int, int, int] | None:
    match = re.match(r"participant(\d+)_day(\d+)_block(\d+)", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _parse_params(path: Path) -> dict[str, float]:
    params: dict[str, float] = {}
    if not path.is_file():
        return params
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            params[key] = float(val)
        except ValueError:
            continue
    return params


def _load_trials_csv(path: Path) -> dict[int, dict[str, int]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "row_number" in df.columns:
        key_col = "row_number"
    else:
        key_col = df.columns[0]
    out: dict[int, dict[str, int]] = {}
    for _, row in df.iterrows():
        try:
            key = int(row[key_col])
        except Exception:
            continue
        out[key] = {
            "trial_no": int(row.get("trial_no", -1)),
            "target_position": int(row.get("target_position", -1)),
            "grasp": int(row.get("grasp", -1)),
            "block": int(row.get("block", -1)),
        }
    return out


def discover_sessions(input_root: Path) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    participant_dirs = sorted(input_root.glob("participant_*"))
    for participant_dir in participant_dirs:
        for block_dir in sorted(participant_dir.iterdir()):
            if not block_dir.is_dir():
                continue
            parsed = _parse_block_dir(block_dir.name)
            if parsed is None:
                continue
            participant_id, day, block = parsed

            emg_path = block_dir / "emg_data.hdf5"
            glove_path = block_dir / "glove_data.hdf5"
            finger_path = block_dir / "finger_data.hdf5"
            if not emg_path.is_file():
                continue

            trials_csv = block_dir / "trials.csv"
            trial_meta = _load_trials_csv(trials_csv)
            params = _parse_params(block_dir / "recording_parameters.txt")
            trial_len_sec = params.get("Trial length time (sec)")

            with h5py.File(emg_path, "r") as emg_file, h5py.File(
                glove_path, "r"
            ) as glove_file, h5py.File(finger_path, "r") as finger_file:
                keys = list(emg_file.keys())
                try:
                    keys = sorted(keys, key=lambda k: int(k))
                except Exception:
                    keys = sorted(keys)

                for key in keys:
                    if key not in glove_file or key not in finger_file:
                        continue
                    emg_len = int(emg_file[key].shape[1])
                    glove_len = int(glove_file[key].shape[1])
                    finger_len = int(finger_file[key].shape[1])
                    length = int(min(emg_len, glove_len, finger_len))
                    if length <= 0:
                        continue

                    try:
                        trial_id = int(key)
                    except ValueError:
                        trial_id = -1

                    meta = trial_meta.get(trial_id, {})
                    trial_no = int(meta.get("trial_no", -1))
                    target_position = int(meta.get("target_position", -1))
                    grasp = int(meta.get("grasp", -1))

                    sample_rate = None
                    if trial_len_sec and trial_len_sec > 0:
                        sample_rate = float(length / trial_len_sec)

                    session_id = (
                        f"participant{participant_id}_day{day}_block{block}_trial{trial_id}"
                    )
                    sessions.append(
                        SessionInfo(
                            session_idx=len(sessions),
                            block_dir=block_dir,
                            session_id=session_id,
                            filename=block_dir.name,
                            participant_id=participant_id,
                            day=day,
                            block=block,
                            trial_id=trial_id,
                            trial_no=trial_no,
                            target_position=target_position,
                            grasp=grasp,
                            length=length,
                            emg_len=emg_len,
                            glove_len=glove_len,
                            finger_len=finger_len,
                            sample_rate=sample_rate,
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

    sessions = discover_sessions(input_root)
    if limit is not None:
        sessions = sessions[: int(limit)]

    if not sessions:
        raise ValueError(f"No sessions found under {input_root}")

    total_length = int(sum(s.length for s in sessions))
    print(f"Found {len(sessions)} trials, total frames={total_length}")

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
            "dataset": "posture",
            "source_root": str(input_root),
            "emg_channels": 16,
            "glove_channels": 18,
            "finger_channels": 5,
            "compressor": {
                "cname": blosc_cname,
                "clevel": int(blosc_clevel),
                "shuffle": blosc_shuffle,
            },
        }
    )

    emg_arr = root.create_array(
        "emg",
        shape=(total_length, 16),
        chunks=(chunk_t, 16),
        shards=(shard_t, 16),
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
    finger_arr = root.create_array(
        "finger",
        shape=(total_length, 5),
        chunks=(chunk_t, 5),
        shards=(shard_t, 5),
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
    target_arr = root.create_array(
        "target_position",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i2",
        fill_value=-1,
        **compression,
    )
    grasp_arr = root.create_array(
        "grasp",
        shape=(total_length,),
        chunks=(chunk_t,),
        shards=(shard_t,),
        dtype="i2",
        fill_value=-1,
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

    start_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    length_arr = np.zeros((len(sessions),), dtype=np.int64)
    end_idx_arr = np.zeros((len(sessions),), dtype=np.int64)
    participant_arr = np.zeros((len(sessions),), dtype=np.int16)
    day_arr = np.zeros((len(sessions),), dtype=np.int8)
    block_arr = np.zeros((len(sessions),), dtype=np.int8)
    trial_id_arr = np.zeros((len(sessions),), dtype=np.int16)
    trial_no_arr = np.zeros((len(sessions),), dtype=np.int16)
    target_pos_arr = np.zeros((len(sessions),), dtype=np.int16)
    grasp_meta_arr = np.zeros((len(sessions),), dtype=np.int16)
    sample_rate_arr = np.full((len(sessions),), np.nan, dtype=np.float32)
    emg_len_arr = np.zeros((len(sessions),), dtype=np.int32)
    glove_len_arr = np.zeros((len(sessions),), dtype=np.int32)
    finger_len_arr = np.zeros((len(sessions),), dtype=np.int32)

    session_ids = []
    filenames = []

    offset = 0
    current_block: Path | None = None
    emg_file = glove_file = finger_file = None

    try:
        for s in tqdm(sessions, desc="Convert trials"):
            if current_block != s.block_dir:
                if emg_file is not None:
                    emg_file.close()
                if glove_file is not None:
                    glove_file.close()
                if finger_file is not None:
                    finger_file.close()
                current_block = s.block_dir
                emg_file = h5py.File(s.block_dir / "emg_data.hdf5", "r")
                glove_file = h5py.File(s.block_dir / "glove_data.hdf5", "r")
                finger_file = h5py.File(s.block_dir / "finger_data.hdf5", "r")

            key = str(s.trial_id if s.trial_id >= 0 else s.trial_id)
            if key not in emg_file:
                key = str(s.trial_id)
            emg = np.asarray(emg_file[key], dtype=np.float32)[:, : s.length].T
            glove = np.asarray(glove_file[key], dtype=np.float32)[:, : s.length].T
            finger = np.asarray(finger_file[key], dtype=np.float32)[:, : s.length].T

            start = offset
            end = start + s.length

            emg_arr[start:end] = emg
            glove_arr[start:end] = glove
            finger_arr[start:end] = finger
            valid_arr[start:end] = True

            if s.sample_rate and s.sample_rate > 0:
                time = np.arange(s.length, dtype=np.float32) / s.sample_rate
                root.attrs["time_unit"] = "seconds"
            else:
                time = np.arange(s.length, dtype=np.float32)
                root.attrs["time_unit"] = "sample_index"
            time_arr[start:end] = time

            target_arr[start:end] = s.target_position
            grasp_arr[start:end] = s.grasp

            start_idx_arr[s.session_idx] = start
            length_arr[s.session_idx] = s.length
            end_idx_arr[s.session_idx] = end
            participant_arr[s.session_idx] = s.participant_id
            day_arr[s.session_idx] = s.day
            block_arr[s.session_idx] = s.block
            trial_id_arr[s.session_idx] = s.trial_id
            trial_no_arr[s.session_idx] = s.trial_no
            target_pos_arr[s.session_idx] = s.target_position
            grasp_meta_arr[s.session_idx] = s.grasp
            sample_rate_arr[s.session_idx] = (
                s.sample_rate if s.sample_rate is not None else np.nan
            )
            emg_len_arr[s.session_idx] = s.emg_len
            glove_len_arr[s.session_idx] = s.glove_len
            finger_len_arr[s.session_idx] = s.finger_len
            session_ids.append(s.session_id)
            filenames.append(s.filename)

            offset = end
    finally:
        if emg_file is not None:
            emg_file.close()
        if glove_file is not None:
            glove_file.close()
        if finger_file is not None:
            finger_file.close()

    sessions_group = root.create_group("sessions")
    sessions_group.create_array("session_id", data=_encode_fixed_bytes(session_ids))
    sessions_group.create_array("filename", data=_encode_fixed_bytes(filenames))
    sessions_group.create_array("start_idx", data=start_idx_arr)
    sessions_group.create_array("length", data=length_arr)
    sessions_group.create_array("end_idx", data=end_idx_arr)
    sessions_group.create_array("participant_id", data=participant_arr)
    sessions_group.create_array("day", data=day_arr)
    sessions_group.create_array("block", data=block_arr)
    sessions_group.create_array("trial_id", data=trial_id_arr)
    sessions_group.create_array("trial_no", data=trial_no_arr)
    sessions_group.create_array("target_position", data=target_pos_arr)
    sessions_group.create_array("grasp", data=grasp_meta_arr)
    sessions_group.create_array("sample_rate", data=sample_rate_arr)
    sessions_group.create_array("emg_len", data=emg_len_arr)
    sessions_group.create_array("glove_len", data=glove_len_arr)
    sessions_group.create_array("finger_len", data=finger_len_arr)

    if consolidate_metadata:
        zarr.consolidate_metadata(str(output_root), zarr_format=3)

    print(f"Wrote Zarr store to: {output_root}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/emg_corpus/posture"),
        help="Root directory containing posture dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/emg_corpus/posture/posture_v3"),
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
        help="Process only the first N trials (for debugging).",
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
