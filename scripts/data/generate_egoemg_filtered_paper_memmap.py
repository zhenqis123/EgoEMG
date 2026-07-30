#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.data.filter_emg_into_new_columns import filter_emg_fft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EgoEMG memmap columns with the legacy FFT-domain paper "
            "filter from scripts/data/filter_emg_into_new_columns.py."
        )
    )
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("data/EgoEMG_memmap"),
        help="EgoEMG memmap directory containing manifest.json and metadata.npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing emg_*_filtered_paper.dat files.",
    )
    return parser.parse_args()


def _open_field(memmap_dir: Path, manifest: dict, key: str, mode: str) -> np.memmap:
    spec = manifest["fields"][key]
    return np.memmap(
        memmap_dir / spec["filename"],
        dtype=spec.get("dtype", "float32"),
        mode=mode,
        shape=tuple(spec["shape"]),
    )


def _backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def main() -> None:
    args = parse_args()
    memmap_dir = args.memmap_dir.resolve()
    manifest_path = memmap_dir / "manifest.json"
    metadata_path = memmap_dir / "metadata.npz"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    metadata = np.load(metadata_path, allow_pickle=False)
    starts = metadata["episode_start_idx"].astype(np.int64)
    ends = metadata["episode_end_idx"].astype(np.int64)
    episode_ids = [
        eid.decode("utf-8") if isinstance(eid, bytes) else str(eid)
        for eid in metadata["episode_id"]
    ]

    for hand in ("left", "right"):
        raw_key = f"emg_{hand}_raw"
        out_key = f"emg_{hand}_filtered_paper"
        # Auto-detect available hands: skip if the raw field doesn't exist
        # (e.g. incre dataset is right-hand only).
        if raw_key not in manifest["fields"]:
            print(f"{raw_key}: not in manifest, skipping {hand} hand")
            continue
        raw_spec = manifest["fields"][raw_key]
        raw_spec = manifest["fields"][raw_key]
        out_filename = f"{out_key}.dat"
        out_path = memmap_dir / out_filename
        if out_path.exists() and not args.overwrite:
            print(f"{out_key}: exists, reusing {out_path}")
            mode = "r+"
        else:
            print(f"{out_key}: writing {out_path}")
            mode = "w+"

        raw = _open_field(memmap_dir, manifest, raw_key, "r")
        out = np.memmap(
            out_path,
            dtype="float32",
            mode=mode,
            shape=tuple(raw_spec["shape"]),
        )

        for ep_id, start, end in zip(episode_ids, starts, ends):
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i:
                continue
            raw_episode = np.asarray(raw[start_i:end_i], dtype=np.float32)
            out[start_i:end_i] = filter_emg_fft(raw_episode)
            print(f"  {hand}/{ep_id}: {end_i - start_i:,} samples")

        out.flush()
        del out
        del raw
        manifest["fields"][out_key] = {
            "filename": out_filename,
            "dtype": "float32",
            "shape": raw_spec["shape"],
        }

    manifest.setdefault("emg_filter_paper", {})
    manifest["emg_filter_paper"] = {
        "source": "scripts/data/filter_emg_into_new_columns.py",
        "field_suffix": "filtered_paper",
        "pipeline": [
            "subtract per-channel mean",
            "FFT-domain soft high-pass at 20 Hz",
            "FFT-domain soft low-pass at 850 Hz",
            "FFT-domain notches at 50 Hz and 100 Hz",
            "no normalization",
        ],
    }
    _backup_once(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"updated {manifest_path}")


if __name__ == "__main__":
    main()
