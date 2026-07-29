#!/usr/bin/env python3
"""Regenerate ShowEE filtered EMG from raw memmaps with the canonical filter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.realtime.filter import filter_emg_fft


def open_field(root: Path, manifest: dict, name: str, mode: str) -> np.memmap:
    info = manifest["fields"][name]
    return np.memmap(
        root / info["filename"],
        dtype=info["dtype"],
        shape=tuple(info["shape"]),
        mode=mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.memmap_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = np.load(root / "metadata.npz", allow_pickle=False)
    starts = np.asarray(metadata["episode_start_idx"], dtype=np.int64)
    ends = np.asarray(metadata["episode_end_idx"], dtype=np.int64)

    for hand in ("left", "right"):
        raw = open_field(root, manifest, f"emg_{hand}_raw", "r")
        filtered = open_field(root, manifest, f"emg_{hand}_filtered_paper", "r+")
        for episode, (start, end) in enumerate(zip(starts, ends)):
            filtered[start:end] = filter_emg_fft(np.asarray(raw[start:end]))
            if (episode + 1) % 25 == 0 or episode + 1 == len(starts):
                filtered.flush()
                print(f"{hand}: {episode + 1}/{len(starts)} episodes", flush=True)
        filtered.flush()

    manifest["emg_filter_paper"] = {
        "source": "scripts/realtime/filter.py:filter_emg_fft",
        "sample_rate_hz": 2000.0,
        "highpass_transition_hz": [15.0, 20.0],
        "lowpass_transition_hz": [850.0, 900.0],
        "notch_hz": [50.0, 100.0],
        "notch_stop_half_width_hz": 1.5,
        "notch_transition_half_width_hz": 1.5,
        "scope": "per episode and per hand",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
