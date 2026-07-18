#!/usr/bin/env python3
"""Visualize a single-wrist EMG segment from the EMG2Pose memmap dataset.

Extracts a fixed-length window of EMG from one session, plots all 8 channels
for the selected wrist side.

Usage:
    python scripts/viz/visualize_emg_waveform.py \
        --memmap_dir data/emg_corpus/emg2pose_v3_memmap \
        --session 42 \
        --start_ms 0 \
        --duration_ms 500 \
        --side left

    # Pick a random valid session
    python scripts/viz/visualize_emg_waveform.py \
        --memmap_dir data/emg_corpus/emg2pose_v3_memmap \
        --random \
        --duration_ms 500
"""

import os

# Must happen BEFORE any matplotlib import
os.environ.pop("DISPLAY", None)
os.environ["MPLBACKEND"] = "Agg"

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap_dir", default="data/emg_corpus/emg2pose_v3_memmap")
    parser.add_argument("--session", type=int, default=None,
                        help="Session index to visualize. Use --random to pick randomly.")
    parser.add_argument("--random", action="store_true",
                        help="Pick a random valid session instead of specifying --session.")
    parser.add_argument("--start_ms", type=float, default=0,
                        help="Starting offset within the session (ms).")
    parser.add_argument("--duration_ms", type=float, default=500,
                        help="Duration of the segment to visualize (ms).")
    parser.add_argument("--side", default="left", choices=["left", "right"],
                        help="Which wrist side to visualize (cols 0-7 or 8-15).")
    parser.add_argument("--output", default=None,
                        help="Save figure to this path. If not set, show interactively.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for --random mode.")
    args = parser.parse_args()

    memmap_dir = args.memmap_dir
    manifest_path = f"{memmap_dir}/manifest.json"
    metadata_path = f"{memmap_dir}/metadata.npz"

    with open(manifest_path) as f:
        manifest = json.load(f)

    metadata = np.load(metadata_path, allow_pickle=True)

    # Load EMG memmap
    emg_info = manifest["fields"]["emg"]
    emg = np.memmap(
        f"{memmap_dir}/{emg_info['filename']}",
        dtype=np.dtype(emg_info["dtype"]),
        mode="r",
        shape=tuple(emg_info["shape"]),
    )
    print(f"EMG memmap: shape={emg.shape}, dtype={emg.dtype}")

    # Session info
    n_sessions = len(metadata["session_session_id"])
    sides = [s.decode() for s in metadata["sides_side"]]
    split_names = [s.decode() for s in metadata["splits_split"]]
    session_start = metadata["session_start_idx"]
    session_end = metadata["session_end_idx"]
    session_side_id = metadata["session_side_id"]
    session_split_id = metadata["session_split_id"]

    # Pick session
    if args.random:
        if args.seed is not None:
            np.random.seed(args.seed)
        valid = session_end - session_start > 0
        valid_idx = np.where(valid)[0]
        si = int(np.random.choice(valid_idx))
    else:
        si = args.session if args.session is not None else 0

    s = int(session_start[si])
    e = int(session_end[si])
    side_id = int(session_side_id[si])
    side_name = sides[side_id]
    session_id = metadata["session_session_id"][si].decode()
    split_id = int(session_split_id[si])
    split_name = split_names[split_id]
    session_len_samples = e - s
    session_duration_ms = session_len_samples / 2000.0 * 1000

    print(f"\nSession {si}: id={session_id}")
    print(f"  side={side_name} (id={side_id}), split={split_name}")
    print(f"  memmap rows: [{s}, {e}), length={session_len_samples} ({session_duration_ms:.0f} ms)")

    # Determine which columns for the requested side
    if args.side == "left":
        col_start, col_end = 0, 8
        col_label = "Left wrist (ch 0-7)"
    else:
        col_start, col_end = 8, 16
        col_label = "Right wrist (ch 8-15)"

    # Extract segment
    n_samples = int(args.duration_ms * 2000 / 1000)  # 2000 Hz sample rate
    offset = s + int(args.start_ms * 2000 / 1000)
    end = min(offset + n_samples, e)

    if offset >= e:
        print(f"\nError: start_ms={args.start_ms} exceeds session duration ({session_duration_ms:.0f} ms)")
        return

    segment = emg[offset:end, col_start:col_end].copy()
    actual_samples = segment.shape[0]
    duration_actual = actual_samples / 2000.0 * 1000  # ms

    print(f"  extracted {actual_samples} samples ({duration_actual:.1f} ms)")
    print(f"  data range: [{segment.min():.4f}, {segment.max():.4f}]")
    print(f"  data mean: {segment.mean():.4f}, std: {segment.std():.4f}")

    # Plot — raw waveforms only, no labels/borders
    cmap = plt.get_cmap("cividis")
    colors = [cmap(i / 7) for i in range(8)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    time_axis = np.arange(actual_samples) / 2000.0  # seconds

    # Per-channel normalization so every channel has visible content
    scaled = segment.copy()
    for ch in range(8):
        s = scaled[:, ch]
        mu, sigma = s.mean(), s.std()
        if sigma > 0:
            scaled[:, ch] = (s - mu) / sigma
        else:
            scaled[:, ch] = 0
    scaled *= 0.2  # amplitude scaling after z-score
    spacing = 1.0  # spacing between channel bands

    for ch in range(8):
        offset = (7 - ch + 0.5) * spacing
        ax.plot(time_axis, scaled[:, ch] + offset, color=colors[ch], linewidth=0.5)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
