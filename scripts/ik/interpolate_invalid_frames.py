#!/usr/bin/env python3
"""Fill invalid-frame joint angles by linear interpolation between valid frames.

Usage:
    python scripts/ik/interpolate_invalid_frames.py --memmap-root data/sess_20260530_140912/memmap --hand right
    python scripts/ik/interpolate_invalid_frames.py --memmap-root data/sess_20260530_143229/memmap --hand right
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    args = parser.parse_args()

    root = args.memmap_root
    hand = args.hand

    valid = np.load(root / "valid.npy").astype(bool)
    angles_path = root / f"generated_joint_angles_{hand}.dat"
    angles = np.memmap(angles_path, dtype=np.float32, mode="r+",
                       shape=(len(valid), 20))

    # Find contiguous invalid segments.
    invalid = ~valid
    n_invalid = invalid.sum()
    if n_invalid == 0:
        print("No invalid frames, nothing to do.")
        return

    # Detect gaps: [start, end) of invalid segments.
    boundaries = np.diff(np.concatenate([[0], invalid.astype(np.int8), [0]]))
    starts = np.where(boundaries == 1)[0]
    ends = np.where(boundaries == -1)[0]

    n_filled = 0
    n_edge = 0
    for a, b in zip(starts, ends):
        # Find nearest valid frames before and after.
        before = a - 1
        if before < 0 or not valid[before]:
            # Leading invalid segment: fill with first valid frame's values.
            after = b
            while after < len(valid) and not valid[after]:
                after += 1
            if after < len(valid):
                angles[a:b] = angles[after]
                n_edge += (b - a)
            continue

        after = b
        if after >= len(valid) or not valid[after]:
            # Trailing invalid segment: fill with last valid frame's values.
            angles[a:b] = angles[before]
            n_edge += (b - a)
            continue

        # Linear interpolation.
        w0 = angles[before]  # (20,)
        w1 = angles[after]
        gap_len = b - a
        for i in range(gap_len):
            t = (i + 1) / (gap_len + 1)
            angles[a + i] = w0 + t * (w1 - w0)
        n_filled += gap_len

    angles.flush()
    print(f"Hand: {hand}")
    print(f"  Total frames: {len(valid)}")
    print(f"  Invalid: {n_invalid} ({n_invalid / len(valid) * 100:.1f}%)")
    print(f"  Filled by interpolation: {n_filled}")
    print(f"  Filled by nearest (edge): {n_edge}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
