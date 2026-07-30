#!/usr/bin/env python3
"""Fix non-anchor frame interpolation in fitted Manus data and re-align with EMG.

The bug: fit_manus_to_umetrack.py interpolated non-anchor frames using the
NEXT anchor's values before that anchor was optimized (default zeros/NaN).
This script re-interpolates between adjacent anchors (both now optimized)
and re-runs EMG alignment.

Usage:
  python scripts/data/fix_manus_interpolation.py --session data_20260525_180032
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_IK_DIR = _SCRIPT_DIR.parent / "ik"
if str(_IK_DIR) not in sys.path:
    sys.path.insert(0, str(_IK_DIR))

from fit_manus_to_umetrack import align_with_emg


def fix_interpolation(fit_dir: Path, step: int = 10):
    """Re-interpolate all per-Manus-frame arrays between adjacent anchors."""
    for fname in ["joint_angles_right", "wrist_rotation_right", "scales_right", "wrist_translation_right"]:
        path = fit_dir / f"{fname}.npy"
        arr = np.load(path)
        n_frames = len(arr)

        anchor_indices = list(range(0, n_frames, step))
        if anchor_indices[-1] != n_frames - 1:
            anchor_indices.append(n_frames - 1)

        for idx in range(len(anchor_indices) - 1):
            a, b = anchor_indices[idx], anchor_indices[idx + 1]
            for frame in range(a + 1, b):
                frac = (frame - a) / (b - a)
                arr[frame] = (1 - frac) * arr[a] + frac * arr[b]

        np.save(path, arr)
        is_nan = np.isnan(arr).sum() if arr.ndim == 1 else np.isnan(arr).any(axis=-1).sum()
        print(f"  {fname}: {n_frames} frames, {is_nan} NaN remaining")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session name, e.g. data_20260525_180032")
    parser.add_argument("--data-root", default="data/data")
    parser.add_argument("--fit-root", default="data/manus_fit")
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    session_name = Path(args.session).name
    session_data_dir = Path(args.data_root) / session_name
    session_fit_dir = Path(args.fit_root) / session_name

    print(f"Fixing interpolation for {session_name}")
    fix_interpolation(session_fit_dir, step=args.step)

    print("\nRe-aligning with EMG...")
    aligned_angles, aligned_wrist, aligned_scales, aligned_trans = align_with_emg(
        session_data_dir, session_fit_dir
    )

    np.save(session_fit_dir / "joint_angles_right_emg_aligned.npy", aligned_angles)
    aligned_angles.tofile(session_fit_dir / "joint_angles_right_emg_aligned.dat")
    np.save(session_fit_dir / "wrist_rotation_right_emg_aligned.npy", aligned_wrist)
    np.save(session_fit_dir / "scales_right_emg_aligned.npy", aligned_scales)
    np.save(session_fit_dir / "wrist_translation_right_emg_aligned.npy", aligned_trans)

    print(f"  Angles:   {aligned_angles.shape}, [{aligned_angles.min():.4f}, {aligned_angles.max():.4f}]")
    print(f"  Wrist aa: {aligned_wrist.shape}, [{aligned_wrist.min():.4f}, {aligned_wrist.max():.4f}]")
    print(f"  Scales:   {aligned_scales.shape}, [{aligned_scales.min():.4f}, {aligned_scales.max():.4f}]")
    print(f"  Trans:    {aligned_trans.shape}, [{aligned_trans.min():.4f}, {aligned_trans.max():.4f}]")
    print("Done.")


if __name__ == "__main__":
    main()
