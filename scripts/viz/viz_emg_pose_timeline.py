#!/usr/bin/env python3
"""Visualize EMG signals, joint angles, and MANO pose from EgoEmgMemmapDataset.

Produces a multi-panel matplotlib figure showing aligned time-series data
for a given episode and hand. Useful for verifying data pipeline correctness.

Usage:
    # Default: episode 3, right hand, 2000-frame window at offset 100000
    python scripts/viz/viz_emg_pose_timeline.py --episode 3 --hand right --offset 100000

    # Longer window, save to custom path
    python scripts/viz/viz_emg_pose_timeline.py --episode 10 --hand left --window 5000 \
        --offset 50000 --out-path /tmp/timeline.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
MEMMAP_DIR = PROJ_ROOT / "data" / "EgoEMG_memmap"
MANO_NPY_DIR = PROJ_ROOT / "data" / "EgoEMG" / "mano" / "chunk-000"
DEFAULT_OUT_DIR = PROJ_ROOT / "data" / "EgoEMG" / "mano_viz" / "timelines"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
MANO_JOINT_NAMES = [
    "wrist", "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
]


def make_dataset(hand: str, window_length: int):
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    return EgoEmgMemmapDataset(
        memmap_dir=MEMMAP_DIR,
        window_length=window_length,
        stride=window_length,
        modalities=["emg", "joint_angles", "mocap_hands", "mano"],
        target_hand=hand,
        emg_field_preference="filtered",
        emg_layout="target_hand",
        norm_mode=None,
        dataset_name="egoemg",
        mano_npy_dir=MANO_NPY_DIR,
    )


def find_window_at_offset(ds, ep_idx: int, offset: int) -> int | None:
    block_mask = ds._block_episode_idx == ep_idx
    block_ids = np.where(block_mask)[0]
    for bi in block_ids:
        w_start = int(ds._block_cumsum[bi])
        w_end = int(ds._block_cumsum[bi + 1])
        block_start = int(ds._block_start[bi])
        for w in range(w_start, w_end):
            rel = w - w_start
            start = block_start + rel * ds.stride
            end = start + ds.window_length
            if start <= offset < end:
                return w
    return None


def plot_timeline(sample: dict, ep_id: str, hand: str, out_path: Path):
    has_emg = "emg" in sample
    has_ja = "joint_angles" in sample
    has_mano = "mano_pose" in sample

    n_panels = sum([has_emg, has_ja, has_mano])
    if n_panels == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    panel = 0

    if has_emg:
        ax = axes[panel]
        emg = sample["emg"]  # (C, T)
        T = emg.shape[1]
        t = np.arange(T)
        for ch in range(emg.shape[0]):
            ax.plot(t, emg[ch] + ch * 0.5, linewidth=0.3, alpha=0.8)
        ax.set_ylabel("EMG channels")
        ax.set_title(f"{ep_id} / {hand} hand — Filtered EMG ({emg.shape[0]} ch)")
        ax.set_xlim(0, T)
        panel += 1

    if has_ja:
        ax = axes[panel]
        ja = sample["joint_angles"]  # (C, T)
        T = ja.shape[1]
        t = np.arange(T)
        n_angles = ja.shape[0]
        cmap = plt.cm.tab20(np.linspace(0, 1, n_angles))
        for ch in range(n_angles):
            label = f"j{ch}" if ch < 20 else ("wrist_pitch" if ch == 20 else "wrist_yaw")
            ax.plot(t, ja[ch], linewidth=0.6, alpha=0.8, color=cmap[ch], label=label)
        ax.set_ylabel("Joint angles (rad)")
        ax.set_title(f"Joint angles ({n_angles}-dim)")
        ax.legend(fontsize=5, ncol=6, loc="upper right")
        ax.set_xlim(0, T)
        panel += 1

    if has_mano:
        ax = axes[panel]
        pose = sample["mano_pose"]  # (T, 48)
        T = pose.shape[0]
        t = np.arange(T)
        for j in range(16):
            mag = np.linalg.norm(pose[:, j*3:(j+1)*3], axis=1)
            label = MANO_JOINT_NAMES[j] if j < len(MANO_JOINT_NAMES) else f"j{j}"
            ax.plot(t, mag, linewidth=0.5, alpha=0.7, label=label)
        ax.set_ylabel("MANO joint rotation magnitude (rad)")
        ax.set_title(f"MANO pose (48-dim axis-angle, beta shape: {sample.get('mano_beta', np.zeros(0)).shape})")
        ax.legend(fontsize=5, ncol=4, loc="upper right")
        ax.set_xlim(0, T)
        panel += 1

    axes[-1].set_xlabel("Frame index (within window)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--offset", type=int, default=100000,
                        help="Frame offset within episode")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ds = make_dataset(args.hand, args.window)
    print(f"Dataset: {len(ds)} windows, hand={args.hand}")

    win_idx = find_window_at_offset(ds, args.episode, args.offset)
    if win_idx is None:
        block_mask = ds._block_episode_idx == args.episode
        block_ids = np.where(block_mask)[0]
        if len(block_ids) > 0:
            win_idx = int(ds._block_cumsum[block_ids[0]])
    if win_idx is None:
        print(f"Episode {args.episode} not found in dataset.")
        return

    sample = ds[win_idx]
    ep_id = sample.get("episode_id", f"episode_{args.episode:06d}")
    start = sample.get("window_start_idx", 0)
    print(f"Sample: {ep_id}, frames [{start}:{start + args.window}]")
    print(f"  Keys: {sorted(sample.keys())}")
    if "mano_pose" in sample:
        print(f"  mano_pose: {sample['mano_pose'].shape}")
    if "mano_beta" in sample:
        print(f"  mano_beta: {sample['mano_beta'].shape}")

    if args.out_path:
        out_path = args.out_path
    else:
        out_path = args.out_dir / f"{ep_id}_{args.hand}_f{start}.png"

    plot_timeline(sample, ep_id, args.hand, out_path)


if __name__ == "__main__":
    main()
