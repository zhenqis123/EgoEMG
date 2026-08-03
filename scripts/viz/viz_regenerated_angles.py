#!/usr/bin/env python3
"""Quick visualization of regenerated joint angles to verify correctness."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egoemg.UmeTrack.lib.common.hand_skinning import skin_landmarks
from egoemg.kinematics import broadcast_hand_model_to, load_default_hand_model

ANGLE_NAMES = [
    "Thumb\nCMC_FE", "Thumb\nCMC_AA", "Thumb\nMCP_FE", "Thumb\nIP_FE",
    "Index\nMCP_AA", "Index\nMCP_FE", "Index\nPIP_FE", "Index\nDIP_FE",
    "Middle\nMCP_AA", "Middle\nMCP_FE", "Middle\nPIP_FE", "Middle\nDIP_FE",
    "Ring\nMCP_AA", "Ring\nMCP_FE", "Ring\nPIP_FE", "Ring\nDIP_FE",
    "Pinky\nMCP_AA", "Pinky\nMCP_FE", "Pinky\nPIP_FE", "Pinky\nDIP_FE",
]

MEMORY = ""


def main():
    root = Path("data/EgoEMG_memmap")
    manifest = json.loads((root / "manifest.json").read_text())
    fields = manifest["fields"]
    meta = np.load(root / "metadata.npz", allow_pickle=True)

    def _open(name):
        info = fields[name]
        return np.memmap(root / info["filename"], dtype=np.dtype(info["dtype"]),
                         mode="r", shape=tuple(info["shape"]))

    ja_l = _open("generated_joint_angles_left")
    ja_r = _open("generated_joint_angles_right")
    vm = _open("generated_label_valid")
    gc = _open("label_gesture_class")

    rng = np.random.RandomState(42)
    episode_indices = rng.choice(len(meta["episode_id"]), size=5, replace=False)

    out_dir = Path("/tmp/angle_viz_v2")
    out_dir.mkdir(exist_ok=True)

    # Load hand model once
    hand_model = load_default_hand_model()

    for ep_idx in episode_indices:
        s = int(meta["episode_start_idx"][ep_idx])
        e = int(meta["episode_end_idx"][ep_idx])
        ep_id = meta["episode_id"][ep_idx].decode() if isinstance(meta["episode_id"][ep_idx], bytes) else str(meta["episode_id"][ep_idx])
        n_frames = e - s

        # Sample evenly spaced frames
        sample_idx = np.linspace(s, e - 1, min(500, n_frames), dtype=int)

        left = np.array(ja_l[sample_idx])
        right = np.array(ja_r[sample_idx])
        valid = np.array(vm[sample_idx])
        gestures = np.array(gc[sample_idx])

        # ── Plot 1: Angle trajectories ──
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for i in range(20):
            axes[0].plot(left[:, i], alpha=0.7, linewidth=0.5, label=ANGLE_NAMES[i].replace("\n", " "))
            axes[1].plot(right[:, i], alpha=0.7, linewidth=0.5)
        axes[0].set_title(f"Left Hand — {ep_id} ({n_frames:,} frames)")
        axes[1].set_title(f"Right Hand — {ep_id} ({n_frames:,} frames)")
        axes[0].set_ylabel("Angle (rad)")
        axes[1].set_ylabel("Angle (rad)")
        axes[1].set_xlabel("Sample index")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ep_id}_trajectories.png", dpi=150)
        plt.close(fig)

        # ── Plot 2: Per-DOF histogram (all valid frames in episode) ──
        # Use fewer frames for histogram
        hist_idx = np.linspace(s, e - 1, min(20000, n_frames), dtype=int)
        left_hist = np.array(ja_l[hist_idx])
        right_hist = np.array(ja_r[hist_idx])

        fig, axes = plt.subplots(4, 5, figsize=(16, 12))
        for i, (ax, name) in enumerate(zip(axes.flat, ANGLE_NAMES)):
            ax.hist(left_hist[:, i], bins=50, alpha=0.5, label="Left", density=True)
            ax.hist(right_hist[:, i], bins=50, alpha=0.5, label="Right", density=True)
            ax.set_title(name.replace("\n", " "), fontsize=8)
            ax.tick_params(labelsize=6)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Angle Distributions — {ep_id}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ep_id}_histograms.png", dpi=150)
        plt.close(fig)

        # ── Plot 3: Hand landmark visualization ──
        # Pick 3 frames with valid labels and different gestures
        valid_mask = valid[:, 0] & valid[:, 1]
        valid_frames = np.where(valid_mask)[0]
        if len(valid_frames) > 0:
            pick = valid_frames[np.linspace(0, len(valid_frames) - 1, min(4, len(valid_frames)), dtype=int)]

            fig = plt.figure(figsize=(12, 8))
            for pi, pf in enumerate(pick):
                actual_idx = sample_idx[pf]
                ja_l_t = torch.from_numpy(left[pf:pf + 1])
                ja_r_t = torch.from_numpy(right[pf:pf + 1])
                zeros = torch.zeros(1, 2)
                ja_l_22 = torch.cat([ja_l_t, zeros], dim=1)
                ja_r_22 = torch.cat([ja_r_t, zeros], dim=1)
                hm = broadcast_hand_model_to(load_default_hand_model(), (1,))
                wt = torch.eye(4).unsqueeze(0)
                lm_l = skin_landmarks(hm, ja_l_22, wrist_transforms=wt)[0].numpy()
                lm_r = skin_landmarks(hm, ja_r_22, wrist_transforms=wt)[0].numpy()

                ax = fig.add_subplot(2, 4, pi + 1, projection="3d")
                ax.scatter(lm_l[:, 0], lm_l[:, 1], lm_l[:, 2], c="blue", s=10, label="Left")
                ax.scatter(lm_r[:, 0], lm_r[:, 1], lm_r[:, 2], c="red", s=10, label="Right")
                g = gestures[pf]
                ax.set_title(f"Frame {actual_idx}\nGesture={g}", fontsize=8)
                ax.set_xlim(-200, 200)
                ax.set_ylim(-300, 100)
                ax.set_zlim(-200, 200)
                ax.tick_params(labelsize=6)
            fig.suptitle(f"UmeTrack FK Landmarks — {ep_id}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f"{ep_id}_landmarks.png", dpi=150)
            plt.close(fig)

        print(f"Saved {ep_id}: trajectories, histograms, landmarks")

    # ── Global summary ──
    print(f"\nGlobal statistics (10k random valid frames):")
    idx = rng.choice(66161725, size=10000, replace=False)
    l_sample = np.array(ja_l[idx])
    r_sample = np.array(ja_r[idx])
    v_sample = vm[idx]
    valid = v_sample[:, 0] & v_sample[:, 1]
    l_valid = l_sample[valid]
    r_valid = r_sample[valid]

    print(f"  Left:  min={l_valid.min():.3f}  max={l_valid.max():.3f}")
    print(f"  Right: min={r_valid.min():.3f}  max={r_valid.max():.3f}")
    print(f"  Left  per-DOF range: {[f'{l_valid[:,i].max()-l_valid[:,i].min():.2f}' for i in range(20)]}")
    print(f"  Right per-DOF range: {[f'{r_valid[:,i].max()-r_valid[:,i].min():.2f}' for i in range(20)]}")
    print(f"  Left  total range: {l_valid.max(axis=0).sum() - l_valid.min(axis=0).sum():.1f} rad")
    print(f"  Right total range: {r_valid.max(axis=0).sum() - r_valid.min(axis=0).sum():.1f} rad")

    # Check for any ±π artifacts
    near_pi = np.abs(np.abs(l_valid) - np.pi) < 0.01
    if near_pi.any():
        print(f"  ⚠️ Left: {near_pi.sum()} near-π values")
    else:
        print(f"  ✓ Left: no ±π artifacts")
    near_pi = np.abs(np.abs(r_valid) - np.pi) < 0.01
    if near_pi.any():
        print(f"  ⚠️ Right: {near_pi.sum()} near-π values")
    else:
        print(f"  ✓ Right: no ±π artifacts")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
