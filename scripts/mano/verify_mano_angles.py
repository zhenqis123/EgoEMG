"""Verify stored joint_angles match recomputed MANO-derived angles."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")

ANGLE_NAMES = [
    "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
    "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
    "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
    "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
    "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
]

EULER_TO_EMG2POSE = {
    "thumb_cmc_fe": (13, 2), "thumb_cmc_aa": (13, 1),
    "thumb_mcp_fe": (14, 2), "thumb_ip_fe": (15, 2),
    "index_mcp_aa": (1, 1), "index_mcp_fe": (1, 2),
    "index_pip_fe": (2, 2), "index_dip_fe": (3, 2),
    "middle_mcp_aa": (4, 1), "middle_mcp_fe": (4, 2),
    "middle_pip_fe": (5, 2), "middle_dip_fe": (6, 2),
    "ring_mcp_aa": (10, 1), "ring_mcp_fe": (10, 2),
    "ring_pip_fe": (11, 2), "ring_dip_fe": (12, 2),
    "pinky_mcp_aa": (7, 1), "pinky_mcp_fe": (7, 2),
    "pinky_pip_fe": (8, 2), "pinky_dip_fe": (9, 2),
}


def _decode_bytes(values: np.ndarray):
    decoded = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


class AngleRecomputer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.right_mano = ManoLayer(
            rot_mode="axisang", side="right",
            mano_assets_root=str(MANO_ASSETS_ROOT),
            use_pca=False, flat_hand_mean=False,
        ).to(self.device)
        self.right_axis = AxisLayerFK(
            side="right", mano_assets_root=str(MANO_ASSETS_ROOT),
        ).to(self.device)
        self._indices = [(joint_idx, axis_idx) for joint_idx, axis_idx in
                         (EULER_TO_EMG2POSE[name] for name in ANGLE_NAMES)]

    def compute_angles(self, pose: np.ndarray, beta: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Recompute 20D joint angles from MANO pose + beta."""
        beta_t = torch.from_numpy(beta.astype(np.float32)[None]).to(self.device)
        outputs = []
        for start in range(0, pose.shape[0], batch_size):
            pose_b = torch.from_numpy(pose[start:start + batch_size].astype(np.float32)).to(self.device)
            beta_b = beta_t.expand(pose_b.shape[0], -1)
            with torch.no_grad():
                mano_out = self.right_mano(pose_b, beta_b)
                ee = self.right_axis(mano_out.transforms_abs)[2]
            angles = torch.stack([ee[:, j, a] for j, a in self._indices], dim=1)
            outputs.append(angles.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-root", type=str,
                        default="/home/xiziheng/develop/emg2pose/data/EgoEMG_memmap")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--samples-per-episode", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    memmap_root = Path(args.memmap_root)
    manifest = json.load(open(memmap_root / "manifest.json"))
    metadata = dict(np.load(memmap_root / "metadata.npz", allow_pickle=False))

    episode_ids = _decode_bytes(metadata["episode_id"])
    start_idx = metadata["episode_start_idx"]
    end_idx = metadata["episode_end_idx"]
    beta_idx = metadata["episode_beta_idx"]

    left_pose_mm = np.memmap(
        memmap_root / manifest["fields"]["generated_mano_left_pose"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["fields"]["generated_mano_left_pose"]["shape"]))
    right_pose_mm = np.memmap(
        memmap_root / manifest["fields"]["generated_mano_right_pose"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["fields"]["generated_mano_right_pose"]["shape"]))
    left_angles_mm = np.memmap(
        memmap_root / manifest["fields"]["generated_joint_angles_left"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["fields"]["generated_joint_angles_left"]["shape"]))
    right_angles_mm = np.memmap(
        memmap_root / manifest["fields"]["generated_joint_angles_right"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["fields"]["generated_joint_angles_right"]["shape"]))
    left_beta_mm = np.memmap(
        memmap_root / manifest["episode_fields"]["generated_mano_left_beta"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["episode_fields"]["generated_mano_left_beta"]["shape"]))
    right_beta_mm = np.memmap(
        memmap_root / manifest["episode_fields"]["generated_mano_right_beta"]["filename"],
        dtype=np.float32, mode="r",
        shape=tuple(manifest["episode_fields"]["generated_mano_right_beta"]["shape"]))

    recon = AngleRecomputer(args.device)
    rng = np.random.RandomState(args.seed)

    # Pick random episodes
    ep_indices = sorted(rng.choice(len(episode_ids), size=min(args.num_episodes, len(episode_ids)), replace=False))

    print(f"Memmap: {args.memmap_root}")
    print(f"Total episodes: {len(episode_ids)}, total rows: {manifest['total_rows']:,}")
    print(f"Sampling {args.num_episodes} episodes × ~{args.samples_per_episode} frames each")
    print()

    all_diffs_left = []
    all_diffs_right = []

    for ep_idx in ep_indices:
        ep_id = episode_ids[ep_idx]
        s, e = int(start_idx[ep_idx]), int(end_idx[ep_idx])
        b = int(beta_idx[ep_idx])
        n_frames = e - s

        sample_n = min(args.samples_per_episode, n_frames)
        local_indices = sorted(rng.choice(n_frames, size=sample_n, replace=False))
        global_indices = [s + li for li in local_indices]

        left_beta = left_beta_mm[b]
        right_beta = right_beta_mm[b]

        # Left hand
        left_pose = left_pose_mm[global_indices]
        left_stored = left_angles_mm[global_indices]
        left_recomputed = recon.compute_angles(left_pose, left_beta)
        left_diff = np.abs(left_stored - left_recomputed)

        # Right hand
        right_pose = right_pose_mm[global_indices]
        right_stored = right_angles_mm[global_indices]
        right_recomputed = recon.compute_angles(right_pose, right_beta)
        right_diff = np.abs(right_stored - right_recomputed)

        all_diffs_left.append(left_diff)
        all_diffs_right.append(right_diff)

        print(f"[{ep_id}] {n_frames:,} frames, sampled {sample_n}")
        print(f"  Left  - mean diff: {left_diff.mean():.6f}, max: {left_diff.max():.6f}")
        print(f"  Right - mean diff: {right_diff.mean():.6f}, max: {right_diff.max():.6f}")

    left_all = np.concatenate(all_diffs_left, axis=0)
    right_all = np.concatenate(all_diffs_right, axis=0)

    print(f"\n{'='*60}")
    print(f"Summary (N={left_all.shape[0]} frames × {len(ANGLE_NAMES)} joints)")
    print(f"  Left  - mean: {left_all.mean():.6f}, max: {left_all.max():.6f}, "
          f"median: {np.median(left_all):.6f}, >0.01: {(left_all > 0.01).mean()*100:.2f}%")
    print(f"  Right - mean: {right_all.mean():.6f}, max: {right_all.max():.6f}, "
          f"median: {np.median(right_all):.6f}, >0.01: {(right_all > 0.01).mean()*100:.2f}%")

    print(f"\nPer-joint MAE (radians):")
    print(f"{'Joint':<20s} {'Left MAE':>10s} {'Right MAE':>10s}")
    print("-" * 42)
    for i, name in enumerate(ANGLE_NAMES):
        print(f"{name:<20s} {left_all[:, i].mean():>10.6f} {right_all[:, i].mean():>10.6f}")

    # Check if angles stored in degrees instead of radians
    deg_factor_left = (np.abs(left_stored) / (np.abs(left_recomputed) + 1e-8)).mean()
    deg_factor_right = (np.abs(right_stored) / (np.abs(right_recomputed) + 1e-8)).mean()
    print(f"\nScale check (stored / recomputed): left={deg_factor_left:.4f}, right={deg_factor_right:.4f}")
    if 50 < deg_factor_left < 60:
        print("  WARNING: Left angles may be in degrees (stored) vs radians (recomputed)!")
    if 50 < deg_factor_right < 60:
        print("  WARNING: Right angles may be in degrees (stored) vs radians (recomputed)!")

    if left_all.max() < 1e-6 and right_all.max() < 1e-6:
        print("\nPASS: stored angles are numerically identical to recomputed angles.")
    elif left_all.mean() < 0.01 and right_all.mean() < 0.01:
        print("\nApproximate match (small floating-point differences, likely OK).")
    else:
        print("\nMISMATCH: stored angles differ significantly from recomputed values.")


if __name__ == "__main__":
    main()
