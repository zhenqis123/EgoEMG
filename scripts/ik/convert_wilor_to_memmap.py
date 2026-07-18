#!/usr/bin/env python3
"""Convert a session's WiLoR MANO .npy outputs to a minimal memmap dataset
compatible with scripts/ik/batch_ik_mesh.py.

Usage:
    python scripts/ik/convert_wilor_to_memmap.py \
        --input data/sess_20260530_140912/wilor_mano \
        --output data/sess_20260530_140912/memmap \
        --hand right

    python scripts/ik/convert_wilor_to_memmap.py \
        --input data/incre_2/wilor_mano \
        --output data/incre_2/memmap \
        --hand right
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing WiLoR .npy outputs")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output memmap directory")
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    args = parser.parse_args()

    src = args.input
    dst = args.output
    dst.mkdir(parents=True, exist_ok=True)
    hand = args.hand

    # Load source data
    hand_pose = np.load(src / "mano_hand_pose.npy").astype(np.float32)  # (N, 45)
    global_orient = np.load(src / "mano_global_orient.npy").astype(np.float32)  # (N, 3)
    betas = np.load(src / "mano_betas.npy").astype(np.float32)  # (N, 10)
    transl = np.load(src / "mano_transl.npy").astype(np.float32) if (src / "mano_transl.npy").exists() else np.zeros((hand_pose.shape[0], 3), dtype=np.float32)
    valid = np.load(src / "valid.npy").astype(bool) if (src / "valid.npy").exists() else np.ones(hand_pose.shape[0], dtype=bool)

    n_frames = hand_pose.shape[0]
    print(f"Frames: {n_frames}")

    # Combine global_orient + hand_pose into 48D MANO pose
    mano_pose = np.concatenate([global_orient, hand_pose], axis=1)  # (N, 48)

    # Write memmap files (float32)
    def _save(name: str, data: np.ndarray) -> None:
        path = dst / f"{name}.dat"
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=data.shape)
        mm[:] = data[:]
        mm.flush()
        print(f"  {name}: {data.shape} → {path}")

    _save(f"generated_mano_{hand}_pose", mano_pose)
    _save(f"generated_mano_{hand}_beta", betas)
    _save(f"generated_mano_{hand}_world_transform", transl)

    # Write metadata.npz (batch_ik_mesh.py reads episode fields for beta)
    episode_id = np.array([src.parent.name], dtype=object)
    np.savez(
        dst / "metadata.npz",
        episode_id=episode_id,
        episode_beta_idx=np.zeros(1, dtype=np.int64),
    )

    # Write manifest.json
    manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": n_frames,
        "num_episodes": 1,
        "fields": {
            f"generated_mano_{hand}_pose": {
                "filename": f"generated_mano_{hand}_pose.dat",
                "dtype": "float32",
                "shape": [n_frames, 48],
            },
            f"generated_mano_{hand}_beta": {
                "filename": f"generated_mano_{hand}_beta.dat",
                "dtype": "float32",
                "shape": [n_frames, 10],
            },
            f"generated_mano_{hand}_world_transform": {
                "filename": f"generated_mano_{hand}_world_transform.dat",
                "dtype": "float32",
                "shape": [n_frames, 3],
            },
        },
        "episode_fields": {
            f"generated_mano_{hand}_beta": {
                "filename": f"generated_mano_{hand}_beta.dat",
                "dtype": "float32",
                "shape": [1, 10],
            },
        },
        "generated_joint_angles_semantics": [
            "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
            "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
            "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
            "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
            "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
        ],
    }
    with open(dst / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written")

    # Copy valid and other metadata for reference
    np.save(dst / "valid.npy", valid)
    shutil.copy2(src / "metadata.json", dst / "wilor_metadata.json")

    print(f"\nDone. Memmap dataset at: {dst}")
    print(f"\nNext step:")
    print(f"  python scripts/ik/batch_ik_mesh.py --memmap-root {dst} --gpus 0 --hand {hand}")


if __name__ == "__main__":
    main()
