"""Visualize per-episode hand crops from LMDB.

Writes a grid image per sampled frame showing left/right hand crops side by side.

Usage:
    python scripts/viz/visualize_egoemg_crops.py \
        --crops-dir /mnt/nvme/xiziheng/EgoEMG_v2_crops \
        --output-dir /tmp/egoemg_crops_viz \
        --num-frames 16
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image


def _read_crop(txn, key: str) -> np.ndarray | None:
    raw = txn.get(key.encode("ascii"))
    if raw is None:
        return None
    img = Image.open(io.BytesIO(raw))
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of frames to sample per episode")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices (0-based)")
    parser.add_argument("--grid-cols", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.crops_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {args.crops_dir}")

    import json
    with manifest_path.open() as f:
        manifest = json.load(f)

    ep_ids = manifest["episode_ids"]
    patch_size = manifest.get("patch_size", 256)
    print(f"Found {len(ep_ids)} episodes, patch_size={patch_size}")

    if args.episodes is not None:
        indices = [int(x) for x in args.episodes.split(",")]
    else:
        indices = list(range(len(ep_ids)))

    for ep_i in indices:
        if ep_i >= len(ep_ids):
            print(f"  Episode index {ep_i} out of range, skipping")
            continue
        ep_id = ep_ids[ep_i]
        lmdb_path = args.crops_dir / f"{ep_id}.lmdb"
        done_path = args.crops_dir / f"{ep_id}.done"

        if not lmdb_path.exists():
            print(f"  [{ep_id}] LMDB not found, skipping")
            continue

        env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            all_keys = [k.decode("ascii") for k, _ in cursor]

        if not all_keys:
            print(f"  [{ep_id}] empty LMDB, skipping")
            env.close()
            continue

        # Collect unique vfi (video frame indices) and sort
        vfis = sorted(set(int(k.split("_")[0]) for k in all_keys))
        print(f"  [{ep_id}] {len(all_keys)} crops, {len(vfis)} frames")

        # Sample frames evenly
        n_sample = min(args.num_frames, len(vfis))
        step = max(len(vfis) // n_sample, 1)
        sampled_vfis = vfis[::step][:n_sample]

        with env.begin() as txn:
            rows = []
            for vfi in sampled_vfis:
                left_key = f"{vfi:08d}_L"
                right_key = f"{vfi:08d}_R"
                left = _read_crop(txn, left_key)
                right = _read_crop(txn, right_key)

                # Convert RGB to BGR for OpenCV
                if left is not None:
                    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
                else:
                    left = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    cv2.putText(left, "NO L", (10, patch_size // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if right is not None:
                    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
                else:
                    right = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    cv2.putText(right, "NO R", (10, patch_size // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add label
                left_lab = left.copy()
                right_lab = right.copy()
                cv2.putText(left_lab, f"L vfi={vfi}", (4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(right_lab, f"R vfi={vfi}", (4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                pair = np.concatenate([left_lab, right_lab], axis=1)
                rows.append(pair)

        env.close()

        # Arrange into grid
        cols = args.grid_cols
        n_rows = (len(rows) + cols - 1) // cols
        pad_val = 30
        grid_rows = []
        for r in range(n_rows):
            chunks = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(rows):
                    chunks.append(rows[idx])
                else:
                    chunks.append(np.zeros_like(rows[0]))
            grid_rows.append(np.concatenate(chunks, axis=1))

        grid = np.concatenate(grid_rows, axis=0)
        # Add thin white borders between cells
        h, w = rows[0].shape[:2]
        for r in range(1, n_rows):
            y = r * h
            if y < grid.shape[0]:
                grid[y, :] = pad_val
        for c in range(1, cols):
            x = c * w
            if x < grid.shape[1]:
                grid[:, x] = pad_val

        out_path = args.output_dir / f"{ep_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"  [{ep_id}] saved {out_path} ({grid.shape[1]}x{grid.shape[0]})")

    print(f"\nDone. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
