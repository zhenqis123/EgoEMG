"""Quick-and-dirty: save one sample per episode for offset debugging."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import numpy as np

_runtime_cache_root = Path(tempfile.gettempdir()) / "emg2pose_viz_runtime"
_runtime_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache_root / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache_root / "xdg-cache"))

import cv2
from egoemg.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

MEMMAP_DIR = Path("data/EgoEMG_v2_memmap")
VIDEO_ROOT = Path("./data/EgoEMG")
ALLINTRA_ROOT = Path("data/EgoEMG_allintra")
OUTPUT_DIR = Path("./visualizations/per_episode_samples").resolve()
PATCH_SIZE = 256

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load episode info
metadata = np.load(MEMMAP_DIR / "metadata.npz", allow_pickle=False)
ep_ids = [x.decode().rstrip("\x00") if isinstance(x, bytes) else str(x) for x in metadata["episode_id"]]
ep_starts = metadata["episode_start_idx"]
ep_ends = metadata["episode_end_idx"]

# Pick a stride that gives ~10 samples per episode, then take the 5th (mid-episode)
STRIDE = 30  # coarse stride
SAMPLE_PER_EP = 1
NUM_TOTAL_PER_EP = 5  # grab first 5 at coarse stride, save the middle one
TARGET = NUM_TOTAL_PER_EP // 2  # index 2

for ep_idx, ep_id in enumerate(ep_ids):
    frames = int(ep_ends[ep_idx] - ep_starts[ep_idx])
    if frames < 100:
        print(f"[{ep_id}] too short ({frames} frames), skip")
        continue

    if (OUTPUT_DIR / f"{ep_id}_left.png").exists() and (OUTPUT_DIR / f"{ep_id}_right.png").exists():
        print(f"[{ep_id}] already done, skip")
        continue

    try:
        dataset_left = EgoEmgVisionDataset(
            memmap_dir=MEMMAP_DIR,
            video_root=VIDEO_ROOT,
            allintra_root=ALLINTRA_ROOT,
            allowed_episode_ids=[ep_id],
            target_hand="left",
            stride=STRIDE,
            index_limit=NUM_TOTAL_PER_EP,
            patch_size=PATCH_SIZE,
            return_frame_bgr=True,
            log_init_timing=False,
        )
        dataset_right = EgoEmgVisionDataset(
            memmap_dir=MEMMAP_DIR,
            video_root=VIDEO_ROOT,
            allintra_root=ALLINTRA_ROOT,
            allowed_episode_ids=[ep_id],
            target_hand="right",
            stride=STRIDE,
            index_limit=NUM_TOTAL_PER_EP,
            patch_size=PATCH_SIZE,
            return_frame_bgr=True,
            log_init_timing=False,
        )
    except Exception as e:
        print(f"[{ep_id}] dataset build failed: {e}")
        continue

    if len(dataset_left) < TARGET + 1 or len(dataset_right) < TARGET + 1:
        print(f"[{ep_id}] not enough samples (L={len(dataset_left)} R={len(dataset_right)}), skip")
        continue

    for side, ds, hand in [("left", dataset_left, "L"), ("right", dataset_right, "R")]:
        sample = ds[TARGET]
        frame_bgr = sample["frame_bgr"].copy()
        bbox = sample["bbox"].astype(np.float32)
        keypoints_2d = sample["orig_keypoints_2d"].copy()
        markers_2d = sample["orig_markers_2d"].copy()
        ep_id_str = str(sample["episode_id"])
        frame_idx = int(sample["frame_index"])
        raw_right = float(sample["raw_right"])

        # Draw bbox
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (255, 180, 0), 2)

        # Draw keypoints
        for pt in keypoints_2d:
            if pt[2] > 0:
                cv2.circle(frame_bgr, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 220, 0), -1)

        # Draw markers
        for pt in markers_2d:
            if pt[2] > 0:
                cv2.circle(frame_bgr, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1)

        # Text overlay
        cv2.putText(frame_bgr, f"ep={ep_id_str} frame={frame_idx} {side}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(str(OUTPUT_DIR / f"{ep_id}_{side}.png"), frame_bgr)
        print(f"[{ep_id}] saved {side} (frame={frame_idx})")

print("Done.")
