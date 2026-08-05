#!/usr/bin/env python3
"""Validate marker permutation against stored camera extrinsics for ep1-3.

For each episode, samples a few frames, tries all 6 marker permutations,
and compares the recovered camera pose against the memmap ground truth.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

DATA_DIR = Path("./data/EgoEMG/data")
MEMMAP_DIR = Path("data/EgoEMG_v2_memmap")

R_M2C_MEAN = np.array([
    [ 8.25415214e-01, -5.64515920e-01, -3.39126844e-03],
    [ 5.64526080e-01,  8.25398437e-01,  5.26546352e-03],
    [-1.73290315e-04, -6.26065318e-03,  9.99980387e-01],
], dtype=np.float64)
T_M2C_MEAN = np.array([-0.01020487, -0.03292008, 0.00147814], dtype=np.float64)

PARQUET_COLUMNS = ["observation.mocap.webcam.rigid_markers"]
NUM_SAMPLES = 50


def load_parquet_markers(pq_file):
    pf = pq.ParquetFile(pq_file)
    all_markers = []
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=PARQUET_COLUMNS)
        col = tbl.column(PARQUET_COLUMNS[0]).combine_chunks()
        n_per = len(col[0])
        flat_py = col.flatten().to_pylist()
        arr = np.array(flat_py, dtype=np.float64).reshape(len(col), n_per, 3)
        all_markers.append(arr)
    return np.concatenate(all_markers, axis=0)


def compute_single_cam_pose(marker_row, perm):
    """Compute (R_cam, t_cam) for a single marker row with given permutation."""
    m = marker_row[:3][list(perm)]
    c = m.mean(axis=0)
    e0 = m[1] - m[0]
    e1 = m[2] - m[0]
    n = np.cross(e0, e1)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return None, None
    n /= n_norm
    e0_norm = np.linalg.norm(e0)
    if e0_norm < 1e-12:
        return None, None
    x = e0 / e0_norm
    y = np.cross(n, x)
    R_mw = np.column_stack([x, y, n])
    R_cam = R_mw @ R_M2C_MEAN
    t_cam = c + R_mw @ T_M2C_MEAN
    return R_cam, t_cam


def main():
    with open(MEMMAP_DIR / "manifest.json") as f:
        manifest = json.load(f)
    fields = manifest["fields"]
    md = np.load(MEMMAP_DIR / "metadata.npz", allow_pickle=False)
    ep_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in md["episode_id"]]
    ep_starts = np.asarray(md["episode_start_idx"], dtype=np.int64)
    ep_ends = np.asarray(md["episode_end_idx"], dtype=np.int64)

    cam_tf_mm = np.memmap(
        MEMMAP_DIR / fields["mocap_head_transform"]["filename"],
        dtype=fields["mocap_head_transform"]["dtype"], mode="r",
        shape=tuple(fields["mocap_head_transform"]["shape"]),
    )

    all_perms = list(permutations([0, 1, 2]))

    for ep_id in ["episode_000001", "episode_000002", "episode_000003"]:
        print(f"\n{'='*60}")
        print(f"Episode: {ep_id}")
        ep_pq = sorted(DATA_DIR.glob(f"chunk-*/{ep_id}.parquet"))[0]
        markers = load_parquet_markers(ep_pq)

        ep_idx = ep_ids.index(ep_id)
        start = int(ep_starts[ep_idx])
        length = int(ep_ends[ep_idx] - ep_starts[ep_idx])
        print(f"  markers: {len(markers)} rows, memmap range: [{start}, {start+length})")

        # Check what the stored transform looks like
        sample_tf = cam_tf_mm[start]
        print(f"  stored tf[0] (12 floats): {sample_tf}")
        R_stored = sample_tf[:9].reshape(3, 3)
        t_stored = sample_tf[9:12]
        print(f"  R_stored[0]:\n{R_stored}")
        print(f"  t_stored[0]: {t_stored}")
        print(f"  det(R_stored): {np.linalg.det(R_stored):.6f}")

        # Sample frames with valid markers
        valid_mask = ~np.all(np.isclose(markers[:, :, 0], 0), axis=1)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            print("  No valid markers!")
            continue
        step = max(1, len(valid_indices) // NUM_SAMPLES)
        sample_indices = valid_indices[::step][:NUM_SAMPLES]
        print(f"  Sampling {len(sample_indices)} frames from {len(valid_indices)} valid")

        # For each perm, compute error vs stored
        for perm in all_perms:
            perm_label = f"{perm[0]}{perm[1]}{perm[2]}"
            rot_errors = []
            t_errors = []
            for idx in sample_indices:
                global_idx = start + idx
                stored = cam_tf_mm[global_idx].astype(np.float64)
                R_gt = stored[:9].reshape(3, 3)
                t_gt = stored[9:12]
                # Skip if stored is zero/invalid
                if np.allclose(R_gt, 0):
                    continue
                R_pred, t_pred = compute_single_cam_pose(markers[idx], perm)
                if R_pred is None:
                    continue
                # Rotation error in degrees
                R_rel = R_gt.T @ R_pred
                cos_a = (np.trace(R_rel) - 1) / 2
                rot_err = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                rot_errors.append(rot_err)
                # Translation error in mm
                t_errors.append(np.linalg.norm(t_gt - t_pred) * 1000)

            if rot_errors:
                rot_errors = np.array(rot_errors)
                t_errors = np.array(t_errors)
                print(f"  perm={perm_label}: n={len(rot_errors)}  "
                      f"rot_err={rot_errors.mean():.2f}+/-{rot_errors.std():.2f}deg  "
                      f"t_err={t_errors.mean():.2f}+/-{t_errors.std():.2f}mm")
            else:
                print(f"  perm={perm_label}: no valid comparisons")


if __name__ == "__main__":
    main()
