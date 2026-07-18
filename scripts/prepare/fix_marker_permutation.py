#!/usr/bin/env python3
"""Fix episodes 4,5,6 rigid body marker ID assignment, recompute transforms.

Uses Kabsch algorithm with the correct marker permutation to recover
T_Rigid3_to_Camera, then computes world-to-camera transforms and updates
the memmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from scipy.spatial.transform import Rotation

DATA_DIR = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG/data")
ASSETS_DIR = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG/reprojection_assets")
MEMMAP_DIR = Path("/mnt/nvme/xiziheng/EgoEMG_v2_memmap")

# Permutation to apply to markers to match reference (Ep 1) geometry
# (2, 0, 1) means: new_M0 = old_M2, new_M1 = old_M0, new_M2 = old_M1
PERM = {4: [2, 0, 1], 5: [2, 0, 1], 6: [1, 2, 0]}


def get_markers(pq_file):
    """Load all markers for a parquet file, return (N, 3) numpy array."""
    pf = pq.ParquetFile(pq_file)
    tbl = pf.read_row_group(0, columns=["observation.mocap.webcam.rigid_markers"])
    marker_col = tbl.column("observation.mocap.webcam.rigid_markers")
    combined = marker_col.combine_chunks()
    flat = combined.flatten()
    n_per_frame = len(combined[0].values)
    flat_py = flat.to_pylist()
    all_markers = np.array([flat_py[j] for j in range(len(flat_py))], dtype=np.float32)
    return all_markers.reshape(-1, n_per_frame, 3)


def kabsch(P, Q):
    """Find optimal rotation R and translation t that aligns P -> Q.
    P, Q: (N, 3) arrays. Returns R (3,3), t (3,) such that Q ≈ P @ R.T + t.
    """
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_c = P - p_mean
    Q_c = Q - q_mean
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    return R, t


def compute_transform_for_episode(ep_num):
    """Recompute T_Rigid3_to_Camera for a single episode using corrected markers."""
    ep_id = f"episode_{ep_num:06d}"
    pq_file = sorted(DATA_DIR.glob(f"chunk-*/{ep_id}.parquet"))[0]

    # Load all markers
    markers_raw = get_markers(pq_file)  # (N_frames, 4, 3) - including zero marker

    # Filter frames with valid markers (non-zero)
    nonzero_mask = ~np.all(np.isclose(markers_raw[:, :, 0], 0), axis=1) & \
                   ~np.all(np.isclose(markers_raw[:, :, 1], 0), axis=1) & \
                   ~np.all(np.isclose(markers_raw[:, :, 2], 0), axis=1)

    # Get first valid frame's active markers
    valid = markers_raw[nonzero_mask]
    active_markers = valid[0]  # (N_active, 3)

    print(f"  Episode {ep_num}: {len(valid)} valid frames, {len(active_markers)} active markers")

    # Apply permutation to match reference
    perm = PERM[ep_num]
    reordered = active_markers[perm]

    print(f"  Original markers:    {active_markers[:3]}")
    print(f"  Reordered markers:   {reordered}")

    # Use reference markers from Ep 1 as the rigid body template
    ref_pq = sorted(DATA_DIR.glob("chunk-*/episode_000001.parquet"))[0]
    ref_markers_raw = get_markers(ref_pq)
    ref_valid = ref_markers_raw[~np.all(np.isclose(ref_markers_raw[:, :, 0], 0), axis=1)]
    ref_markers = ref_valid[0][:3]  # Use first 3 markers only (matching active count)

    print(f"  Reference (Ep 1):    {ref_markers}")

    # Kabsch: find R, t that maps ref_markers -> reordered
    R, t = kabsch(ref_markers, reordered)

    print(f"  R = \n{R}")
    print(f"  t = {t}")

    # Verify: ref_markers @ R.T + t should ≈ reordered
    check = ref_markers @ R.T + t
    error = np.linalg.norm(check - reordered, axis=1).mean()
    print(f"  Mean fit error: {error*1000:.3f}mm")

    # Build T_Rigid3_to_Camera matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t * 1000  # meters to mm (matches original format)

    return T


def _quat_to_matrix(q_xyzw):
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((len(q_xyzw), 3, 3), dtype=q_xyzw.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w);   R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w);   R[:, 2, 1] = 2 * (y * z + x * w);     R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _read_list_column(table, name):
    col = table.column(name)
    combined = col.combine_chunks()
    values = combined.values.to_numpy()
    element_size = len(values) // len(combined)
    return values.reshape(len(combined), element_size)


def _smooth_position(pos):
    from scipy.signal import savgol_filter
    if len(pos) < 51:
        return pos
    return savgol_filter(pos, 51, 3, axis=0)


def _ensure_quat_consistent_sign(q):
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def _quat_to_rotvec(q_xyzw):
    q = q_xyzw.copy()
    w, x, y, z = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w * w)
    s = np.where(s < 1e-8, 1.0, s)
    axis = np.stack([x / s, y / s, z / s], axis=-1)
    return axis * angle[:, None]


def _rotvec_to_quat(rv):
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = rv / np.where(angle < 1e-8, 1.0, angle)
    half = angle / 2.0
    w = np.cos(half)
    s = np.sin(half)
    return np.concatenate([axis * s, w], axis=-1)


def _smooth_orientation(q_xyzw):
    from scipy.signal import savgol_filter
    if len(q_xyzw) < 51:
        return q_xyzw
    q = _ensure_quat_consistent_sign(q_xyzw)
    rv = _quat_to_rotvec(q)
    rv_smooth = savgol_filter(rv, 51, 3, axis=0)
    return _rotvec_to_quat(rv_smooth)


def main():
    # Step 1: Compute corrected T_Rigid3_to_Camera for each episode
    print("=== Computing corrected T_Rigid3_to_Camera ===\n")

    transforms_per_ep = {}
    for ep_num in [4, 5, 6]:
        T = compute_transform_for_episode(ep_num)
        transforms_per_ep[ep_num] = T
        print()

    # Also load old calibration for comparison
    with open(ASSETS_DIR / "old_rigid_transform_result.json") as f:
        T_old = np.array(json.load(f)["T_Rigid3_to_Camera"])

    print("\n=== Comparison ===")
    print(f"OLD calibration T:\n{np.array2string(T_old[:3, :3], precision=4)}")
    print(f"                 t={T_old[:3, 3]}")
    print()
    for ep_num, T in transforms_per_ep.items():
        print(f"Ep {ep_num} corrected T:\n{np.array2string(T[:3, :3], precision=4)}")
        print(f"                 t={T[:3, 3]}mm")
        print()

    # Save to JSON
    for ep_num, T in transforms_per_ep.items():
        output_path = ASSETS_DIR / f"ep{ep_num}_corrected_rigid_transform.json"
        result = {"T_Rigid3_to_Camera": T.tolist()}
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {output_path}")

    print("\n=== Done ===")
    print("Transforms saved. Now regenerate sidecar and update memmap.")


if __name__ == "__main__":
    main()
