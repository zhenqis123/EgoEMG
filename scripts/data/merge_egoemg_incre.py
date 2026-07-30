#!/usr/bin/env python
"""Merge multiple EgoEMG incremental datasets into one, keeping only right-hand
EMG + joint_angles + essential index fields. All synthetic/left-hand/image/mocap
fields are dropped."""

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiziheng/develop/emg2pose/data/EgoEMG_incre")
SRC_DIRS = ["data_20260526_172725", "data_20260526_230859", "data_20260527_124150"]
OUT_DIR = ROOT / "data_merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fields to KEEP in the merged dataset ────────────────────────────
FRAME_FIELDS = [
    "emg_right_raw",
    "emg_right_filtered",
    "generated_joint_angles_right",
    "generated_label_valid",
    "timestamp",
    "timestamp_us",
    "episode_index",
    "frame_index",
    "is_first",
    "is_last",
    "is_terminal",
    "frame_split_id",
    "subject_id",
    "source_index",
    "task_index",
]

EPISODE_FIELDS = ["generated_mano_right_beta"]

# ── Load manifests & metadata ───────────────────────────────────────
manifests = []
metadatas = []
row_counts = []

for d in SRC_DIRS:
    with open(ROOT / d / "manifest.json") as f:
        manifests.append(json.load(f))
    metadatas.append(np.load(ROOT / d / "metadata.npz", allow_pickle=True))
    row_counts.append(manifests[-1]["total_rows"])

total_rows = sum(row_counts)
total_episodes = sum(m["num_episodes"] for m in manifests)
print(f"Merging {len(SRC_DIRS)} datasets: {row_counts}  ->  {total_rows} total rows, {total_episodes} episodes")

# ── Verify field compatibility ──────────────────────────────────────
for field in FRAME_FIELDS + EPISODE_FIELDS:
    shapes = set()
    dtypes = set()
    for m, d in zip(manifests, SRC_DIRS):
        if field in m.get("episode_fields", {}):
            fi = m["episode_fields"][field]
        elif field in m.get("fields", {}):
            fi = m["fields"][field]
        else:
            print(f"  WARNING: {field} missing in {d}, will be zero-filled")
            continue
        shapes.add(tuple(fi["shape"][1:] if len(fi["shape"]) > 1 else ()))
        dtypes.add(fi["dtype"])
    if len(shapes) > 1 or len(dtypes) > 1:
        raise ValueError(f"Field {field} has incompatible shapes/dtypes across datasets: shapes={shapes}, dtypes={dtypes}")

# ── Merge frame-level fields ────────────────────────────────────────
new_manifest_fields = {}
offset = 0

for field in FRAME_FIELDS:
    # Determine dtype and shape from first dataset that has it
    ref_dtype = None
    ref_per_row_shape = ()
    for m in manifests:
        if field in m.get("fields", {}):
            fi = m["fields"][field]
            ref_dtype = np.dtype(fi["dtype"])
            ref_per_row_shape = tuple(fi["shape"][1:])
            break
    if ref_dtype is None:
        print(f"  SKIP {field}: not in any manifest")
        continue

    full_shape = (total_rows,) + ref_per_row_shape
    out_path = OUT_DIR / f"{field}.dat"
    print(f"  Writing {field}: {ref_dtype} {full_shape}  ->  {out_path}")

    out_mmap = np.memmap(out_path, dtype=ref_dtype, mode="w+", shape=full_shape)

    offset = 0
    for src_dir in SRC_DIRS:
        src_path = ROOT / src_dir / f"{field}.dat"
        n = row_counts[SRC_DIRS.index(src_dir)]
        if src_path.exists():
            src_mmap = np.memmap(src_path, dtype=ref_dtype, mode="r", shape=(n,) + ref_per_row_shape)
            out_mmap[offset:offset + n] = src_mmap[:]
        # else: leave as zeros
        offset += n

    out_mmap.flush()

    new_manifest_fields[field] = {
        "filename": f"{field}.dat",
        "dtype": ref_dtype.name,
        "shape": list(full_shape),
    }

# ── Merge episode_index (re-number from 0..N-1 across datasets) ────
ep_mmap = np.memmap(
    OUT_DIR / "episode_index.dat", dtype=np.int64, mode="r+", shape=(total_rows,)
)
row_offset = 0
ep_offset = 0
for i, src_dir in enumerate(SRC_DIRS):
    n = row_counts[i]
    n_ep = manifests[i]["num_episodes"]
    src_ep = np.memmap(ROOT / src_dir / "episode_index.dat", dtype=np.int64, mode="r", shape=(n,))
    ep_mmap[row_offset:row_offset + n] = src_ep[:] + ep_offset
    row_offset += n
    ep_offset += n_ep
ep_mmap.flush()

# ── Merge episode-level fields ──────────────────────────────────────
new_episode_fields = {}
for field in EPISODE_FIELDS:
    ref_dtype = None
    ref_per_ep_shape = ()
    for m in manifests:
        if field in m.get("episode_fields", {}):
            fi = m["episode_fields"][field]
            ref_dtype = np.dtype(fi["dtype"])
            ref_per_ep_shape = tuple(fi["shape"][1:])
            break
    if ref_dtype is None:
        continue

    full_shape = (total_episodes,) + ref_per_ep_shape
    out_path = OUT_DIR / f"{field}.dat"
    print(f"  Writing episode field {field}: {ref_dtype} {full_shape}")

    out_mmap = np.memmap(out_path, dtype=ref_dtype, mode="w+", shape=full_shape)
    ep_idx = 0
    for i, src_dir in enumerate(SRC_DIRS):
        src_path = ROOT / src_dir / f"{field}.dat"
        n_ep = manifests[i]["num_episodes"]
        if src_path.exists():
            src_mmap = np.memmap(src_path, dtype=ref_dtype, mode="r", shape=(n_ep,) + ref_per_ep_shape)
            out_mmap[ep_idx:ep_idx + n_ep] = src_mmap[:]
        ep_idx += n_ep
    out_mmap.flush()

    new_episode_fields[field] = {
        "filename": f"{field}.dat",
        "dtype": ref_dtype.name,
        "shape": list(full_shape),
    }

# ── Write manifest.json ─────────────────────────────────────────────
new_manifest = {
    "format_version": "egoemg_v2_memmap",
    "total_rows": total_rows,
    "num_episodes": total_episodes,
    "left_hand_strategy": "flip_local_z",
    "mano_label_policy": "generated_only",
    "source_root": str(OUT_DIR),
    "fields": new_manifest_fields,
    "episode_fields": new_episode_fields,
    "generated_joint_angles_semantics": manifests[0]["generated_joint_angles_semantics"],
    "frame_split_labels": ["train", "val", "test"],
    "frame_split_policy": "merged from 3 incremental sessions; per-session split preserved",
}

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(new_manifest, f, indent=2)
print(f"  Wrote manifest.json")

# ── Merge metadata.npz ─────────────────────────────────────────────
all_episode_ids = []
all_subjects = []
all_subject_ids = []
episode_start_idx = []
episode_end_idx = []

cumulative = 0
for i, src_dir in enumerate(SRC_DIRS):
    n = row_counts[i]
    meta = metadatas[i]
    all_episode_ids.append(str(meta["episode_id"][0]).lstrip("b'").rstrip("'"))
    all_subjects.append(str(meta["episode_subject"][0]).lstrip("b'").rstrip("'"))
    all_subject_ids.append(int(meta["episode_subject_id"][0]))
    episode_start_idx.append(cumulative)
    episode_end_idx.append(cumulative + n - 1)
    cumulative += n

unique_subjects = list(dict.fromkeys(all_subjects))
unique_subject_ids = list(range(len(unique_subjects)))

new_meta = {
    "episode_id": np.array([s.encode() for s in all_episode_ids]),
    "episode_subject": np.array([s.encode() for s in all_subjects]),
    "episode_subject_id": np.array([unique_subjects.index(s) for s in all_subjects]),
    "episode_chunk_id": np.array([s.encode() for s in all_episode_ids]),
    "episode_source_parquet": np.array([b""] * total_episodes),
    "episode_zed_video_path": np.array([b""] * total_episodes),
    "episode_webcam_video_path": np.array([b""] * total_episodes),
    "episode_start_idx": np.array(episode_start_idx),
    "episode_end_idx": np.array(episode_end_idx),
    "episode_length": np.array(row_counts),
    "episode_beta_idx": np.arange(total_episodes),
    "episode_split_id": np.arange(total_episodes),
    "subjects_subject": np.array([s.encode() for s in unique_subjects]),
    "subjects_subject_id": np.array(unique_subject_ids),
    "splits_split": np.array([b"train", b"val", b"test"]),
    "splits_split_id": np.array([0, 1, 2]),
}
np.savez(OUT_DIR / "metadata.npz", **new_meta)
print(f"  Wrote metadata.npz")

# ── Summary ─────────────────────────────────────────────────────────
size_gb = sum(f.stat().st_size for f in OUT_DIR.glob("*.dat")) / (1024 ** 3)
print(f"\nMerged dataset: {OUT_DIR}")
print(f"  Total rows: {total_rows:,} ({total_rows / 2000 / 3600:.1f} hours @ 2000Hz)")
print(f"  Episodes: {total_episodes}")
print(f"  Fields: {len(new_manifest_fields)} frame + {len(new_episode_fields)} episode")
print(f"  Size: {size_gb:.2f} GB")
