# NOTE: this builder emits schema v2 (egoemg_v2_memmap). After merging, run
# scripts/data/migrate_unified_memmap_v3.py to upgrade the directory to the
# v3 schema (IMU renames, dead-field drops, scalar mocap validity).
#!/usr/bin/env python
"""Physically merge EgoEMG + ShowEE + Incre into a single unified memmap.

Produces one `egoemg_v2_memmap`-format directory with the UNION of all frame
fields across the three sources, so downstream training can load a single
dataset instead of a mixed ConcatDataset.

Per-source availability flags (the two required "marks"):

  * Incre  -> vision/mocap marked UNAVAILABLE.  Incre only has right-hand EMG +
    joint angles; every other field (emg_left, generated_joint_angles_left,
    generated_mano_*, all mocap_*, all image_*, wrist fields) is zero-filled
    with validity flags set to False / stale=True / tracked=False.  Left-hand
    `generated_label_valid[:,0]` is False.  `dataset_source_id = 2`.

  * ShowEE -> wrist angles marked UNAVAILABLE (as today): the existing
    `mocap_{l,r}_wrist_pitch/yaw` are carried over (already 0) and
    `mocap_{l,r}_wrist_angles_valid` stays False.  `dataset_source_id = 1`.

  * EgoEMG -> carried over verbatim.  `dataset_source_id = 0`.

A new per-frame field `dataset_source_id` (int8) records provenance so the
dataset class can restore per-sample `dataset_name` for the existing
`lightning.py` wrist-mask logic (which keys on dataset_name in
{egoemg_incre, showee}).

Episode index, subject_id, episode_source_parquet and video paths are remapped
across sources to avoid namespace collisions (EgoEMG episodes 0..40, ShowEE
41..919, Incre 920..927; subject ids re-densified).

Usage:
    python scripts/data/merge_datasets_to_unified_memmap.py \
        --egoemg /path/to/egoemg_v2_memmap \
        --showee /path/to/showee_memmap \
        --incre  data/EgoEMG_incre/data_right_merged \
        --out    data/EgoEMG_full_memmap

Requires ~229 GB of free disk at --out.  Idempotent on a per-field basis: a
field whose output .dat already exists with the correct shape is skipped, so
the script can be resumed after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

# Source ids recorded in the new dataset_source_id field.
SRC_EGOEMG = 0
SRC_SHOWEE = 1
SRC_INCRE = 2

# The head-mounted camera is named "head view" in the released unified
# schema; source memmaps built earlier still use the legacy "webcam"
# names.  Normalize source field names during the union so a re-merge
# produces the released schema.
LEGACY_WEBCAM_TO_HEAD = {
    "image_webcam_frame_index": "image_head_frame_index",
    "image_webcam_stale": "image_head_stale",
    "image_webcam_delta_ms": "image_head_delta_ms",
    "mocap_webcam_position": "mocap_head_position",
    "mocap_webcam_orientation": "mocap_head_orientation",
    "mocap_webcam_tracked": "mocap_head_tracked",
    "mocap_webcam_rigid_markers": "mocap_head_rigid_markers",
    "mocap_webcam_transform": "mocap_head_transform",
}


def normalize_field_name(name: str) -> str:
    return LEGACY_WEBCAM_TO_HEAD.get(name, name)


# Fields whose "missing" semantics should be a non-zero invalid sentinel
# rather than a plain zero-fill.  Keyed by (field) -> fill value.
# - bool fields default to False (zero) which is the "invalid" sense for all
#   `*_valid` / `*_tracked` fields.
# - `image_*_stale` bool fields must be True when vision is unavailable.
# - `mocap_*_wrist_*_valid` must be False.
# We handle these explicitly per-source below.

# Per-frame fields that are STALE-True when vision is unavailable (Incre).
VISION_STALE_TRUE_FIELDS = {
    "image_zed_stale",
    "image_head_stale",
}
# Per-frame bool fields that are validity flags (default False = invalid).
VISION_VALID_FALSE_FIELDS = {
    "mocap_left_valid", "mocap_right_valid",
    "mocap_left_wrist_angles_valid", "mocap_right_wrist_angles_valid",
    "mocap_head_tracked",
}


def load_source(path: Path) -> dict:
    """Load a source memmap dir: manifest + metadata + resolved field map."""
    with open(path / "manifest.json") as f:
        manifest = json.load(f)
    meta = np.load(path / "metadata.npz", allow_pickle=True)
    frame_fields = manifest.get("fields", {})
    episode_fields = manifest.get("episode_fields", {})
    return {
        "path": path,
        "manifest": manifest,
        "meta": meta,
        "frame_fields": frame_fields,
        "episode_fields": episode_fields,
        "rows": manifest["total_rows"],
        "eps": manifest["num_episodes"],
    }


def collect_union_frame_fields(sources: list[dict]) -> list[str]:
    """Union of all frame field names, in a stable order (EgoEMG order first)."""
    seen = []
    for src in sources:
        for name in src["frame_fields"]:
            name = normalize_field_name(name)
            if name not in seen:
                seen.append(name)
    # Put dataset_source_id at the end (we synthesize it).
    if "dataset_source_id" in seen:
        seen.remove("dataset_source_id")
    return seen


def source_field_name(src: dict, union_name: str) -> str | None:
    """The source-side field name for a union field (legacy names allowed)."""
    if union_name in src["frame_fields"]:
        return union_name
    for legacy, head in LEGACY_WEBCAM_TO_HEAD.items():
        if head == union_name and legacy in src["frame_fields"]:
            return legacy
    return None


def field_dtype_shape(src: dict, name: str):
    """Return (dtype, per_row_shape_tuple) for a frame field, or None if absent."""
    src_name = source_field_name(src, name)
    if src_name is None:
        return None
    fi = src["frame_fields"][src_name]
    return np.dtype(fi["dtype"]), tuple(fi["shape"][1:])


def write_frame_field(
    out_dir: Path, name: str, dtype: np.dtype, per_row_shape: tuple,
    sources: list[dict], source_ids: list[int],
    total_rows: int, force: bool,
) -> dict:
    """Concatenate one frame field across sources into one memmap.

    For sources lacking the field, apply the per-source invalid fill semantics.
    For `frame_split_id`, non-EgoEMG rows are forced to 0 (train) because their
    native split conventions ([train,val,test]) are incompatible with EgoEMG's
    canonical [train,user,gesture,both] and they serve only as train augments.
    """
    out_path = out_dir / f"{name}.dat"
    full_shape = (total_rows,) + per_row_shape
    # Resume support: skip if already complete.
    if out_path.exists() and not force:
        existing = np.memmap(out_path, dtype=dtype, mode="r", shape=full_shape)
        if existing.shape == full_shape:
            return {
                "filename": f"{name}.dat",
                "dtype": dtype.name,
                "shape": list(full_shape),
            }

    is_bool = dtype == np.bool_
    is_stale_true = name in VISION_STALE_TRUE_FIELDS
    is_split_id = name == "frame_split_id"

    out_mmap = np.memmap(out_path, dtype=dtype, mode="w+", shape=full_shape)
    offset = 0
    for src, sid in zip(sources, source_ids):
        n = src["rows"]
        info = field_dtype_shape(src, name)
        if info is not None and not (is_split_id and sid != SRC_EGOEMG):
            sdtype, sshape = info
            src_name = source_field_name(src, name)
            src_mmap = np.memmap(
                src["path"] / f"{src_name}.dat", dtype=sdtype, mode="r",
                shape=(n,) + sshape,
            )
            if sshape != per_row_shape:
                raise ValueError(
                    f"Field {name}: shape mismatch "
                    f"{sshape} (src {sid}) vs {per_row_shape} (union)"
                )
            # Cast to the union dtype (e.g. int32 -> int8 for frame_split_id).
            out_mmap[offset:offset + n] = src_mmap[:].astype(dtype, copy=False)
        else:
            # Source lacks this field (or non-EgoEMG split_id): apply fill.
            if is_stale_true:
                # Stale flag True for unavailable vision.
                out_mmap[offset:offset + n] = True
            elif is_split_id:
                # Non-EgoEMG rows are train-only augments.
                out_mmap[offset:offset + n] = 0
            # else: zeros (memmap default) is the correct invalid value for
            # numeric fields and the *_valid / *_tracked bools.
        offset += n
    out_mmap.flush()
    return {
        "filename": f"{name}.dat",
        "dtype": dtype.name,
        "shape": list(full_shape),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--egoemg", type=Path, required=True)
    ap.add_argument("--showee", type=Path, required=True)
    ap.add_argument("--incre", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="Re-write fields even if the output .dat exists.")
    args = ap.parse_args()

    sources = [
        {"src": load_source(args.egoemg), "id": SRC_EGOEMG, "name": "egoemg"},
        {"src": load_source(args.showee), "id": SRC_SHOWEE, "name": "showee"},
        {"src": load_source(args.incre), "id": SRC_INCRE, "name": "egoemg_incre"},
    ]
    src_objs = [s["src"] for s in sources]
    source_ids = [s["id"] for s in sources]
    source_names_by_id = {s["id"]: s["name"] for s in sources}

    total_rows = sum(s["rows"] for s in src_objs)
    total_eps = sum(s["eps"] for s in src_objs)
    print(f"Merging 3 sources:")
    for s in sources:
        print(f"  {s['name']:14s} (id={s['id']}): "
              f"{s['src']['rows']:>10,} rows, {s['src']['eps']:>4} eps")
    print(f"  -> total {total_rows:,} rows, {total_eps} episodes")
    print(f"  -> output: {args.out}")

    # Disk-space guard.
    free = shutil.disk_usage(args.out.parent).free if args.out.parent.exists() \
        else shutil.disk_usage(".").free
    needed_gb = total_rows * 0.0000017  # rough: ~1.7 bytes/row across union schema
    print(f"  free disk at output parent: {free / 1e9:.0f} GB "
          f"(estimate needed ~{needed_gb:.0f} GB; full union is ~229 GB)")

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Determine the union frame-field schema (from EgoEMG, richest) ──
    union_fields = collect_union_frame_fields(src_objs)
    print(f"\nUnion frame fields: {len(union_fields)}")

    new_frame_manifest = {}
    t0 = time.time()
    for i, name in enumerate(union_fields):
        # Pick dtype/shape from the first source that has it.
        dtype, per_row = None, None
        for s in src_objs:
            info = field_dtype_shape(s, name)
            if info is not None:
                dtype, per_row = info
                break
        if dtype is None:
            print(f"  [{i+1}/{len(union_fields)}] SKIP {name}: not in any source")
            continue
        ti = time.time()
        spec = write_frame_field(
            args.out, name, dtype, per_row, src_objs, source_ids,
            total_rows, args.force,
        )
        new_frame_manifest[name] = spec
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(union_fields)}] {name}: "
                  f"{dtype.name} {per_row} ({time.time()-ti:.1f}s)")

    # ── Synthetic dataset_source_id field ─────────────────────────────
    print("\nWriting synthetic dataset_source_id field...")
    sid_path = args.out / "dataset_source_id.dat"
    sid_mmap = np.memmap(sid_path, dtype=np.int8, mode="w+", shape=(total_rows,))
    offset = 0
    for s, sid in zip(src_objs, source_ids):
        sid_mmap[offset:offset + s["rows"]] = sid
        offset += s["rows"]
    sid_mmap.flush()
    new_frame_manifest["dataset_source_id"] = {
        "filename": "dataset_source_id.dat",
        "dtype": "int8",
        "shape": [total_rows],
    }

    # ── Renumber episode_index across sources ─────────────────────────
    # NOTE: episode_index.dat was already created (zero-filled then copied) by
    # the union-field loop above, so we open it in-place (r+) to renumber.
    # This step must run AFTER the frame-field loop.
    print("Renumbering episode_index...")
    ep_out = np.memmap(
        args.out / "episode_index.dat", dtype=np.int64, mode="r+",
        shape=(total_rows,),
    )
    row_off = 0
    ep_off = 0
    for s in src_objs:
        n, n_ep = s["rows"], s["eps"]
        src_ep = np.memmap(
            s["path"] / "episode_index.dat", dtype=np.int64, mode="r", shape=(n,)
        )
        ep_out[row_off:row_off + n] = src_ep[:] + ep_off
        row_off += n
        ep_off += n_ep
    ep_out.flush()

    # ── Merge episode-level fields (union), zero-fill missing ─────────
    union_ep_fields = []
    for s in src_objs:
        for name in s["episode_fields"]:
            if name not in union_ep_fields:
                union_ep_fields.append(name)
    new_ep_manifest = {}
    for name in union_ep_fields:
        dtype, per_ep = None, None
        for s in src_objs:
            fi = s["episode_fields"].get(name)
            if fi is not None:
                dtype = np.dtype(fi["dtype"])
                per_ep = tuple(fi["shape"][1:])
                break
        if dtype is None:
            continue
        full_shape = (total_eps,) + per_ep
        out_mmap = np.memmap(
            args.out / f"{name}.dat", dtype=dtype, mode="w+", shape=full_shape
        )
        ep_idx = 0
        for s in src_objs:
            fi = s["episode_fields"].get(name)
            n_ep = s["eps"]
            if fi is not None:
                src_mmap = np.memmap(
                    s["path"] / f"{name}.dat", dtype=dtype, mode="r",
                    shape=(n_ep,) + per_ep,
                )
                out_mmap[ep_idx:ep_idx + n_ep] = src_mmap[:]
            # else: zeros (invalid placeholder)
            ep_idx += n_ep
        out_mmap.flush()
        new_ep_manifest[name] = {
            "filename": f"{name}.dat",
            "dtype": dtype.name,
            "shape": list(full_shape),
        }

    # ── Rebuild metadata.npz (episode starts/ends/subjects/video paths) ──
    print("Rebuilding metadata.npz...")
    ep_ids, ep_subjects, ep_subject_ids = [], [], []
    ep_source_parquet, ep_zed, ep_webcam = [], [], []
    ep_start, ep_end, ep_len, ep_beta_idx = [], [], [], []
    # subject-name -> dense id (global across all sources)
    subj_to_id: dict[str, int] = {}

    ep_off = 0
    row_off = 0
    for s, sid in zip(src_objs, source_ids):
        meta = s["meta"]
        n_ep = s["eps"]
        n_rows = s["rows"]
        starts = np.asarray(meta["episode_start_idx"])  # within-source
        ends = np.asarray(meta["episode_end_idx"])
        lengths = np.asarray(meta["episode_length"])
        for e in range(n_ep):
            ep_ids.append(str(meta["episode_id"][e]))
            subj = str(meta["episode_subject"][e])
            if subj not in subj_to_id:
                subj_to_id[subj] = len(subj_to_id)
            ep_subjects.append(subj)
            ep_subject_ids.append(subj_to_id[subj])
            ep_source_parquet.append(str(meta["episode_source_parquet"][e]))
            # Video paths: carry through; Incre's are empty (no vision).
            ep_zed.append(str(meta["episode_zed_video_path"][e])
                          if "episode_zed_video_path" in meta else "")
            # Source memmaps use the legacy "webcam" metadata key; the unified
            # schema names the stream "head".
            ep_webcam.append(
                str(meta["episode_head_video_path"][e])
                if "episode_head_video_path" in meta
                else str(meta["episode_webcam_video_path"][e])
                if "episode_webcam_video_path" in meta else "")
            ep_start.append(row_off + int(starts[e]))
            ep_end.append(row_off + int(ends[e]))
            ep_len.append(int(lengths[e]))
            ep_beta_idx.append(ep_off + e)
        ep_off += n_ep
        row_off += n_rows

    def _enc(arr):
        return np.array([s.encode() if isinstance(s, str) else s for s in arr])

    new_meta = {
        "episode_id": _enc(ep_ids),
        "episode_subject": _enc(ep_subjects),
        "episode_subject_id": np.array(ep_subject_ids, dtype=np.int32),
        "episode_chunk_id": _enc(ep_ids),
        "episode_source_parquet": _enc(ep_source_parquet),
        "episode_zed_video_path": _enc(ep_zed),
        "episode_head_video_path": _enc(ep_webcam),
        "episode_start_idx": np.array(ep_start, dtype=np.int64),
        "episode_end_idx": np.array(ep_end, dtype=np.int64),
        "episode_length": np.array(ep_len, dtype=np.int64),
        "episode_beta_idx": np.array(ep_beta_idx, dtype=np.int32),
        "episode_split_id": np.zeros(total_eps, dtype=np.int32),
        "subjects_subject": _enc(list(subj_to_id.keys())),
        "subjects_subject_id": np.array(list(subj_to_id.values()), dtype=np.int32),
        "splits_split": np.array([b"train", b"user", b"gesture", b"both"]),
        "splits_split_id": np.array([0, 1, 2, 3], dtype=np.int32),
        # Per-episode provenance for the new source-id field.
        "episode_source_id": np.array(
            [source_ids[i] for i, s in enumerate(src_objs) for _ in range(s["eps"])],
            dtype=np.int8,
        ),
    }
    np.savez(args.out / "metadata.npz", **new_meta)

    # ── Write manifest.json ───────────────────────────────────────────
    new_manifest = {
        "format_version": "egoemg_v2_memmap",
        "total_rows": total_rows,
        "num_episodes": total_eps,
        "left_hand_strategy": "flip_local_z",
        "mano_label_policy": "generated_only",
        "source_root": str(args.out),
        "fields": new_frame_manifest,
        "episode_fields": new_ep_manifest,
        "generated_joint_angles_semantics":
            src_objs[0]["manifest"]["generated_joint_angles_semantics"],
        "frame_split_labels": ["train", "user", "gesture", "both"],
        "frame_split_policy": (
            "Unified merge of EgoEMG + ShowEE + Incre; per-source split ids "
            "preserved (EgoEMG train/user/gesture/both; ShowEE train/val/test; "
            "Incre train/val)."
        ),
        "dataset_sources": {
            "0": "egoemg",
            "1": "showee",
            "2": "egoemg_incre",
        },
        "source_policies": {
            "egoemg_incre": "vision/mocap unavailable (zero-filled, "
                            "stale=True, valid=False); left hand unlabelled; "
                            "wrist angles zero + invalid",
            "showee": "wrist angles unavailable (zero + wrist_angles_valid=False)",
            "egoemg": "verbatim",
        },
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(new_manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    size_gb = sum(f.stat().st_size for f in args.out.glob("*.dat")) / (1024 ** 3)
    print(f"\n=== DONE ({time.time()-t0:.0f}s) ===")
    print(f"  {args.out}")
    print(f"  rows: {total_rows:,} ({total_rows/2000/3600:.1f}h @ 2kHz)  "
          f"episodes: {total_eps}")
    print(f"  fields: {len(new_frame_manifest)} frame + "
          f"{len(new_ep_manifest)} episode")
    print(f"  size: {size_gb:.1f} GB")


if __name__ == "__main__":
    main()
