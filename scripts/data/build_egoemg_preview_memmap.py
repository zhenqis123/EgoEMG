#!/usr/bin/env python3
"""Build a small self-contained preview shard from the unified EgoEMG memmap.

This slices whole episodes out of the merged ``egoemg_v3_memmap`` and lays them
down in the familiar release-package layout, so the README scripts run
end-to-end on a small download instead of the full dataset:

    <out>/data/memmap_data/                     v3 memmap (grouped *.dat + manifest + metadata)
    <out>/data/webcam_videos/                   episode_XXXXXX_head_allintra.mp4  (symlink/copy)
    <out>/data/pre-crop_webcam_videoframes/     episode_XXXXXX.lmdb + manifest.json + .done
    <out>/data/<calibration json>               copied calibration

The default episodes span three splits and three subjects:

- episode_000020 (wmh)  — train + user + gesture + both  (best all-rounder)
- episode_000008 (zbk)  — user + both
- episode_000028 (wsj)  — train + gesture  (the reporter's known episode)

Each EgoEMG episode is one recording, so the shard renumbers ``episode_index``
to 0..K-1 and rebuilds ``is_first``/``is_last`` at the episode boundaries, which
keeps the validator's "is_first count == beta rows" invariant for the K rows of
``generated_mano_{left,right}_beta`` it carries.

Usage:
    python scripts/data/build_egoemg_preview_memmap.py --out ./data/egoemg_preview
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


DEFAULT_EPISODES = ["episode_000020", "episode_000008", "episode_000028"]

FRAME_GROUPS = {
    "core", "emg", "imu", "labels", "mano", "mocap_hands",
    "mocap_head", "mocap_wrist", "vision",
}
# Fields recomputed for the shard rather than copied verbatim.
RECOMPUTE_EPISODE_INDEX = "episode_index"
RECOMPUTE_FIRST = "is_first"
RECOMPUTE_LAST = "is_last"


def _resolve_episode_rows(meta: dict, episode_ids: list[str]) -> dict:
    """Map each requested episode id to its source (start, end, subject, ...)."""
    ids = [e.decode() if isinstance(e, bytes) else str(e) for e in meta["episode_id"]]
    rows: dict[str, dict] = {}
    for eid in episode_ids:
        if eid not in ids:
            raise SystemExit(f"Unknown episode id {eid!r}; choose from: {ids[:5]} …")
        i = ids.index(eid)
        rows[eid] = {
            "row": i,
            "start": int(meta["episode_start_idx"][i]),
            "end": int(meta["episode_end_idx"][i]),
            "length": int(meta["episode_length"][i]),
            "beta_idx": int(meta["episode_beta_idx"][i]),
            "subject": meta["episode_subject"][i].decode(),
            "subject_id": int(meta["episode_subject_id"][i]),
            "head_video": meta["episode_head_video_path"][i].decode(),
        }
    return rows


def _copy_frame_field(
    out_root: Path,
    name: str,
    spec: dict,
    src_memmap_dir: Path,
    ranges: list[tuple[int, int]],
    offsets: list[int],
    total_rows: int,
) -> dict:
    """Copy one frame field's rows into the shard, or recompute it."""
    src_rel = spec["filename"]
    group = src_rel.split("/", 1)[0]
    if group not in FRAME_GROUPS:
        raise RuntimeError(f"Unexpected field group {group!r} for {name}")
    # Flat output layout, matching the published `dataset_egoemg_unified`
    # (and `merge_datasets_to_unified_memmap.py`). The source is grouped but we
    # read it by its own path and write the shard flat.
    out_rel = f"{name}.dat"
    out_path = out_root / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_row_shape = tuple(spec["shape"][1:])
    dtype = np.dtype(spec["dtype"])
    full_shape = (total_rows,) + per_row_shape
    out_mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=full_shape)

    if name == RECOMPUTE_EPISODE_INDEX:
        # Renumber to contiguous 0..K-1 across the concatenated ranges.
        for k, (off, (s, e)) in enumerate(zip(offsets, ranges)):
            out_mm[off:off + (e - s)] = k
    else:
        # Open the source with its *own* shape (full source length), then slice
        # the episode row range. The output shard shape is narrower than the
        # source, so the source view must not use `full_shape`.
        src_mm = np.memmap(src_memmap_dir / src_rel, dtype=dtype, mode="r",
                           shape=tuple(spec["shape"]))
        for (off, (s, e)) in zip(offsets, ranges):
            out_mm[off:off + (e - s)] = src_mm[s:e]
        del src_mm

    if name == RECOMPUTE_FIRST:
        out_mm[:] = False
        for (off, (s, e)) in zip(offsets, ranges):
            out_mm[off] = True
    elif name == RECOMPUTE_LAST:
        out_mm[:] = False
        for (off, (s, e)) in zip(offsets, ranges):
            out_mm[off + (e - s) - 1] = True

    out_mm.flush()
    return {"filename": out_rel, "dtype": spec["dtype"], "shape": [total_rows, *list(per_row_shape)]}


def _copy_episode_fields(
    out_root: Path,
    ep_fields: dict,
    src_memmap_dir: Path,
    beta_indices: list[int],
) -> dict:
    """Copy the per-recording beta rows referenced by the shard's episodes."""
    out: dict = {}
    for name, spec in ep_fields.items():
        src_rel = spec["filename"]
        out_rel = f"{name}.dat"  # flat layout
        per_row_shape = tuple(spec["shape"][1:])
        dtype = np.dtype(spec["dtype"])
        src_mm = np.memmap(src_memmap_dir / src_rel, dtype=dtype, mode="r", shape=tuple(spec["shape"]))
        data = np.asarray(src_mm[beta_indices])  # (K, 10)
        del src_mm
        out_path = out_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=data.shape)
        arr[:] = data
        arr.flush()
        out[name] = {
            "filename": out_rel,
            "dtype": spec["dtype"],
            "shape": [int(data.shape[0]), *list(per_row_shape)],
        }
    return out


def _write_metadata(out_root: Path, rows: dict, lengths: list[int]) -> None:
    order = list(rows.values())
    k = len(order)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    ends = np.cumsum(lengths).astype(np.int64)  # exclusive-end (matches source)
    subject_ids = np.array([o["subject_id"] for o in order], dtype=np.int32)

    out = {
        "episode_id": np.array(
            [f"episode_{o['row']:06d}".encode() for o in order], dtype="|S14"
        ),
        "episode_subject": np.array(
            [o["subject"].encode() for o in order], dtype="|S25"
        ),
        "episode_subject_id": subject_ids,
        "episode_chunk_id": np.array(
            [f"episode_{o['row']:06d}" for o in order], dtype="<U14"
        ),
        "episode_source_parquet": np.array(["" for _ in order], dtype="<U90"),
        "episode_zed_video_path": np.array(
            [f"episode_{o['row']:06d}.mp4".encode() for o in order], dtype="|S22"
        ),
        "episode_start_idx": starts,
        "episode_end_idx": ends,
        "episode_length": np.array(lengths, dtype=np.int64),
        "episode_beta_idx": np.arange(k, dtype=np.int32),
        "episode_split_id": np.zeros(k, dtype=np.int32),
        "episode_head_video_path": np.array(
            [o["head_video"].encode() for o in order], dtype="|S23"
        ),
        "episode_wrist_left_video_path": np.array([b"" for _ in order], dtype="|S29"),
        "episode_wrist_right_video_path": np.array([b"" for _ in order], dtype="|S30"),
        "subjects_subject": np.array(
            [o["subject"].encode() for o in order], dtype="|S25"
        ),
        "subjects_subject_id": subject_ids,
        "splits_split": np.array([b"train", b"user", b"gesture", b"both"], dtype="|S7"),
        "splits_split_id": np.array([0, 1, 2, 3], dtype=np.int32),
        "schema_version": np.array(["egoemg_metadata_v3"], dtype="<U18"),
    }
    np.savez(out_root / "metadata.npz", **out)


def _write_manifest(
    out_root: Path,
    src_manifest: dict,
    fields: dict,
    ep_fields: dict,
    total_rows: int,
    num_episodes: int,
) -> None:
    manifest = {
        "format_version": "egoemg_v3_memmap",
        "total_rows": total_rows,
        "num_episodes": num_episodes,
        "layout": None,  # flat layout, matching the published unified dataset
        "left_hand_strategy": src_manifest.get("left_hand_strategy"),
        "mano_label_policy": src_manifest.get("mano_label_policy"),
        "generated_joint_angles_semantics": src_manifest.get("generated_joint_angles_semantics"),
        "frame_split_labels": src_manifest.get("frame_split_labels"),
        "frame_split_policy": src_manifest.get("frame_split_policy"),
        "dataset_sources": src_manifest.get("dataset_sources"),
        "fields": fields,
        "episode_fields": ep_fields,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _install_vision_assets(
    out: Path,
    rows: dict,
    allintra_root: Path,
    crops_root: Path,
    calibration_json: Path,
    copy_videos: bool,
    copy_crops: bool,
) -> Path:
    """Link/copy all-intra head videos, crop LMDBs, and the calibration file."""
    videos_out = out / "data" / "webcam_videos"
    crops_out = out / "data" / "pre-crop_webcam_videoframes"
    videos_out.mkdir(parents=True, exist_ok=True)
    crops_out.mkdir(parents=True, exist_ok=True)

    for eid, o in rows.items():
        stem = o["head_video"].split(".")[0]  # episode_XXXXXX_head
        vid_src = allintra_root / f"{stem}_allintra.mp4"
        if not vid_src.exists():
            raise SystemExit(f"Missing all-intra video: {vid_src}")
        _link_or_copy(vid_src, videos_out / vid_src.name, copy_videos)

        lmdb_src = crops_root / f"{eid}.lmdb"
        if not lmdb_src.is_dir():
            raise SystemExit(f"Missing crop LMDB: {lmdb_src}")
        _link_or_copy(lmdb_src, crops_out / f"{eid}.lmdb", copy_crops)
        (crops_out / f"{eid}.done").write_text(f"preview release marker for {eid}\n")

    # Pre-crop manifest: visualize_dataset.py reads `patch_size` from here.
    (crops_out / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "format": "per_episode_crops",
                "patch_size": 256,
                "num_episodes": len(rows),
                "episode_ids": list(rows.keys()),
            },
            indent=2,
        ) + "\n"
    )

    # Calibration: vision mode resolves it via --calibration-json.
    _link_or_copy(calibration_json, out / "data" / calibration_json.name, copy=True)
    return out / "data" / calibration_json.name


def main() -> int:
    p = argparse.ArgumentParser(
        description="Slice whole episodes from the unified EgoEMG memmap into a preview shard."
    )
    p.add_argument(
        "--memmap-dir", type=Path,
        default=Path(
            os.environ.get("EGOEMG_UNIFIED_MEMMAP_DIR", "/mnt/nvme/xiziheng/EgoEMG_unified_memmap")
        ),
        help="Source egoemg_v3_unified memmap directory.",
    )
    p.add_argument("--out", type=Path, default=Path("data/egoemg_preview"))
    p.add_argument("--episodes", nargs="+", default=DEFAULT_EPISODES)
    p.add_argument("--allintra-root", type=Path, default=Path("/mnt/nvme/xiziheng/EgoEMG_allintra"))
    p.add_argument("--crops-root", type=Path, default=Path("/mnt/nvme/xiziheng/EgoEMG_v2_crops"))
    p.add_argument(
        "--calibration-json", type=Path,
        default=Path("/mnt/nvme/xiziheng/EgoEMG_release_staging/EgoEMG-dataset-small/meta/GX010023_standard_calibration.json"),
    )
    p.add_argument("--copy-videos", action="store_true", help="copy videos instead of symlinking")
    p.add_argument("--copy-crops", action="store_true", help="copy crop LMDBs instead of symlinking")
    p.add_argument(
        "--relax", action="store_true",
        help="note: validate_memmap.py should get --allow-partial-sources for a shard",
    )
    args = p.parse_args()

    src = args.memmap_dir
    manifest_src = json.loads((src / "manifest.json").read_text())
    meta = np.load(src / "metadata.npz", allow_pickle=False)
    rows = {eid: _resolve_episode_rows(meta, [eid])[eid] for eid in args.episodes}
    order = [rows[e] for e in args.episodes]
    lengths = [o["length"] for o in order]
    total_rows = int(sum(lengths))
    num_episodes = len(order)

    offsets = list(np.cumsum(np.array(lengths, dtype=np.int64)) - np.array(lengths, dtype=np.int64))
    ranges = [(o["start"], o["end"]) for o in order]

    memmap_out = args.out / "data" / "memmap_data"
    memmap_out.mkdir(parents=True, exist_ok=True)

    print(f"Source : {src}")
    print(f"Output : {args.out}")
    print(f"Episodes: {num_episodes} -> total_rows={total_rows:,}")
    for eid, o in zip(args.episodes, order):
        print(f"  {eid}: subject={o['subject']:>6} rows=[{o['start']:,} {o['end']:,}) "
              f"len={o['length']:,} beta={o['beta_idx']}")

    fields: dict = {}
    for name, spec in manifest_src["fields"].items():
        fields[name] = _copy_frame_field(memmap_out, name, spec, src, ranges, offsets, total_rows)
    print(f"Copied {len(fields)} frame fields to {memmap_out}")

    ep_fields = _copy_episode_fields(
        memmap_out, manifest_src["episode_fields"], src, [o["beta_idx"] for o in order]
    )
    print(f"Copied {len(ep_fields)} episode fields (beta table -> {num_episodes} rows)")

    _write_metadata(memmap_out, rows, lengths)
    _write_manifest(memmap_out, manifest_src, fields, ep_fields, total_rows, num_episodes)
    print("Wrote metadata.npz + manifest.json")

    gsrc = src / "gesture_classes.json"
    if gsrc.exists():
        shutil.copy2(gsrc, memmap_out / "gesture_classes.json")
    (memmap_out / "README.txt").write_text(
        f"Preview shard: {num_episodes} episodes, {total_rows:,} rows, "
        + ", ".join(args.episodes) + ".\n"
    )

    calib = _install_vision_assets(
        args.out, rows, args.allintra_root, args.crops_root,
        args.calibration_json, args.copy_videos, args.copy_crops,
    )

    mm = memmap_out
    print("\nDone. Ready-to-run commands:\n")
    print(
        "  # Validate a preview shard (relaxed source/split checks)\n"
        f"  python scripts/data/validate_memmap.py --memmap-dir {mm} --allow-partial-sources\n"
    )
    print(
        "  # EMG eval (middle = 8ch target_hand)\n"
        f"  python -m egoemg.test_analysis experiment=emgformer/egoemg_emgformer_middle \\\n"
        f"    'checkpoint=checkpoints/egoemg_emgformer_middle.ckpt' \\\n"
        f"    egoemg_unified_memmap_dir={mm} 'trainer.devices=[0]' \\\n"
        f"    datamodule.per_dataset_norm_stats_path=assets/per_dataset_norm_stats_unified.json\n"
    )
    print(
        "  # Vision visualization (needs MANO model path, see ASSET_SETUP)\n"
        f"  python scripts/viz/visualize_dataset.py vision \\\n"
        f"    --memmap-dir {mm} \\\n"
        f"    --allintra-root {args.out}/data/webcam_videos \\\n"
        f"    --crops-dir {args.out}/data/pre-crop_webcam_videoframes \\\n"
        f"    --data-root {args.out}/data \\\n"
        f"    --calibration-json {calib} \\\n"
        f"    --episode-id {args.episodes[-1]} --stride 10 --max-frames 300 \\\n"
        f"    --mano-model-path $WILOR_PATH/mano_data/models\n"
    )
    print(
        "  # Smoke train (1 epoch, 2 batches; small batch_size for a single GPU)\n"
        f"  python -m egoemg.train experiment=emgformer/egoemg_emgformer_small \\\n"
        f"    egoemg_unified_memmap_dir={mm} 'trainer.devices=[0]' 'trainer.max_epochs=1' batch_size=8 \\\n"
        f"    '+trainer.limit_train_batches=2' '+trainer.limit_val_batches=0' \\\n"
        f"    datamodule.per_dataset_norm_stats_path=assets/per_dataset_norm_stats_unified.json\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
