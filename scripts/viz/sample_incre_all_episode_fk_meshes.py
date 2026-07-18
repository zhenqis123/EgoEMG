#!/usr/bin/env python3
"""Randomly export FK mesh GLBs from every episode in an incre memmap dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VIZ_SPEC = importlib.util.spec_from_file_location(
    "emg2pose_viz_mod",
    str(_PROJECT_ROOT / "emg2pose" / "visualization.py"),
)
if _VIZ_SPEC is None or _VIZ_SPEC.loader is None:
    raise RuntimeError("Could not load emg2pose/visualization.py")
_VIZ_MOD = importlib.util.module_from_spec(_VIZ_SPEC)
_VIZ_SPEC.loader.exec_module(_VIZ_MOD)
skin_mesh_from_angles = _VIZ_MOD.skin_mesh_from_angles


def _open_memmap(memmap_dir: Path, field: dict, mode: str = "r") -> np.memmap:
    return np.memmap(
        memmap_dir / field["filename"],
        dtype=np.dtype(field["dtype"]),
        mode=mode,
        shape=tuple(field["shape"]),
    )


def _decode(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _episode_slice(metadata: np.lib.npyio.NpzFile, episode_idx: int) -> tuple[int, int]:
    start = int(metadata["episode_start_idx"][episode_idx])
    if "episode_length" in metadata.files:
        stop = start + int(metadata["episode_length"][episode_idx])
    else:
        stop = int(metadata["episode_end_idx"][episode_idx])
    return start, stop


def _split_ids(metadata: np.lib.npyio.NpzFile) -> dict[str, int]:
    if "splits_split" not in metadata.files:
        return {"train": 0, "val": 1, "test": 2}
    return {_decode(name): idx for idx, name in enumerate(metadata["splits_split"])}


def _parse_episode_aliases(values: list[str] | None) -> dict[int, str]:
    aliases: dict[int, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid alias '{value}', expected IDX=NAME")
        idx_text, name = value.split("=", 1)
        aliases[int(idx_text)] = name
    return aliases


def _sample_candidates(
    *,
    episode_idx: int,
    start: int,
    stop: int,
    joint_angles: np.memmap,
    label_valid: np.memmap | None,
    frame_split: np.memmap | None,
    split_id: int | None,
    hand_idx: int,
) -> np.ndarray:
    candidates = np.arange(start, stop, dtype=np.int64)
    if frame_split is not None and split_id is not None:
        candidates = candidates[np.asarray(frame_split[candidates]) == split_id]
    if label_valid is not None:
        candidates = candidates[np.asarray(label_valid[candidates, hand_idx], dtype=bool)]

    if candidates.size == 0:
        return candidates

    # Drop non-finite or zero rows without materializing the whole episode.
    keep = []
    chunk = 200_000
    for offset in range(0, candidates.size, chunk):
        idx = candidates[offset : offset + chunk]
        angles = np.asarray(joint_angles[idx], dtype=np.float32)
        valid = np.isfinite(angles).all(axis=1) & (np.abs(angles).sum(axis=1) > 1e-6)
        keep.append(valid)
    keep_mask = np.concatenate(keep)
    return candidates[keep_mask]


def _export_mesh(path: Path, joint_angles: np.ndarray, color: list[int]) -> None:
    joint_angles = np.asarray(joint_angles[:20], dtype=np.float32).copy()
    vertices, faces = skin_mesh_from_angles(joint_angles=joint_angles, flip=False)
    vertices = vertices.copy()
    span = np.median(vertices.max(axis=0) - vertices.min(axis=0))
    if span > 1e-6:
        vertices *= 0.09 / span
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = color
    mesh.export(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=Path("data/EgoEMG_incre/data_right_merged"),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/incre_all_episode_fk_meshes"))
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument(
        "--split",
        default="any",
        choices=["any", "train", "val", "test"],
        help="Optional frame_split_id filter.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write all GLBs into one directory instead of episode subdirectories.",
    )
    parser.add_argument(
        "--episode-name-alias",
        action="append",
        default=[],
        help="Override output episode name, format IDX=NAME. Can be repeated.",
    )
    args = parser.parse_args()

    memmap_dir = args.memmap_dir
    with open(memmap_dir / "manifest.json") as f:
        manifest = json.load(f)
    metadata = np.load(memmap_dir / "metadata.npz", allow_pickle=True)

    hand_idx = 1 if args.hand == "right" else 0
    joint_key = f"generated_joint_angles_{args.hand}"
    joint_angles = _open_memmap(memmap_dir, manifest["fields"][joint_key])
    label_valid = (
        _open_memmap(memmap_dir, manifest["fields"]["generated_label_valid"])
        if "generated_label_valid" in manifest["fields"]
        else None
    )
    frame_split = (
        _open_memmap(memmap_dir, manifest["fields"]["frame_split_id"])
        if "frame_split_id" in manifest["fields"]
        else None
    )
    split_id = None
    if args.split != "any":
        split_id = _split_ids(metadata).get(args.split)
        if split_id is None:
            raise ValueError(f"Unknown split '{args.split}' in metadata")

    episode_ids = [_decode(value) for value in metadata["episode_id"]]
    episode_aliases = _parse_episode_aliases(args.episode_name_alias)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "memmap_dir": str(memmap_dir),
        "output": str(args.output),
        "hand": args.hand,
        "split": args.split,
        "n_samples_requested": args.n_samples,
        "seed": args.seed,
        "episodes": [],
    }

    for episode_idx, episode_id in enumerate(episode_ids):
        output_episode_id = episode_aliases.get(episode_idx, episode_id)
        start, stop = _episode_slice(metadata, episode_idx)
        candidates = _sample_candidates(
            episode_idx=episode_idx,
            start=start,
            stop=stop,
            joint_angles=joint_angles,
            label_valid=label_valid,
            frame_split=frame_split,
            split_id=split_id,
            hand_idx=hand_idx,
        )
        n = min(args.n_samples, int(candidates.size))
        if n == 0:
            print(f"episode {episode_idx} {episode_id}: no candidates")
            summary["episodes"].append(
                {
                    "episode_idx": episode_idx,
                    "episode_id": output_episode_id,
                    "memmap_episode_id": episode_id,
                    "start": start,
                    "stop": stop,
                    "num_candidates": int(candidates.size),
                    "sampled": [],
                }
            )
            continue

        sampled = np.sort(rng.choice(candidates, size=n, replace=False))
        out_dir = (
            args.output
            if args.flat
            else args.output / f"ep{episode_idx:02d}_{output_episode_id}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        sampled_items = []
        for sample_idx, global_idx in enumerate(sampled.tolist()):
            angles = np.asarray(joint_angles[global_idx], dtype=np.float32)
            local_idx = global_idx - start
            path = out_dir / (
                f"fk_ep{episode_idx:02d}_{output_episode_id}_"
                f"global{global_idx:08d}_local{local_idx:08d}.glb"
            )
            _export_mesh(path, angles, [255, 180, 0, 255])
            sampled_items.append(
                {
                    "sample_idx": sample_idx,
                    "global_idx": int(global_idx),
                    "local_idx": int(local_idx),
                    "glb": str(path),
                }
            )
            print(f"episode {episode_idx} {episode_id}: {global_idx} -> {path}")

        summary["episodes"].append(
            {
                "episode_idx": episode_idx,
                "episode_id": output_episode_id,
                "memmap_episode_id": episode_id,
                "start": start,
                "stop": stop,
                "num_candidates": int(candidates.size),
                "sampled": sampled_items,
            }
        )

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Summary: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
