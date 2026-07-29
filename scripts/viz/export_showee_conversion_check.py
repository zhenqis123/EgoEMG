#!/usr/bin/env python3
"""Export synchronized webcam, MANO GLB, and UmeTrack GLB check samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
import trimesh

from emg2pose.realtime_local.mesh_visualizer import UmeTrackMeshForwarder
from scripts.mano.infer_mano_for_egoemg import load_mano_layer


def open_field(root: Path, manifest: dict, name: str) -> np.memmap:
    info = manifest["fields"][name]
    return np.memmap(
        root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"])
    )


def open_episode_field(root: Path, manifest: dict, name: str) -> np.memmap:
    info = manifest["episode_fields"][name]
    return np.memmap(
        root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"])
    )


def normalized_mesh(vertices: np.ndarray, faces: np.ndarray, hand: str, color) -> trimesh.Trimesh:
    vertices = np.asarray(vertices, dtype=np.float32).copy()
    vertices -= vertices.mean(axis=0, keepdims=True)
    span = float(np.median(vertices.max(axis=0) - vertices.min(axis=0)))
    if span > 1e-8:
        vertices *= 0.09 / span
    vertices[:, 0] += -0.07 if hand == "left" else 0.07
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = color
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    root = args.memmap_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = np.load(root / "metadata.npz", allow_pickle=False)
    starts, ends = metadata["episode_start_idx"], metadata["episode_end_idx"]
    sources = np.char.decode(metadata["episode_source_parquet"])
    videos = np.char.decode(metadata["episode_webcam_video_path"])
    episode_ids = np.char.decode(metadata["episode_id"])
    valid = open_field(root, manifest, "generated_label_valid")
    stale = open_field(root, manifest, "image_webcam_stale")
    video_indices = open_field(root, manifest, "image_webcam_frame_index")
    timestamps = open_field(root, manifest, "timestamp_us")
    poses = {h: open_field(root, manifest, f"generated_mano_{h}_pose") for h in ("left", "right")}
    angles = {h: open_field(root, manifest, f"generated_joint_angles_{h}") for h in ("left", "right")}
    betas = {h: open_episode_field(root, manifest, f"generated_mano_{h}_beta") for h in ("left", "right")}

    rng = np.random.default_rng(args.seed)
    episode_order = rng.permutation(len(starts))
    rows: list[tuple[int, int]] = []
    for episode in episode_order:
        if not videos[episode] or not Path(videos[episode]).exists():
            continue
        candidates = np.arange(int(starts[episode]), int(ends[episode]), dtype=np.int64)
        candidates = candidates[
            valid[candidates].all(axis=1)
            & ~np.asarray(stale[candidates], dtype=bool)
            & (np.asarray(video_indices[candidates]) >= 0)
        ]
        if len(candidates) == 0:
            continue
        rows.append((int(episode), int(rng.choice(candidates))))
        if len(rows) == args.num_samples:
            break

    mano = load_mano_layer(torch.device(args.device))
    umetrack_forwarder = UmeTrackMeshForwarder()
    mano_faces = mano.th_faces.detach().cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_number, (episode, row) in enumerate(rows, 1):
        sample_dir = args.output_dir / f"sample_{sample_number:02d}"
        sample_dir.mkdir(exist_ok=True)
        video_frame = int(video_indices[row])
        reader = decord.VideoReader(videos[episode], ctx=decord.cpu(0))
        video_frame = min(max(video_frame, 0), len(reader) - 1)
        image_rgb = reader[video_frame].asnumpy()
        cv2.imwrite(
            str(sample_dir / "webcam.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        )

        mano_scene, umetrack_scene = trimesh.Scene(), trimesh.Scene()
        for hand in ("left", "right"):
            pose_t = torch.from_numpy(np.asarray(poses[hand][row]).copy())[None].to(args.device)
            beta_t = torch.from_numpy(np.asarray(betas[hand][episode]).copy())[None].to(args.device)
            with torch.no_grad():
                mano_vertices = mano(pose_t, beta_t).verts[0].cpu().numpy()
            faces = mano_faces
            if hand == "left":
                mano_vertices[:, 2] *= -1
                faces = faces[:, [0, 2, 1]]
            mano_scene.add_geometry(
                normalized_mesh(
                    mano_vertices,
                    faces,
                    hand,
                    (40, 150, 245, 255) if hand == "left" else (60, 220, 100, 255),
                ),
                node_name=f"mano_{hand}",
            )

            ut_mesh = umetrack_forwarder(np.asarray(angles[hand][row]))
            ut_vertices, ut_faces = ut_mesh.vertices.copy(), ut_mesh.triangles.copy()
            if hand == "left":
                ut_vertices[:, 0] *= -1
                ut_faces = ut_faces[:, [0, 2, 1]]
            umetrack_scene.add_geometry(
                normalized_mesh(
                    ut_vertices,
                    ut_faces,
                    hand,
                    (40, 150, 245, 255) if hand == "left" else (60, 220, 100, 255),
                ),
                node_name=f"umetrack_{hand}",
            )
        mano_scene.export(sample_dir / "mano.glb")
        umetrack_scene.export(sample_dir / "umetrack.glb")
        record = {
            "sample": sample_number,
            "directory": sample_dir.name,
            "episode_index": episode,
            "episode_id": episode_ids[episode],
            "source": sources[episode],
            "global_row": row,
            "timestamp_us": int(timestamps[row]),
            "video_path": videos[episode],
            "video_frame_index": video_frame,
            "files": {"image": "webcam.png", "mano": "mano.glb", "umetrack": "umetrack.glb"},
        }
        (sample_dir / "sample.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        print(f"[{sample_number}/{len(rows)}] {sources[episode]} row={row} video={video_frame}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "selection": "one random synchronized valid row from distinct episodes",
                "mesh_layout": "left and right shown side-by-side in canonical display space",
                "colors": {"left": "blue", "right": "green"},
                "samples": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
