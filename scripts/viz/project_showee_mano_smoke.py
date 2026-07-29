#!/usr/bin/env python3
"""Smoke-test projection of ShowEE MANO meshes using raw head/cam poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from markers2mano.rigid_align import compute_rigid_transform
from scripts.mano.infer_mano_for_egoemg import MARKER_VERT_INDICES, load_mano_layer


def open_field(root: Path, manifest: dict, name: str, episode: bool = False):
    group = "episode_fields" if episode else "fields"
    info = manifest[group][name]
    return np.memmap(root / info["filename"], mode="r", dtype=info["dtype"], shape=tuple(info["shape"]))


def draw_mesh(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    color,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    fill = image.copy()
    depth = vertices[:, 2]
    uv = cv2.projectPoints(
        vertices.astype(np.float64),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )[0].reshape(-1, 2).astype(np.float32)
    face_depth = depth[faces].mean(axis=1)
    for face_index in np.argsort(face_depth)[::-1]:
        face = faces[face_index]
        if np.any(depth[face] <= 0.02):
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if np.any(np.abs(polygon) > 10000):
            continue
        cv2.fillConvexPoly(fill, polygon, color, lineType=cv2.LINE_AA)
    output = cv2.addWeighted(fill, 0.28, output, 0.72, 0)
    for face in faces:
        if np.any(depth[face] <= 0.02):
            continue
        polygon = np.rint(uv[face]).astype(np.int32)
        if np.any(np.abs(polygon) > 10000):
            continue
        cv2.polylines(output, [polygon], True, color, 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--sync-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    root = args.memmap_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = np.load(root / "metadata.npz", allow_pickle=False)
    sync = json.loads(args.sync_manifest.read_text())
    starts, ends = metadata["episode_start_idx"], metadata["episode_end_idx"]
    sources = np.char.decode(metadata["episode_source_parquet"])
    poses = {h: open_field(root, manifest, f"generated_mano_{h}_pose") for h in ("left", "right")}
    betas = {h: open_field(root, manifest, f"generated_mano_{h}_beta", True) for h in ("left", "right")}
    keypoints = {h: open_field(root, manifest, f"mocap_{h}_keypoints") for h in ("left", "right")}
    mano = load_mano_layer(torch.device(args.device))
    faces_right = mano.th_faces.detach().cpu().numpy()
    marker_indices = MARKER_VERT_INDICES.to(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for sample in sync["samples"]:
        episode, row = int(sample["episode_index"]), int(sample["global_row"])
        sample_dir = args.output_dir / sample["directory"]
        sample_dir.mkdir(exist_ok=True)
        image = cv2.imread(str(args.sync_manifest.parent / sample["directory"] / "webcam.png"))
        mocap_path = Path("data/showee") / sources[episode] / "luster_mocap/mocap.h5"
        task_metadata = json.loads(
            (Path("data/showee") / sources[episode] / "metadata.json").read_text()
        )
        head_source = next(
            source
            for source in task_metadata["sources"]
            if source["source_id"] == "showee_head"
        )
        calibration = head_source["camera_calibration"]
        camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(calibration["distortion"], dtype=np.float64)
        with h5py.File(mocap_path, "r") as handle:
            fraction = (row - int(starts[episode])) / max(int(ends[episode] - starts[episode] - 1), 1)
            mocap_index = int(round(fraction * (len(handle["timestamp"]) - 1)))
            camera_position = handle["head/cam_position"][mocap_index].astype(np.float64) / 1000.0
            camera_quaternion = handle["head/cam_quaternion"][mocap_index].astype(np.float64)
        rotation_world_camera = Rotation.from_quat(camera_quaternion).as_matrix()
        scene = trimesh.Scene()
        overlay = image.copy()
        hand_report = {}
        for hand in ("left", "right"):
            pose = torch.from_numpy(np.asarray(poses[hand][row]).copy())[None].to(args.device)
            beta = torch.from_numpy(np.asarray(betas[hand][episode]).copy())[None].to(args.device)
            with torch.no_grad():
                vertices = mano(pose, beta).verts[0]
            faces = faces_right
            if hand == "left":
                vertices = vertices * vertices.new_tensor([1.0, 1.0, -1.0])
                faces = faces[:, [0, 2, 1]]
            gt_world = torch.from_numpy(np.asarray(keypoints[hand][row]).copy()).to(args.device)
            predicted_markers = vertices[marker_indices]
            rigid_rotation, rigid_translation = compute_rigid_transform(predicted_markers, gt_world)
            vertices_world = vertices @ rigid_rotation.T + rigid_translation
            vertices_camera = (
                (vertices_world.cpu().numpy() - camera_position) @ rotation_world_camera
            )
            overlay = draw_mesh(
                overlay,
                vertices_camera,
                faces,
                (245, 140, 40) if hand == "left" else (40, 220, 80),
                camera_matrix,
                distortion,
            )
            mesh = trimesh.Trimesh(vertices=vertices_camera, faces=faces, process=False)
            mesh.visual.face_colors = (40, 140, 245, 220) if hand == "left" else (60, 220, 100, 220)
            scene.add_geometry(mesh, node_name=f"mano_{hand}_camera")
            hand_report[hand] = {
                "median_depth_m": float(np.median(vertices_camera[:, 2])),
                "all_vertices_in_front": bool(np.all(vertices_camera[:, 2] > 0)),
            }
        cv2.putText(overlay, "calibrated K + distortion from metadata.json", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(sample_dir / "mano_overlay_calibrated.png"), overlay)
        scene.export(sample_dir / "mano_camera.glb")
        record = {
            **sample,
            "mocap_index": mocap_index,
            "camera_matrix": camera_matrix.tolist(),
            "distortion": distortion.tolist(),
            "hands": hand_report,
        }
        (sample_dir / "projection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        report.append(record)
        print(sample["directory"], "mocap", mocap_index, hand_report)
    (args.output_dir / "report.json").write_text(
        json.dumps({"intrinsics_source": "per-task metadata.json showee_head.camera_calibration", "samples": report}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
