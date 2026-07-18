from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class UmeTrackMesh:
    vertices: np.ndarray
    triangles: np.ndarray


class UmeTrackMeshForwarder:
    """CPU UmeTrack mesh FK for realtime visualization."""

    def __init__(self) -> None:
        from emg2pose.kinematics import apply_to_hand_model, load_default_hand_model

        hand_model = load_default_hand_model()
        self.hand_model = apply_to_hand_model(hand_model, lambda t: t.float().unsqueeze(0))
        self.triangles = self.hand_model.mesh_triangles[0].long().cpu().numpy()
        self.wrist_transform = torch.eye(4, dtype=torch.float32).unsqueeze(0)

    @torch.inference_mode()
    def __call__(self, angles: np.ndarray) -> UmeTrackMesh:
        from emg2pose.UmeTrack.lib.common.hand_skinning import (
            _get_skinned_vertices,
            _hand_skinning_transform,
            _lbs,
        )

        angles_np = np.asarray(angles, dtype=np.float32).reshape(-1)
        if angles_np.shape[0] < 20:
            raise ValueError(f"Expected at least 20 joint angles, got {angles_np.shape[0]}")
        if angles_np.shape[0] < 22:
            angles_np = np.pad(angles_np, (0, 22 - angles_np.shape[0]))
        # Match scripts/ik/batch_ik_mesh.py: pass 22D angles into UmeTrack's
        # skinning transform. The current UmeTrack FK uses the first 20 finger
        # DOFs; the final two wrist pitch/yaw channels are supervision targets
        # but do not affect mesh skinning without an external wrist transform.
        joint_angles = torch.from_numpy(angles_np[:22]).reshape(1, 22)
        skin_xfs = _hand_skinning_transform(
            self.hand_model.joint_rotation_axes.reshape(1, -1, 3),
            self.hand_model.joint_rest_positions.reshape(1, -1, 3),
            joint_angles,
            self.wrist_transform,
        )
        weights = self.hand_model.dense_bone_weights.reshape(1, -1, 17)
        rest_vertices = self.hand_model.mesh_vertices.reshape(1, -1, 3)
        skinned_vertices = _get_skinned_vertices(rest_vertices, weights)
        vertices = _lbs(skin_xfs, skinned_vertices)[..., :3][0].cpu().numpy()
        return UmeTrackMesh(vertices=vertices.astype(np.float32), triangles=self.triangles)


def angles_to_umetrack_mesh(angles: np.ndarray) -> UmeTrackMesh:
    return UmeTrackMeshForwarder()(angles)


class RealtimeMeshVisualizer:
    """Best-effort Open3D mesh visualizer driven from the main thread."""

    def __init__(self, max_queue: int = 2, window_name: str = "EMG UmeTrack Mesh") -> None:
        try:
            import open3d as o3d
        except Exception as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError(f"Open3D import failed: {exc}") from exc

        self._o3d = o3d
        self._queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=max_queue)
        self._forwarder = UmeTrackMeshForwarder()
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(window_name, width=960, height=720):
            raise RuntimeError("Open3D failed to create a visualization window")
        self._mesh = o3d.geometry.TriangleMesh()
        self._first_frame = True
        self._closed = False

    def update(self, angles: np.ndarray, inference_ms: float = 0.0) -> None:
        payload = (np.asarray(angles, dtype=np.float32).copy(), float(inference_ms))
        while True:
            try:
                self._queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def poll(self) -> bool:
        if self._closed:
            return False

        item = _drain_latest(self._queue)
        if item is not None:
            angles, _inference_ms = item
            umetrack_mesh = self._forwarder(angles)
            self._mesh.vertices = self._o3d.utility.Vector3dVector(
                _center_for_display(umetrack_mesh.vertices)
            )
            self._mesh.triangles = self._o3d.utility.Vector3iVector(
                umetrack_mesh.triangles
            )
            self._mesh.compute_vertex_normals()
            self._mesh.paint_uniform_color([0.90, 0.48, 0.25])

            if self._first_frame:
                self._vis.add_geometry(self._mesh)
                frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                self._vis.add_geometry(frame)
                ctr = self._vis.get_view_control()
                ctr.set_front([0.0, -0.3, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.85)
                self._first_frame = False
            else:
                self._vis.update_geometry(self._mesh)

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        if not alive:
            self.close()
        return alive

    def close(self) -> None:
        if self._closed:
            return
        self._vis.destroy_window()
        self._closed = True


def _drain_latest(q: queue.Queue):
    latest = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return latest
        latest = item


def _center_for_display(vertices: np.ndarray) -> np.ndarray:
    centered = vertices.astype(np.float64, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    return centered
