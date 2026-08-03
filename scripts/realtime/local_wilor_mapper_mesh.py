#!/usr/bin/env python3
"""Realtime camera -> WiLoR -> MANO-theta mapper -> UmeTrack mesh visualization."""

from __future__ import annotations

import argparse
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tv_ops

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_COLLECT_ROOT = _PROJECT_ROOT / "data_collect"
if _DATA_COLLECT_ROOT.exists() and str(_DATA_COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECT_ROOT))

from egoemg.realtime_local.mano_mapper import RuntimeManoToUmeTrackMapper
from egoemg.realtime_local.mesh_visualizer import RealtimeMeshVisualizer
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)


ALIGN_SCALE = 1.0843137502670288
ALIGN_TRANS = np.array([106.72334, -11.8804455, -4.48328], dtype=np.float32)
FLIP_MATRIX = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
MANO_TO_UMETRACK = {
    0: 5,
    2: 6,
    3: 7,
    4: 0,
    5: 8,
    6: 9,
    7: 10,
    8: 1,
    9: 11,
    10: 12,
    11: 13,
    12: 2,
    13: 14,
    14: 15,
    15: 16,
    16: 3,
    17: 17,
    18: 18,
    19: 19,
    20: 4,
}
MANO_IDX = sorted(MANO_TO_UMETRACK.keys())
UMETRACK_IDX = [MANO_TO_UMETRACK[m] for m in MANO_IDX]


@dataclass
class BBox:
    center: np.ndarray
    size: float
    xyxy: np.ndarray
    detected: bool


@dataclass
class WilorMapperResult:
    angles: np.ndarray
    bbox: BBox
    inference_ms: float
    pose_source: str
    mano_vertices: np.ndarray | None = None
    mano_triangles: np.ndarray | None = None


def _align_mano_points_to_umetrack(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) * 1000.0
    return (ALIGN_SCALE * (pts @ FLIP_MATRIX.T) + ALIGN_TRANS[None, :]).astype(np.float32)


def _flip_triangle_winding(triangles: np.ndarray | None) -> np.ndarray | None:
    if triangles is None:
        return None
    tri = np.asarray(triangles, dtype=np.int32).copy()
    tri[:, [1, 2]] = tri[:, [2, 1]]
    return tri


class UmeTrackLbfgsFitter:
    """Fit UmeTrack 20D finger angles to canonical WiLoR MANO landmarks."""

    def __init__(
        self,
        mano_layer,
        *,
        device: torch.device,
        max_iter: int,
        lr: float,
        history_size: int,
    ) -> None:
        from egoemg.kinematics import apply_to_hand_model, load_default_hand_model

        self.mano_layer = mano_layer
        self.device = device
        self.mano_dtype = next(mano_layer.buffers()).dtype
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.history_size = int(history_size)
        self.last_angles: torch.Tensor | None = None

        hand_model = load_default_hand_model()
        self.hand_model = apply_to_hand_model(hand_model, lambda t: t.float().unsqueeze(0).to(device))
        json_path = (
            _DATA_COLLECT_ROOT
            / "emg2pose"
            / "UmeTrack"
            / "dataset"
            / "generic_hand_model.json"
        )
        if not json_path.exists():
            json_path = (
                _PROJECT_ROOT
                / "emg2pose"
                / "UmeTrack"
                / "dataset"
                / "generic_hand_model.json"
            )
        import json

        with open(json_path, encoding="utf-8") as f:
            limits = np.asarray(json.load(f)["joint_limits"][:20], dtype=np.float32)
        self.angle_min = torch.from_numpy(limits[:, 0]).to(device)
        self.angle_max = torch.from_numpy(limits[:, 1]).to(device)
        self.flip_t = torch.from_numpy(FLIP_MATRIX).float().to(device)
        self.align_trans = torch.from_numpy(ALIGN_TRANS).float().to(device)
        self.mano_idx = torch.tensor(MANO_IDX, dtype=torch.long, device=device)
        self.umetrack_idx = torch.tensor(UMETRACK_IDX, dtype=torch.long, device=device)

    @torch.no_grad()
    def canonical_mano_geometry(
        self,
        hand_pose45: np.ndarray,
        *,
        need_vertices: bool,
    ) -> tuple[torch.Tensor, np.ndarray | None]:
        import roma

        hand_pose = torch.from_numpy(hand_pose45.reshape(1, 15, 3)).to(
            device=self.device,
            dtype=self.mano_dtype,
        )
        hand_pose_rot = roma.rotvec_to_rotmat(hand_pose.reshape(-1, 3)).reshape(
            1,
            15,
            3,
            3,
        )
        global_orient = torch.eye(3, dtype=self.mano_dtype, device=self.device).reshape(
            1,
            1,
            3,
            3,
        )
        beta = torch.zeros(1, 10, dtype=self.mano_dtype, device=self.device)
        # Match scripts/ik/batch_ik_mesh.py: zero global_orient, zero beta,
        # then fixed MANO-right to UmeTrack-space flip/scale/translation.
        # WiLoR's MANO wrapper is smplx.MANOLayer(flat_hand_mean=False).
        out = self.mano_layer(
            global_orient=global_orient,
            hand_pose=hand_pose_rot,
            betas=beta,
            pose2rot=False,
        )
        mano_j = out.joints.float() * 1000.0
        target = ALIGN_SCALE * (mano_j @ self.flip_t.T) + self.align_trans.view(1, 1, 3)
        vertices_np = None
        if need_vertices:
            mano_v = out.vertices.float() * 1000.0
            vertices = ALIGN_SCALE * (mano_v @ self.flip_t.T) + self.align_trans.view(1, 1, 3)
            vertices_np = vertices[0].detach().cpu().numpy().astype(np.float32)
        return target[:, self.mano_idx, :].detach(), vertices_np

    def _initial_raw(self) -> torch.Tensor:
        if self.last_angles is None:
            angles = torch.zeros(1, 20, device=self.device)
        else:
            angles = self.last_angles.detach().reshape(1, 20).to(self.device)
        eps = 1e-4
        ratio = (angles - self.angle_min) / (self.angle_max - self.angle_min).clamp_min(1e-6)
        ratio = ratio.clamp(eps, 1.0 - eps)
        return torch.logit(ratio).detach().requires_grad_(True)

    def _angles_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        return self.angle_min + torch.sigmoid(raw) * (self.angle_max - self.angle_min)

    def _umetrack_landmarks(self, angles20: torch.Tensor) -> torch.Tensor:
        from egoemg.UmeTrack.lib.common.hand_skinning import skin_landmarks

        wrist_tf = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        return skin_landmarks(self.hand_model, angles20.float(), wrist_transforms=wrist_tf)

    def fit(self, hand_pose45: np.ndarray) -> np.ndarray:
        target_lm, _ = self.canonical_mano_geometry(hand_pose45, need_vertices=False)
        return self.fit_target_landmarks(target_lm)

    def fit_target_landmarks(self, target_landmarks: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(target_landmarks, np.ndarray):
            target_lm = torch.from_numpy(target_landmarks).float().to(self.device)
        else:
            target_lm = target_landmarks.float().to(self.device)
        if target_lm.ndim == 2:
            target_lm = target_lm.unsqueeze(0)
        raw_angles = self._initial_raw()
        optimizer = torch.optim.LBFGS(
            [raw_angles],
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            angles = self._angles_from_raw(raw_angles)
            pred_lm = self._umetrack_landmarks(angles)[:, self.umetrack_idx, :]
            # Landmark-only objective: no mesh/chamfer term is used here.
            loss = ((pred_lm - target_lm) ** 2).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            angles = self._angles_from_raw(raw_angles).detach()
            self.last_angles = angles.reshape(20)
        return self.last_angles.cpu().numpy().astype(np.float32)


class RealtimeDualMeshVisualizer:
    """Open3D visualizer for UmeTrack mesh and optional aligned MANO mesh."""

    def __init__(self, max_queue: int = 2) -> None:
        try:
            import open3d as o3d
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Open3D import failed: {exc}") from exc

        from egoemg.realtime_local.mesh_visualizer import UmeTrackMeshForwarder

        self._o3d = o3d
        self._queue: queue.Queue[WilorMapperResult] = queue.Queue(maxsize=max_queue)
        self._forwarder = UmeTrackMeshForwarder()
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window("WiLoR UmeTrack + MANO Mesh", width=1100, height=760):
            raise RuntimeError("Open3D failed to create a visualization window")
        self._ut_mesh = o3d.geometry.TriangleMesh()
        self._mano_mesh = o3d.geometry.TriangleMesh()
        self._first_frame = True
        self._has_mano = False
        self._closed = False

    def update_result(self, result: WilorMapperResult) -> None:
        while True:
            try:
                self._queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def poll(self) -> bool:
        if self._closed:
            return False
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            ut = self._forwarder(latest.angles)
            vertices = [ut.vertices]
            if latest.mano_vertices is not None:
                vertices.append(latest.mano_vertices)
            center = np.concatenate(vertices, axis=0).mean(axis=0, keepdims=True)

            self._ut_mesh.vertices = self._o3d.utility.Vector3dVector(ut.vertices - center)
            self._ut_mesh.triangles = self._o3d.utility.Vector3iVector(ut.triangles)
            self._ut_mesh.compute_vertex_normals()
            self._ut_mesh.paint_uniform_color([0.90, 0.48, 0.25])

            if latest.mano_vertices is not None and latest.mano_triangles is not None:
                self._mano_mesh.vertices = self._o3d.utility.Vector3dVector(
                    latest.mano_vertices - center
                )
                self._mano_mesh.triangles = self._o3d.utility.Vector3iVector(
                    latest.mano_triangles
                )
                self._mano_mesh.compute_vertex_normals()
                self._mano_mesh.paint_uniform_color([0.20, 0.48, 0.95])

            if self._first_frame:
                self._vis.add_geometry(self._ut_mesh)
                if latest.mano_vertices is not None and latest.mano_triangles is not None:
                    self._vis.add_geometry(self._mano_mesh)
                    self._has_mano = True
                frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                self._vis.add_geometry(frame)
                ctr = self._vis.get_view_control()
                ctr.set_front([0.0, -0.3, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.85)
                self._first_frame = False
            else:
                self._vis.update_geometry(self._ut_mesh)
                if latest.mano_vertices is not None and latest.mano_triangles is not None:
                    if not self._has_mano:
                        self._vis.add_geometry(self._mano_mesh)
                        self._has_mano = True
                    else:
                        self._vis.update_geometry(self._mano_mesh)

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


def _extract_mano_faces(mano_layer) -> np.ndarray | None:
    for name in ("faces_tensor", "faces"):
        faces = getattr(mano_layer, name, None)
        if faces is None:
            continue
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        faces_np = np.asarray(faces, dtype=np.int32)
        if faces_np.ndim == 2 and faces_np.shape[1] == 3:
            return faces_np
    return None


def _open_camera(camera: int, width: int, height: int, fps: float, backend: str):
    backend_id = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _detect_bbox(
    detector,
    frame_rgb: np.ndarray,
    *,
    target_is_right: bool,
    yolo_conf: float,
    yolo_input_height: int,
) -> BBox | None:
    image_h, image_w = frame_rgb.shape[:2]
    if yolo_input_height > 0:
        yolo_h = int(yolo_input_height)
        yolo_w = max(int(image_w * yolo_h / image_h), 1)
        image = cv2.resize(frame_rgb, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
        scale_x = image_w / float(yolo_w)
        scale_y = image_h / float(yolo_h)
    else:
        image = frame_rgb
        scale_x = scale_y = 1.0

    results = detector([image], conf=yolo_conf, verbose=False)
    result = results[0] if isinstance(results, list) else results
    if result is None or len(result) == 0:
        return None

    best_xyxy: np.ndarray | None = None
    best_conf = -1.0
    for det in result:
        is_right = bool(int(det.boxes.cls.cpu().item()))
        if is_right != target_is_right:
            continue
        conf = float(det.boxes.conf.cpu().item()) if det.boxes.conf is not None else 0.0
        box = det.boxes.data.cpu().squeeze().numpy()[:4].astype(np.float32)
        box *= np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        if conf > best_conf:
            best_conf = conf
            best_xyxy = box
    if best_xyxy is None:
        return None

    x1, y1, x2, y2 = [float(v) for v in best_xyxy]
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    size = float(max(x2 - x1, y2 - y1) * 2.5)
    return BBox(center=center, size=size, xyxy=best_xyxy, detected=True)


class RealtimeWilorMapper:
    def __init__(
        self,
        *,
        mapper_checkpoint: str | Path,
        hand: str,
        device: str,
        dtype: str,
        yolo_conf: float,
        yolo_input_height: int,
        wilor_pretrained_dir: str | None,
        detect_interval: int,
        max_bbox_age: int,
        pose_source: str,
        lbfgs_max_iter: int,
        lbfgs_lr: float,
        lbfgs_history_size: int,
        visualize_mano_mesh: bool,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.target_is_right = hand == "right"
        self.yolo_conf = yolo_conf
        self.yolo_input_height = yolo_input_height
        self.detect_interval = max(1, int(detect_interval))
        self.max_bbox_age = max(0, int(max_bbox_age))
        self.pose_source = pose_source
        self.visualize_mano_mesh = bool(visualize_mano_mesh)
        self.frame_index = 0
        self.last_bbox: BBox | None = None
        self.last_bbox_frame = -10**9

        kwargs = {"device": self.device, "dtype": self.dtype}
        if wilor_pretrained_dir:
            kwargs["wilor_pretrained_dir"] = wilor_pretrained_dir
        self.pipe = WiLorHandPose3dEstimationPipeline(**kwargs)
        self.detector = self.pipe.hand_detector
        self.model = self.pipe.wilor_model
        self.model.eval()
        self.mano_triangles = _flip_triangle_winding(_extract_mano_faces(self.model.mano))
        self.mapper = (
            RuntimeManoToUmeTrackMapper(mapper_checkpoint, device=self.device)
            if pose_source == "mapper"
            else None
        )
        self.lbfgs_fitter = (
            UmeTrackLbfgsFitter(
                self.model.mano,
                device=self.device,
                max_iter=lbfgs_max_iter,
                lr=lbfgs_lr,
                history_size=lbfgs_history_size,
            )
            if pose_source == "lbfgs" or visualize_mano_mesh
            else None
        )

    def predict(self, frame_rgb: np.ndarray) -> WilorMapperResult | None:
        t0 = time.perf_counter()
        should_detect = self.frame_index % self.detect_interval == 0 or self.last_bbox is None
        bbox = None
        if should_detect:
            bbox = _detect_bbox(
                self.detector,
                frame_rgb,
                target_is_right=self.target_is_right,
                yolo_conf=self.yolo_conf,
                yolo_input_height=self.yolo_input_height,
            )
            if bbox is not None:
                self.last_bbox = bbox
                self.last_bbox_frame = self.frame_index

        if bbox is None and self.last_bbox is not None:
            if self.frame_index - self.last_bbox_frame <= self.max_bbox_age:
                bbox = BBox(
                    center=self.last_bbox.center.copy(),
                    size=self.last_bbox.size,
                    xyxy=self.last_bbox.xyxy.copy(),
                    detected=False,
                )
        self.frame_index += 1
        if bbox is None:
            return None

        image_h, image_w = frame_rgb.shape[:2]
        half = bbox.size * 0.5
        roi = [[0, bbox.center[0] - half, bbox.center[1] - half, bbox.center[0] + half, bbox.center[1] + half]]
        frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )
        boxes_t = torch.tensor(roi, device=self.device, dtype=torch.float32)
        patches_chw = tv_ops.roi_align(
            frame_t,
            boxes_t,
            output_size=(self.pipe.IMAGE_SIZE, self.pipe.IMAGE_SIZE),
            spatial_scale=1.0,
            aligned=True,
        )
        patches_nhwc = patches_chw.permute(0, 2, 3, 1).to(dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model(patches_nhwc)
        hand_pose = outputs["hand_pose"][0].detach().float().cpu().numpy().copy()
        mano_vertices = None
        if not self.target_is_right:
            hand_pose[:, 1:3] = -hand_pose[:, 1:3]

        target_landmarks = None
        if self.lbfgs_fitter is not None and (
            self.pose_source == "lbfgs" or self.visualize_mano_mesh
        ):
            target_landmarks, mano_vertices = self.lbfgs_fitter.canonical_mano_geometry(
                hand_pose.reshape(45),
                need_vertices=self.visualize_mano_mesh,
            )

        if self.pose_source == "mapper":
            assert self.mapper is not None
            angles20 = self.mapper.predict(hand_pose.reshape(45))
        else:
            assert self.lbfgs_fitter is not None
            assert target_landmarks is not None
            angles20 = self.lbfgs_fitter.fit_target_landmarks(target_landmarks)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        # Keep image dimensions referenced for easier debugging of bad boxes.
        _ = (image_w, image_h)
        return WilorMapperResult(
            angles=angles20,
            bbox=bbox,
            inference_ms=inference_ms,
            pose_source=self.pose_source,
            mano_vertices=mano_vertices,
            mano_triangles=self.mano_triangles if mano_vertices is not None else None,
        )


def _draw_status(frame_bgr: np.ndarray, result: WilorMapperResult | None, fps: float) -> None:
    if result is not None:
        x1, y1, x2, y2 = [int(v) for v in result.bbox.xyxy]
        color = (60, 220, 60) if result.bbox.detected else (0, 180, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"WiLoR+{result.pose_source} {result.inference_ms:.1f}ms fps={fps:.1f}"
    else:
        text = f"no hand fps={fps:.1f}"
    cv2.putText(frame_bgr, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 3)
    cv2.putText(frame_bgr, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=["dshow", "any"], default="dshow")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--mapper-checkpoint", default="pretrained_models/mano_to_umetrack_mapper.pt")
    parser.add_argument("--pose-source", choices=["mapper", "lbfgs"], default="mapper")
    parser.add_argument("--lbfgs-max-iter", type=int, default=3)
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--wilor-pretrained-dir", default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.1)
    parser.add_argument("--yolo-input-height", type=int, default=512)
    parser.add_argument("--detect-interval", type=int, default=1)
    parser.add_argument("--max-bbox-age", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N camera frames; 0 means run forever.")
    parser.add_argument("--visualize-mano-mesh", action="store_true")
    parser.add_argument("--no-camera-preview", action="store_true")
    args = parser.parse_args()

    cap = _open_camera(args.camera, args.width, args.height, args.fps, args.backend)
    predictor = RealtimeWilorMapper(
        mapper_checkpoint=args.mapper_checkpoint,
        hand=args.hand,
        device=args.device,
        dtype=args.dtype,
        yolo_conf=args.yolo_conf,
        yolo_input_height=args.yolo_input_height,
        wilor_pretrained_dir=args.wilor_pretrained_dir,
        detect_interval=args.detect_interval,
        max_bbox_age=args.max_bbox_age,
        pose_source=args.pose_source,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_history_size=args.lbfgs_history_size,
        visualize_mano_mesh=args.visualize_mano_mesh,
    )
    visualizer = RealtimeDualMeshVisualizer() if args.visualize_mano_mesh else RealtimeMeshVisualizer()

    n_frames = 0
    n_results = 0
    t_start = time.perf_counter()
    print(
        f"Started realtime WiLoR {args.pose_source} mesh. "
        "Press q in camera preview or close mesh window to stop."
    )
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            n_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = predictor.predict(frame_rgb)
            if result is not None:
                n_results += 1
                if args.visualize_mano_mesh:
                    visualizer.update_result(result)
                else:
                    visualizer.update(result.angles, result.inference_ms)

            elapsed = max(time.perf_counter() - t_start, 1e-6)
            fps = n_frames / elapsed
            if n_frames % 10 == 0 or result is not None:
                infer_ms = result.inference_ms if result is not None else 0.0
                print(
                    f"\rframes={n_frames:6d} results={n_results:6d} "
                    f"fps={fps:5.1f} infer={infer_ms:6.1f}ms",
                    end="",
                    flush=True,
                )
            if not visualizer.poll():
                break
            if args.max_frames > 0 and n_frames >= args.max_frames:
                break
            if not args.no_camera_preview:
                _draw_status(frame_bgr, result, fps)
                cv2.imshow("WiLoR camera", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        visualizer.close()
        if not args.no_camera_preview:
            cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
