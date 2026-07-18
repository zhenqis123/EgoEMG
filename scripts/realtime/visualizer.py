"""Real-time 3D hand skeleton visualizer using Open3D.

Renders 21 hand keypoints as a connected skeleton with color-coded fingers.
Uses a separate process for rendering to avoid Open3D blocking the main thread
(which handles ZMQ I/O and serial data collection).
Falls back to a simple terminal display if Open3D is not available.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from queue import Empty

import numpy as np

# UmeTrack 21-landmark ordering (from emg2pose/UmeTrack/lib/common/hand.py):
#   0: Thumb TIP
#   1: Index TIP
#   2: Middle TIP
#   3: Ring TIP
#   4: Pinky TIP
#   5: Wrist
#   6: Thumb CMC
#   7: Thumb MCP
#   8: Index CMC (proximal)
#   9: Index MCP (intermediate)
#   10: Index IP (distal)
#   11: Middle CMC (proximal)
#   12: Middle MCP (intermediate)
#   13: Middle IP (distal)
#   14: Ring CMC (proximal)
#   15: Ring MCP (intermediate)
#   16: Ring IP (distal)
#   17: Pinky CMC (proximal)
#   18: Pinky MCP (intermediate)
#   19: Pinky IP (distal)
#   20: Palm center

# Bone connectivity: (parent_idx, child_idx)
BONES = [
    # Thumb: wrist → CMC → MCP → TIP
    (5, 6), (6, 7), (7, 0),
    # Index: wrist → CMC → MCP → IP → TIP
    (5, 8), (8, 9), (9, 10), (10, 1),
    # Middle: wrist → CMC → MCP → IP → TIP
    (5, 11), (11, 12), (12, 13), (13, 2),
    # Ring: wrist → CMC → MCP → IP → TIP
    (5, 14), (14, 15), (15, 16), (16, 3),
    # Pinky: wrist → CMC → MCP → IP → TIP
    (5, 17), (17, 18), (18, 19), (19, 4),
    # Palm structure
    (5, 20), (20, 8), (20, 11),
]

FINGER_COLORS = [
    [1.0, 0.3, 0.3],   # thumb: red
    [0.3, 1.0, 0.3],   # index: green
    [0.3, 0.5, 1.0],   # middle: blue
    [1.0, 0.9, 0.2],   # ring: yellow
    [0.8, 0.4, 1.0],   # pinky: purple
]

# Map each bone to its finger color index
_BONE_TO_FINGER = [
    0, 0, 0,             # thumb (3 bones)
    1, 1, 1, 1,          # index (4 bones)
    2, 2, 2, 2,          # middle (4 bones)
    3, 3, 3, 3,          # ring (4 bones)
    4, 4, 4, 4,          # pinky (4 bones)
    0, 1, 2,             # palm connections (gray-ish)
]

# UmeTrack landmark index → finger assignment for point coloring
# -1 means "not a finger joint" (wrist or palm)
_LANDMARK_FINGER = np.array([
    0,  # 0: Thumb TIP
    1,  # 1: Index TIP
    2,  # 2: Middle TIP
    3,  # 3: Ring TIP
    4,  # 4: Pinky TIP
    -1, # 5: Wrist
    0,  # 6: Thumb CMC
    0,  # 7: Thumb MCP
    1,  # 8: Index CMC
    1,  # 9: Index MCP
    1,  # 10: Index IP
    2,  # 11: Middle CMC
    2,  # 12: Middle MCP
    2,  # 13: Middle IP
    3,  # 14: Ring CMC
    3,  # 15: Ring MCP
    3,  # 16: Ring IP
    4,  # 17: Pinky CMC
    4,  # 18: Pinky MCP
    4,  # 19: Pinky IP
    -1, # 20: Palm center
])

# Pre-compute static point colors
_POINT_COLORS = np.tile([0.6, 0.6, 0.6], (21, 1))
_POINT_COLORS[5] = [1.0, 1.0, 1.0]  # wrist: white
_POINT_COLORS[20] = [0.5, 0.5, 0.5]  # palm center: dark gray
for i, fi in enumerate(_LANDMARK_FINGER):
    if fi >= 0:
        _POINT_COLORS[i] = FINGER_COLORS[fi]

# Pre-compute bone colors
_BONE_COLORS = [FINGER_COLORS[_BONE_TO_FINGER[i]] for i in range(len(BONES))]


def _render_process(cmd_queue: mp.Queue) -> None:
    """Open3D render loop running in a dedicated process.

    This isolates Open3D's event loop from the main process's ZMQ/serial I/O,
    preventing GIL contention and blocking issues on Windows.
    """
    try:
        import open3d as o3d
    except ImportError:
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="EMG Hand Pose", width=800, height=600)

    # Render options: larger points for visibility
    opt = vis.get_render_option()
    opt.point_size = 8.0
    opt.line_width = 3.0
    opt.background_color = np.array([0.15, 0.15, 0.15])  # dark gray bg

    # Create point cloud for joints
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    pc.colors = o3d.utility.Vector3dVector(_POINT_COLORS.copy())

    # Create line set for bones
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    ls.lines = o3d.utility.Vector2iVector(np.array(BONES))
    ls.colors = o3d.utility.Vector3dVector(_BONE_COLORS)

    # Add a coordinate frame at origin for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, 0, 0]
    )

    vis.add_geometry(pc)
    vis.add_geometry(ls)
    vis.add_geometry(coord_frame)

    first_real_data = True
    running = True
    latest_pts = None
    frame_count = 0

    while running:
        # Drain ALL pending commands (non-blocking) to get the latest update
        # This prevents lag when multiple predictions queue up
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except Empty:
                break

            if cmd[0] == "quit":
                running = False
                break
            elif cmd[0] == "update":
                _, landmarks, _angles, _inference_ms = cmd
                pts = np.asarray(landmarks, dtype=np.float64)
                frame_count += 1

                is_valid = (
                    pts.shape == (21, 3)
                    and np.all(np.isfinite(pts))
                    and not np.allclose(pts, 0)
                )

                if frame_count <= 3:
                    print(
                        f"[viz] frame {frame_count}: shape={pts.shape}, "
                        f"valid={is_valid}, "
                        f"min={pts.min():.4f}, max={pts.max():.4f}",
                        flush=True,
                    )

                if is_valid:
                    latest_pts = pts

                if not running:
                    break

        # Apply the latest update (if any) before rendering
        if latest_pts is not None:
            pc.points = o3d.utility.Vector3dVector(latest_pts)
            pc.colors = o3d.utility.Vector3dVector(_POINT_COLORS)
            ls.points = o3d.utility.Vector3dVector(latest_pts)
            vis.update_geometry(pc)
            vis.update_geometry(ls)

            if first_real_data:
                vis.reset_view_point(True)
                first_real_data = False
                print(
                    f"[viz] First valid data, camera reset. "
                    f"range=[{latest_pts.min():.4f}, {latest_pts.max():.4f}]",
                    flush=True,
                )
            latest_pts = None

        # poll_events returns False when the user closes the window
        if not vis.poll_events():
            running = False
            break

    vis.destroy_window()


def _update_terminal(
    landmarks: np.ndarray,
    angles: np.ndarray | None,
    inference_ms: float,
) -> None:
    """Simple terminal display of joint angles."""
    if angles is None:
        return

    sys.stdout.write("\033[2J\033[H")
    print("=== EMG Hand Pose (terminal mode) ===")
    print(f"  Inference: {inference_ms:.1f} ms")
    print(f"  Landmarks: {landmarks.shape}")
    print()

    if len(angles) >= 20:
        finger_names = ["Thumb", "Index", "Middle", "Ring ", "Pinky"]
        joint_names = ["MCP_aa", "MCP_fe", "PIP_fe", "DIP_fe"]
        thumb_names = ["CMC_fe", "CMC_aa", "MCP_fe", "IP_fe  "]

        for f in range(5):
            name = finger_names[f]
            start = f * 4
            vals = np.degrees(angles[start:start + 4])
            jnames = thumb_names if f == 0 else joint_names
            print(f"  {name}: ", end="")
            for j, (jn, v) in enumerate(zip(jnames, vals)):
                print(f"{jn}={v:6.1f}°", end="  ")
            print()

        if len(angles) > 20:
            wrist = np.degrees(angles[20:22])
            print(f"\n  Wrist: pitch={wrist[0]:.1f}°  yaw={wrist[1]:.1f}°")

    sys.stdout.flush()


class HandVisualizer:
    """Real-time 3D hand skeleton renderer.

    Runs Open3D in a separate process to avoid blocking the main thread's
    ZMQ/serial I/O.  Falls back to terminal display if Open3D is unavailable.
    """

    def __init__(self, use_gui: bool = True):
        self.use_gui = use_gui
        self._queue: mp.Queue | None = None
        self._process: mp.Process | None = None

        if use_gui:
            try:
                import open3d as o3d  # noqa: F401
            except ImportError:
                print("Warning: open3d not installed, using terminal display")
                self.use_gui = False

    def _ensure_started(self) -> None:
        """Lazily start the render process on first update."""
        if self._process is not None:
            return
        if self.use_gui:
            ctx = mp.get_context("spawn")
            self._queue = ctx.Queue()
            self._process = ctx.Process(
                target=_render_process,
                args=(self._queue,),
                daemon=True,
            )
            self._process.start()
        else:
            # Terminal mode needs no subprocess
            self._process = None

    def update(
        self,
        landmarks: np.ndarray,
        angles: np.ndarray | None = None,
        inference_ms: float = 0.0,
    ) -> bool:
        """Update the visualization with new landmarks.

        Args:
            landmarks: (21, 3) array of hand keypoints.
            angles: (20,) or (22,) array of joint angles (optional).
            inference_ms: Inference time in ms.

        Returns:
            True while the visualizer is active.
        """
        self._ensure_started()

        if self.use_gui and self._queue is not None:
            if self._process is not None and not self._process.is_alive():
                return False
            try:
                self._queue.put_nowait(
                    ("update", landmarks.copy(),
                     angles.copy() if angles is not None else None,
                     inference_ms)
                )
            except Exception:
                pass
            return True
        else:
            _update_terminal(landmarks, angles, inference_ms)
            return True

    def close(self) -> None:
        """Shut down the render process."""
        if self._queue is not None:
            try:
                self._queue.put_nowait(("quit",))
            except Exception:
                pass
        if self._process is not None:
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
        self._queue = None
