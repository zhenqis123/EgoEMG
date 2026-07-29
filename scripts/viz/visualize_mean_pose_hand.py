#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 mean(natural) pose 双手场景，并支持：
1) 四指稍微伸直（默认在 mean pose 基础上做轻微展开）
2) 拇指 CMC 关节主导外展（带动大鱼际打开）
3) 在指定顶点序号位置渲染 marker 小球

输出文件：
- both_hands_mean_pose.glb
- both_hands_mean_pose_topdown.png

默认输出目录：
<本脚本目录>/verification_samples/mean_pose
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import trimesh
import smplx

# 和 verify_training_with_emg.py 保持一致
MANO_MODEL_PATH = Path("../WiLoR/mano_data/models")
MANOTORCH_REPO_ROOT = Path("../manotorch")
MANOTORCH_MANO_ASSETS_ROOT = Path("../HandVQVAE/assets/mano")

# 21 个红外标记点对应顶点序号
DEFAULT_MARKER_INDICES = [
    191, 88, 253, 708, 729, 144, 87,
    295, 319, 220, 365, 407, 445,
    183, 477, 518, 556, 83, 589, 635, 673
]

# MANO 45D hand pose 中每个关节占 3 维。
# 这里按常见 MANO 顺序假设四指关节为：
# index(0-2), middle(3-5), pinky(6-8), ring(9-11), thumb(12-14)
FOUR_FINGER_JOINT_IDS = list(range(0, 12))
THUMB_JOINT_IDS = [12, 13, 14]
HAND_EXTRA_MARKERS = {
    "right": [180, 204],
    "left": [180, 158],
}

# -------------------- Hardcoded Config --------------------
HANDS = ["left"]
OUT_DIR = Path(__file__).resolve().parent / "verification_samples" / "mean_pose"
BASENAME = f"{HANDS[0]}_mean_pose"
DEVICE = "cpu"
IMAGE_SIZE = 1024
FILL_RATIO = 0.62
VIEW_ROTATE_DEG = 90.0
HAND_Z_GAP = 0.12  # 保证 left.z > right.z
STRAIGHTEN_STRENGTH = 0.60
THUMB_CMC_ABDUCT_STRENGTH = 0.1
THUMB_CMC_DORSAL_STRENGTH = -0.3
# 你要的“拇指 MCP 往手背方向横摆”控制量（越大越明显）
THUMB_MCP_DORSAL_SWING_STRENGTH = 0
# 小拇指 MCP 外展微调：负值=降低外展，正值=增加外展
PINKY_MCP_ABDUCT_DELTA = 0.15
MARKER_INDICES = list(DEFAULT_MARKER_INDICES)
MARKER_RADIUS = 0.003
SHOW_COORD_AXES = False
AXIS_ORIGIN_WORLD = np.array([0.0, 0.0, 0.0], dtype=np.float32)
AXIS_LENGTH_RATIO = 0.22  # 相对双手包围盒跨度
AXIS_THICKNESS = 3
USE_MANOTORCH_CLOSED_WRIST = True
# ---------------------------------------------------------

_CLOSED_FACES_CACHE: np.ndarray | None = None


def _get_closed_faces_from_manotorch(default_faces: np.ndarray) -> np.ndarray:
    global _CLOSED_FACES_CACHE
    if _CLOSED_FACES_CACHE is not None:
        return _CLOSED_FACES_CACHE

    if not USE_MANOTORCH_CLOSED_WRIST:
        _CLOSED_FACES_CACHE = default_faces.astype(np.int32, copy=True)
        return _CLOSED_FACES_CACHE

    try:
        if str(MANOTORCH_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(MANOTORCH_REPO_ROOT))
        from manotorch.manolayer import ManoLayer as MTManoLayer

        mt_layer = MTManoLayer(
            mano_assets_root=str(MANOTORCH_MANO_ASSETS_ROOT),
            use_pca=False,
            flat_hand_mean=False,
        )
        closed_faces = mt_layer.get_mano_closed_faces()
        if isinstance(closed_faces, torch.Tensor):
            closed_faces = closed_faces.detach().cpu().numpy()
        closed_faces = np.asarray(closed_faces, dtype=np.int32)
        _CLOSED_FACES_CACHE = closed_faces
        print(f"[Info] Using manotorch closed wrist faces: {len(closed_faces)} triangles")
        return _CLOSED_FACES_CACHE
    except Exception as e:
        print(f"[Warn] manotorch closed faces unavailable, fallback to open MANO faces: {e}")
        _CLOSED_FACES_CACHE = default_faces.astype(np.int32, copy=True)
        return _CLOSED_FACES_CACHE


def _build_hand_pose_offset(
    mano_layer: smplx.MANO,
    straighten_strength: float,
    thumb_cmc_abduct_strength: float,
    thumb_cmc_dorsal_strength: float,
    thumb_mcp_dorsal_swing_strength: float,
    pinky_mcp_abduct_delta: float,
    hand: str,
    device: str,
) -> torch.Tensor:
    """
    在 natural mean pose 基础上，让四指更伸直。

    smplx.MANO(flat_hand_mean=False) 内部会把 hand_mean 加到 hand_pose 上。
    这里给四指施加 -k * hand_mean 的 offset，使其向 flat hand 方向靠近。
    """
    pose_offset = torch.zeros(1, 45, dtype=torch.float32, device=device)

    hand_mean = mano_layer.hand_mean.view(1, 45).to(device=device, dtype=torch.float32)
    if straighten_strength > 0:
        four_finger_dims = []
        for jid in FOUR_FINGER_JOINT_IDS:
            base = jid * 3
            four_finger_dims.extend([base, base + 1, base + 2])

        four_finger_dims_t = torch.tensor(four_finger_dims, dtype=torch.long, device=device)
        pose_offset[:, four_finger_dims_t] = -straighten_strength * hand_mean[:, four_finger_dims_t]

    if thumb_cmc_abduct_strength > 0 or thumb_cmc_dorsal_strength > 0:
        thumb_dims = []
        for jid in THUMB_JOINT_IDS:
            base = jid * 3
            thumb_dims.extend([base, base + 1, base + 2])
        thumb_dims_t = torch.tensor(thumb_dims, dtype=torch.long, device=device)

        # 对拇指链整体做轻微“展开”趋势，避免只动CMC导致不自然折返。
        chain_open = 0.5 * (abs(float(thumb_cmc_abduct_strength)) + abs(float(thumb_cmc_dorsal_strength)))
        pose_offset[:, thumb_dims_t] += -0.10 * chain_open * hand_mean[:, thumb_dims_t]
        # 重点作用在 CMC 两个自由度：外展 + 朝手背方向抬起
        sign = 1.0 if hand == "right" else -1.0
        thumb_cmc = 12 * 3
        pose_offset[:, thumb_cmc + 1] += sign * thumb_cmc_abduct_strength
        pose_offset[:, thumb_cmc + 2] += sign * thumb_cmc_dorsal_strength
        pose_offset[:, thumb_cmc + 0] += -0.10 * thumb_cmc_dorsal_strength

    if thumb_mcp_dorsal_swing_strength > 0:
        # 拇指 MCP（joint 13）进一步朝手背方向“横摆”
        # 左右手用相反符号保持同方向语义。
        sign = 1.0 if hand == "right" else -1.0
        thumb_mcp = 13 * 3
        pose_offset[:, thumb_mcp + 2] += sign * thumb_mcp_dorsal_swing_strength
        pose_offset[:, thumb_mcp + 1] += sign * 0.35 * thumb_mcp_dorsal_swing_strength

    # 小拇指 MCP（joint 6）外展修正
    if abs(float(pinky_mcp_abduct_delta)) > 1e-8:
        sign = 1.0 if hand == "right" else -1.0
        pinky_mcp = 6 * 3
        pose_offset[:, pinky_mcp + 1] += sign * pinky_mcp_abduct_delta

    return pose_offset


def decode_hand_pose(
    hand: str = "right",
    device: str = "cpu",
    straighten_strength: float = 0.25,
    thumb_cmc_abduct_strength: float = 0.22,
    thumb_cmc_dorsal_strength: float = 0.34,
    thumb_mcp_dorsal_swing_strength: float = 0.16,
    pinky_mcp_abduct_delta: float = -0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode MANO hand pose in canonical space.

    Notes:
    - We always decode with MANO-right (`is_rhand=True`).
    - Left hand is produced by mirroring the decoded right-hand mesh.
    """
    mano_layer = smplx.MANO(
        model_path=str(MANO_MODEL_PATH),
        is_rhand=True,
        flat_hand_mean=False,
        use_pca=False,
        num_pca_comps=45,
    ).to(device)

    # Build pose in right-hand parameterization, then mirror for left.
    hand_pose = _build_hand_pose_offset(
        mano_layer,
        straighten_strength,
        thumb_cmc_abduct_strength,
        thumb_cmc_dorsal_strength,
        thumb_mcp_dorsal_swing_strength,
        pinky_mcp_abduct_delta,
        "right",
        device,
    )

    with torch.no_grad():
        out = mano_layer(
            global_orient=torch.zeros(1, 3, dtype=torch.float32, device=device),
            hand_pose=hand_pose,
            betas=torch.zeros(1, 10, dtype=torch.float32, device=device),
            transl=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)
    faces = _get_closed_faces_from_manotorch(mano_layer.faces.astype(np.int32))

    if hand == "left":
        verts = verts.copy()
        joints = joints.copy()
        verts[:, 0] *= -1.0
        joints[:, 0] *= -1.0
        # Keep outward normals after mirroring.
        faces = faces[:, [0, 2, 1]]

    return verts, faces, joints


def _rot_y(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rotate_points_about(points: np.ndarray, center: np.ndarray, rot_y_3x3: np.ndarray) -> np.ndarray:
    return (points - center[None, :]) @ rot_y_3x3.T + center[None, :]


def _normalize_2d(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _compute_wrist_forward_2d(joints: np.ndarray) -> np.ndarray:
    """Use wrist->finger tips average direction projected to XZ plane."""
    wrist = joints[0]
    if joints.shape[0] >= 21:
        tips = joints[[4, 8, 12, 16, 20]]
    else:
        tips = joints[1:]
    forward3 = tips.mean(axis=0) - wrist
    return _normalize_2d(np.array([forward3[0], forward3[2]], dtype=np.float32))


def _select_marker_points(verts: np.ndarray, marker_indices: list[int]) -> tuple[np.ndarray, list[int]]:
    valid_idx = [i for i in marker_indices if 0 <= i < len(verts)]
    if not valid_idx:
        return np.zeros((0, 3), dtype=np.float32), valid_idx
    return verts[np.asarray(valid_idx, dtype=np.int64)], valid_idx


def _merge_marker_indices(base_indices: list[int], hand: str) -> list[int]:
    merged = list(base_indices) + HAND_EXTRA_MARKERS[hand]
    seen = set()
    out = []
    for idx in merged:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def export_glb(
    hand_meshes: list[dict],
    out_glb: Path,
    marker_radius: float = 0.003,
) -> None:
    parts = []
    hand_colors = {
        "right": [160, 190, 230, 255],
        "left": [155, 215, 170, 255],
    }
    marker_colors = {
        "right": [230, 45, 45, 255],
        "left": [230, 45, 45, 255],
    }

    for item in hand_meshes:
        hand = item["hand"]
        mesh = trimesh.Trimesh(vertices=item["verts"], faces=item["faces"], process=False)
        mesh.visual.vertex_colors = hand_colors[hand]
        parts.append(mesh)

        for pt in item["marker_points"]:
            s = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
            s.apply_translation(pt)
            s.visual.vertex_colors = marker_colors[hand]
            parts.append(s)

    scene = trimesh.Scene(parts)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_glb))


def render_topdown_png(
    hand_meshes: list[dict],
    out_png: Path,
    image_size: int = 1024,
    fill_ratio: float = 0.62,
    view_rotate_deg: float = 90.0,
) -> None:
    """简单离屏渲染：相机从上往下看（视线方向 -Y）。"""
    all_verts = np.concatenate([x["verts"] for x in hand_meshes], axis=0).astype(np.float32)
    center = all_verts.mean(axis=0, keepdims=True)
    all_centered = all_verts - center

    span_x = float(all_centered[:, 0].max() - all_centered[:, 0].min())
    span_z = float(all_centered[:, 2].max() - all_centered[:, 2].min())
    span = max(span_x, span_z, 1e-6)
    scale = max(0.05, float(fill_ratio)) * image_size / span

    # 纯白背景（背景不参与任何阴影计算）
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    light_dir = np.array([0.15, 1.0, 0.25], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-8
    theta = np.deg2rad(view_rotate_deg)
    rot2 = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )

    def project_points_world_to_px(points_world: np.ndarray) -> np.ndarray:
        points_centered = points_world.astype(np.float32) - center
        plane = np.stack([points_centered[:, 0], -points_centered[:, 2]], axis=1)
        p2 = plane @ rot2.T
        pxy = p2 * scale + image_size * 0.5
        return pxy

    hand_colors = {
        "right": np.array([105, 160, 220], dtype=np.float32),
        "left": np.array([110, 190, 140], dtype=np.float32),
    }

    draw_records = []
    for item in hand_meshes:
        v = item["verts"].astype(np.float32) - center
        faces = item["faces"]
        hand = item["hand"]

        depth = v[:, 1]
        face_depth = depth[faces].mean(axis=1)
        for fi in np.argsort(face_depth):
            tri = faces[fi]
            tri3d = v[tri]
            plane = np.stack([tri3d[:, 0], -tri3d[:, 2]], axis=1)
            tri2 = plane @ rot2.T
            tri2d = np.round(tri2 * scale + image_size * 0.5).astype(np.int32)
            draw_records.append((float(face_depth[fi]), hand, tri3d, tri2d))

    for _, hand, tri3d, tri2d in sorted(draw_records, key=lambda x: x[0]):
        # 仅对 mesh 面片做法线光照，背景保持纯白不受影响
        n = np.cross(tri3d[1] - tri3d[0], tri3d[2] - tri3d[0])
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            continue
        n = n / n_norm
        shade = float(np.clip(abs(np.dot(n, light_dir)), 0.18, 1.0))
        base_rgb = hand_colors[hand]
        color_rgb = (base_rgb * shade + 20.0).clip(0, 255).astype(np.uint8)
        color_bgr = tuple(int(c) for c in color_rgb[::-1])
        cv2.fillConvexPoly(canvas, tri2d, color_bgr, lineType=cv2.LINE_AA)

    # 不叠加线框边缘，避免过度强调 mesh edges。

    if SHOW_COORD_AXES:
        axis_len = float(span * AXIS_LENGTH_RATIO)
        axis_defs = [
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float32), (0, 0, 255)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float32), (0, 255, 0)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float32), (255, 0, 0)),
        ]
        origin_2d = project_points_world_to_px(AXIS_ORIGIN_WORLD[None, :])[0]
        origin_px = (int(round(origin_2d[0])), int(round(origin_2d[1])))

        for axis_name, axis_dir, axis_color in axis_defs:
            end_world = AXIS_ORIGIN_WORLD + axis_len * axis_dir
            pts2 = project_points_world_to_px(
                np.stack([AXIS_ORIGIN_WORLD, end_world], axis=0)
            )
            p0 = pts2[0]
            p1 = pts2[1]
            pix_len = float(np.linalg.norm(p1 - p0))
            # 顶视图下 Y 轴可能退化为零长度；按需跳过。
            if pix_len < 2.0:
                continue

            p1_px = (int(round(p1[0])), int(round(p1[1])))
            cv2.arrowedLine(
                canvas,
                origin_px,
                p1_px,
                axis_color,
                AXIS_THICKNESS,
                line_type=cv2.LINE_AA,
                tipLength=0.16,
            )
            cv2.putText(
                canvas,
                axis_name,
                (p1_px[0] + 5, p1_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                axis_color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(canvas, origin_px, 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)

    for item in hand_meshes:
        marker_points = item["marker_points"]
        if len(marker_points) == 0:
            continue
        m = marker_points.astype(np.float32) - center
        plane = np.stack([m[:, 0], -m[:, 2]], axis=1)
        m2 = plane @ rot2.T
        mu = m2[:, 0] * scale + image_size * 0.5
        mw = m2[:, 1] * scale + image_size * 0.5
        for x, y in zip(mu, mw):
            p = (int(round(x)), int(round(y)))
            # 普通 marker 圆点（无小球高光）
            cv2.circle(canvas, p, 5, (45, 45, 225), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, 7, (220, 220, 255), 1, lineType=cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def main() -> None:
    hands = [h for h in dict.fromkeys(HANDS) if h in ("left", "right")]
    hand_meshes = []
    hand_joints = {}
    marker_count_total = 0
    for hand in hands:
        verts, faces, joints = decode_hand_pose(
            hand=hand,
            device=DEVICE,
            straighten_strength=max(0.0, float(STRAIGHTEN_STRENGTH)),
            thumb_cmc_abduct_strength=float(THUMB_CMC_ABDUCT_STRENGTH),
            thumb_cmc_dorsal_strength=float(THUMB_CMC_DORSAL_STRENGTH),
            thumb_mcp_dorsal_swing_strength=max(0.0, float(THUMB_MCP_DORSAL_SWING_STRENGTH)),
            pinky_mcp_abduct_delta=float(PINKY_MCP_ABDUCT_DELTA),
        )
        hand_joints[hand] = joints

        marker_indices = _merge_marker_indices(list(MARKER_INDICES), hand)
        marker_points, valid_idx = _select_marker_points(verts, marker_indices)
        marker_count_total += len(valid_idx)
        if len(valid_idx) != len(marker_indices):
            print(f"[Warn] {hand} marker indices total={len(marker_indices)}, valid={len(valid_idx)}")
        hand_meshes.append(
            {
                "hand": hand,
                "verts": verts,
                "faces": faces,
                "marker_points": marker_points,
            }
        )

    if len(hand_meshes) == 0:
        raise ValueError("No valid hands to render.")

    # 固定布局：不做自动配准；仅保证 left.z > right.z。
    mesh_by_hand = {x["hand"]: x for x in hand_meshes}
    if "left" in mesh_by_hand and "right" in mesh_by_hand:
        left_mesh = mesh_by_hand["left"]
        right_mesh = mesh_by_hand["right"]

        # 仅做方向一致性修正：若手腕朝向相反，左手绕Y轴转180度。
        f_r = _compute_wrist_forward_2d(hand_joints["right"])
        f_l = _compute_wrist_forward_2d(hand_joints["left"])
        if float(np.dot(f_r, f_l)) < 0.0:
            rot = _rot_y(np.pi)
            c = hand_joints["left"][0].astype(np.float32)
            left_mesh["verts"] = _rotate_points_about(left_mesh["verts"], c, rot)
            left_mesh["marker_points"] = _rotate_points_about(left_mesh["marker_points"], c, rot)
            hand_joints["left"] = _rotate_points_about(hand_joints["left"], c, rot)

        dz = float(HAND_Z_GAP) * 0.5
        left_mesh["verts"][:, 2] += dz
        left_mesh["marker_points"][:, 2] += dz
        right_mesh["verts"][:, 2] -= dz
        right_mesh["marker_points"][:, 2] -= dz
    else:
        # 单手时不额外平移。
        pass

    basename = BASENAME if BASENAME else ("both_hands_mean_pose" if set(hands) == {"left", "right"} else f"{hands[0]}_hand_mean_pose")

    out_glb = OUT_DIR / f"{basename}.glb"
    out_png = OUT_DIR / f"{basename}_topdown.png"

    export_glb(hand_meshes, out_glb, marker_radius=MARKER_RADIUS)
    render_topdown_png(
        hand_meshes,
        out_png,
        image_size=IMAGE_SIZE,
        fill_ratio=FILL_RATIO,
        view_rotate_deg=VIEW_ROTATE_DEG,
    )

    print(f"Saved GLB: {out_glb}")
    print(f"Saved PNG: {out_png}")
    print(f"Hands: {hands}")
    print(f"Marker count: {marker_count_total}")


if __name__ == "__main__":
    main()
