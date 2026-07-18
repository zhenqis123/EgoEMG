#!/usr/bin/env python3
"""
Single-frame IK test: optimize UmeTrack 20D angles to match MANO keypoints/mesh.

Alignment (hardcoded, session-independent calibration):
    MANO mesh --[flip X]--> [scale] --> [rotate(opt_global_orient)] --> [translate] --> UmeTrack space

Loss: Chamfer (mesh) and/or L2 (landmarks).

Usage:
    python scripts/ik/test_ik_mesh_single.py \
        --memmap-root data/incre_2/memmap --frame 200 --hand right \
        --loss-type landmark --max-iter 200 --opt-global-orient --opt-trans
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))
from manotorch.manolayer import ManoLayer

MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")

# ── Alignment constants (MANO rest → UmeTrack rest, flat_hand_mean=False) ──
# Session-independent MANO-rest to UmeTrack-rest calibration constants.
ALIGN_SCALE = 1.0843137502670288
ALIGN_TRANS = np.array([106.72334, -11.8804455, -4.48328], dtype=np.float32)

# Negate X to convert MANO-right into UmeTrack's left-hand frame.
FLIP_MATRIX = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

# Landmark mapping: MANO joint index → UmeTrack landmark index.
MANO_TO_UMETRACK = {
    0: 5, 2: 6, 3: 7, 4: 0,
    5: 8, 6: 9, 7: 10, 8: 1,
    9: 11, 10: 12, 11: 13, 12: 2,
    13: 14, 14: 15, 15: 16, 16: 3,
    17: 17, 18: 18, 19: 19, 20: 4,
}
MANO_IDX = sorted(MANO_TO_UMETRACK.keys())
UMETRACK_IDX = [MANO_TO_UMETRACK[m] for m in MANO_IDX]


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def axisangle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) → rotation matrix (3, 3)."""
    K = torch.zeros(3, 3, device=aa.device)
    K[0, 1] = -aa[2]; K[0, 2] = aa[1]
    K[1, 0] = aa[2]; K[1, 2] = -aa[0]
    K[2, 0] = -aa[1]; K[2, 1] = aa[0]
    return torch.linalg.matrix_exp(K)


def chamfer_sym(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff_ab = (a.unsqueeze(1) - b.unsqueeze(0))
    diff_ba = (b.unsqueeze(1) - a.unsqueeze(0))
    return (diff_ab ** 2).sum(-1).min(dim=1).values.mean() + \
           (diff_ba ** 2).sum(-1).min(dim=1).values.mean()


def landmark_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# FK helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mano_fk(mano_layer, pose, beta, device, no_grad=True):
    """MANO forward kinematics. Returns joints (21,3) mm, verts (778,3) mm."""
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out = mano_layer(pose, beta)
    return out.joints[0] * 1000.0, out.verts[0] * 1000.0


def umetrack_fk(hand_model, angles_22, device):
    """UmeTrack FK. Returns landmarks (21,3), mesh verts (788,3)."""
    from emg2pose.kinematics import apply_to_hand_model, broadcast_hand_model_to
    from emg2pose.UmeTrack.lib.common.hand_skinning import (
        _get_skinned_vertices, _hand_skinning_transform, _lbs, skin_landmarks,
    )
    hm = broadcast_hand_model_to(hand_model, (1,))
    hm = apply_to_hand_model(hm, lambda t: t.float())
    wrist_tf = torch.eye(4, device=device).unsqueeze(0)
    a = angles_22.reshape(1, -1)
    lm = skin_landmarks(hm, a[:, :20], wrist_tf)[0]
    skin_xfs = _hand_skinning_transform(
        hm.joint_rotation_axes.reshape(1, -1, 3),
        hm.joint_rest_positions.reshape(1, -1, 3),
        a, wrist_tf,
    )
    w = hm.dense_bone_weights.reshape(1, -1, 17)
    mr = hm.mesh_vertices.reshape(1, -1, 3)
    v = _get_skinned_vertices(mr, w)
    mesh = _lbs(skin_xfs, v)[..., :3][0]
    return lm, mesh


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def save_combined_glb(path, verts_a, faces_a, verts_b, faces_b,
                      color_a=(70, 130, 180, 200), color_b=(220, 80, 60, 200)):
    import trimesh
    ma = trimesh.Trimesh(vertices=verts_a, faces=faces_a[:, ::-1], process=False)
    ma.visual.vertex_colors = np.tile(color_a, (len(verts_a), 1)).astype(np.uint8)
    mb = trimesh.Trimesh(vertices=verts_b, faces=faces_b[:, ::-1], process=False)
    mb.visual.vertex_colors = np.tile(color_b, (len(verts_b), 1)).astype(np.uint8)
    trimesh.Scene([ma, mb]).export(str(path))


def save_landmark_glb(path, pts_a, pts_b,
                      color_a=(70, 130, 180, 255), color_b=(220, 80, 60, 255)):
    import trimesh
    from trimesh.transformations import rotation_matrix

    BONES = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6), (6, 7),
        (0, 8), (8, 9), (9, 10), (10, 11),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]
    radius = 1.5
    meshes = []

    def _add_skeleton(pts, color):
        for p in pts:
            s = trimesh.primitives.Sphere(center=p, radius=radius)
            s.visual.vertex_colors = np.tile(color, (len(s.vertices), 1)).astype(np.uint8)
            meshes.append(s)
        for i, j in BONES:
            pa, pb = pts[i], pts[j]
            d = pb - pa
            dist = np.linalg.norm(d)
            if dist < 1e-6:
                continue
            mid = (pa + pb) / 2.0
            ud = d / dist
            ref = np.array([1.0, 0, 0]) if abs(ud[2]) > 0.999 else np.array([0, 0, 1.0])
            axis = np.cross(ref, ud)
            axis_n = np.linalg.norm(axis)
            if axis_n > 1e-8:
                R = rotation_matrix(np.arcsin(axis_n), axis / axis_n)[:3, :3]
            else:
                R = np.eye(3)
            cyl = trimesh.primitives.Cylinder(radius=radius * 0.3, height=dist)
            verts = cyl.vertices @ R.T + mid
            m = trimesh.Trimesh(vertices=verts, faces=cyl.faces, process=False)
            m.visual.vertex_colors = np.tile(color, (len(verts), 1)).astype(np.uint8)
            meshes.append(m)

    _add_skeleton(pts_a, color_a)
    _add_skeleton(pts_b, color_b)
    trimesh.Scene(meshes).export(str(path))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap-root", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--loss-type", default="landmark",
                        choices=["chamfer", "landmark", "both"])
    parser.add_argument("--lm-weight", type=float, default=0.001,
                        help="Landmark loss weight (only for --loss-type both)")
    parser.add_argument("--opt-global-orient", action="store_true",
                        help="Optimize global rotation (3-DoF) in alignment transform.")
    parser.add_argument("--opt-trans", action="store_true",
                        help="Optimize translation (3-DoF) in alignment transform.")
    parser.add_argument("--output", type=Path, default=Path("ik_test_output"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    side = args.hand
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load MANO pose from memmap ────────────────────────────────────────

    with open(args.memmap_root / "manifest.json") as f:
        manifest = json.load(f)
    meta = np.load(args.memmap_root / "metadata.npz", allow_pickle=False)
    pose_info = manifest["fields"][f"generated_mano_{side}_pose"]
    pose_mm = np.memmap(args.memmap_root / pose_info["filename"],
                        dtype=np.dtype(pose_info["dtype"]), mode="r",
                        shape=tuple(pose_info["shape"]))
    fi = args.frame
    offset = int(meta.get("episode_start_idx", np.zeros(1, dtype=np.int64))[0])
    pose_np = pose_mm[offset + fi:offset + fi + 1].astype(np.float32).copy()
    print(f"Frame {fi}/{pose_mm.shape[0]}, hand={side}")

    # ── MANO setup ────────────────────────────────────────────────────────

    mano_layer = ManoLayer(
        rot_mode="axisang", side="right",
        mano_assets_root=str(MANO_ASSETS_ROOT),
        use_pca=False, flat_hand_mean=False,
    ).to(device)
    pose_t = torch.from_numpy(pose_np).to(device)
    if torch.isnan(pose_t).any():
        raise RuntimeError(f"Frame {fi} has NaN MANO pose (WiLoR detection failure), pick another frame.")
    # WiLoR global_orient is replaced by the alignment transform's rotation.
    # Zero it so MANO FK outputs a canonical-orientation hand.
    pose_t[:, :3] = 0.0
    mano_faces = mano_layer.th_faces.cpu().numpy()

    # ── UmeTrack setup ────────────────────────────────────────────────────

    from emg2pose.kinematics import apply_to_hand_model, load_default_hand_model
    hand_model = load_default_hand_model()
    hand_model = apply_to_hand_model(hand_model, lambda t: t.float().to(device))
    faces_u = hand_model.mesh_triangles.cpu().numpy()

    # Joint limits from UmeTrack model spec.
    json_path = (Path(__file__).resolve().parent.parent.parent
                 / "emg2pose" / "UmeTrack" / "dataset" / "generic_hand_model.json")
    with open(json_path) as f:
        limits = np.array(json.load(f)["joint_limits"][:20], dtype=np.float32)
    angle_min = torch.from_numpy(limits[:, 0]).to(device)
    angle_max = torch.from_numpy(limits[:, 1]).to(device)
    angle_range = angle_max - angle_min

    # UmeTrack rest pose (zero angles).
    with torch.no_grad():
        ut_rest_lm, ut_rest_mesh = umetrack_fk(
            hand_model, torch.zeros(22, device=device), device)

    # ── Alignment transform ───────────────────────────────────────────────
    # transform(pt) = ALIGN_SCALE * (pt @ FLIP) @ R(opt_orient) + trans_init + opt_trans

    flip_t = torch.from_numpy(FLIP_MATRIX).float().to(device)
    trans_init = torch.tensor(ALIGN_TRANS.tolist(), dtype=torch.float32, device=device)
    print(f"Alignment: scale={ALIGN_SCALE:.4f}, trans={ALIGN_TRANS}")

    # ── Optimizable parameters ────────────────────────────────────────────

    raw_angles = torch.zeros(20, device=device, requires_grad=True)
    opt_params = [raw_angles]

    if args.opt_global_orient:
        global_orient_raw = torch.zeros(3, device=device, requires_grad=True)
        opt_params.append(global_orient_raw)
    else:
        global_orient_raw = torch.zeros(3, device=device)

    if args.opt_trans:
        trans_raw = torch.zeros(3, device=device, requires_grad=True)
        opt_params.append(trans_raw)
    else:
        trans_raw = torch.zeros(3, device=device)

    def current_R() -> torch.Tensor:
        return axisangle_to_rotmat(global_orient_raw)

    def current_trans() -> torch.Tensor:
        return trans_init + 10.0 * trans_raw

    def transform_mano(pts: torch.Tensor) -> torch.Tensor:
        """Flip X → scale → rotate(opt) → translate."""
        return ALIGN_SCALE * (pts @ flip_t.T) @ current_R().T + current_trans()

    # ── Save initial GLB (before optimization) ────────────────────────────

    with torch.no_grad():
        _, mv_init = mano_fk(mano_layer, pose_t, torch.zeros(1, 10, device=device), device)
        mv_aligned = transform_mano(mv_init).cpu().numpy()
    ut_rest_np = ut_rest_mesh.cpu().numpy()
    save_combined_glb(args.output / f"initial_compare_{side}.glb",
                      mv_aligned, mano_faces, ut_rest_np, faces_u)
    print(f"Saved initial_compare_{side}.glb  (blue=MANO, red=UmeTrack rest)")

    # ── L-BFGS optimizer ──────────────────────────────────────────────────

    optimizer = torch.optim.LBFGS(
        opt_params, lr=0.1, max_iter=50,
        line_search_fn="strong_wolfe", history_size=30,
        tolerance_grad=1e-9, tolerance_change=1e-11,
    )

    use_ch = args.loss_type in ("chamfer", "both")
    use_lm = args.loss_type in ("landmark", "both")

    def closure():
        optimizer.zero_grad()
        angles_20 = angle_min + angle_range * torch.sigmoid(raw_angles)
        a22 = torch.cat([angles_20, torch.zeros(2, device=device)])

        mano_j, mano_v = mano_fk(
            mano_layer, pose_t, torch.zeros(1, 10, device=device), device, no_grad=False)

        pred_lm, pred_mesh = umetrack_fk(hand_model, a22, device)

        loss = torch.tensor(0.0, device=device)
        if use_ch:
            loss = loss + chamfer_sym(pred_mesh, transform_mano(mano_v))
        if use_lm:
            target_lm = transform_mano(mano_j)[MANO_IDX]
            loss = loss + args.lm_weight * landmark_l2(pred_lm[UMETRACK_IDX], target_lm)
        loss.backward()
        return loss

    t0 = time.time()
    prev_loss = float("inf")
    best_loss = float("inf")
    stale_steps = 0
    patience = 30
    for step in range(args.max_iter):
        loss = optimizer.step(closure)
        loss_v = loss.item()

        # Early stop if loss flatlines.
        if loss_v < best_loss - 0.0001:
            best_loss = loss_v
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= patience:
            break

        # Periodic L-BFGS restart to clear stale Hessian approximation.
        if step > 0 and step % 30 == 0 and abs(loss_v - prev_loss) < 0.01:
            optimizer = torch.optim.LBFGS(
                opt_params, lr=0.1, max_iter=50,
                line_search_fn="strong_wolfe", history_size=30,
                tolerance_grad=1e-9, tolerance_change=1e-11,
            )
            stale_steps = 0
        prev_loss = loss_v

        if step % 10 == 0 or step == args.max_iter - 1:
            with torch.no_grad():
                ad = (angle_min + angle_range * torch.sigmoid(raw_angles))
                ad_deg = ad.cpu().numpy() * 180 / np.pi
            print(f"  step {step:4d}: loss={loss_v:.4f}  "
                  f"angles=[{ad_deg.min():.1f},{ad_deg.max():.1f}]°  "
                  f"trans={current_trans().detach().cpu().numpy()}  "
                  f"R_norm={current_R().norm().item():.3f}")

    print(f"\nDone in {time.time() - t0:.1f}s")

    # ── Final evaluation ──────────────────────────────────────────────────

    with torch.no_grad():
        fa = angle_min + angle_range * torch.sigmoid(raw_angles)
        a22 = torch.cat([fa, torch.zeros(2, device=device)])
        final_lm, final_ut = umetrack_fk(hand_model, a22, device)
        fj, fv = mano_fk(mano_layer, pose_t, torch.zeros(1, 10, device=device), device)
        final_mano = transform_mano(fv)

    # Per-joint landmark errors.
    target_lm = transform_mano(fj)[MANO_IDX]
    pred_lm_sel = final_lm[UMETRACK_IDX]
    per_joint = (pred_lm_sel - target_lm).norm(dim=-1).detach().cpu().numpy()

    LM_NAMES = [
        "wrist", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    ]
    # Only finger tips + PIP/DIP joints (exclude wrist and MCPs).
    FINGER_MASK = np.array([
        False, False, False, True,
        False, True, True, True,
        False, True, True, True,
        False, True, True, True,
        False, True, True, True,
    ])

    print("\nFinal landmark error (mm) [* = excluded from mean]:")
    for name, d, masked in zip(LM_NAMES, per_joint, ~FINGER_MASK):
        print(f"  {name:20s}: {d:6.2f}{' *' if masked else ''}")
    finger_errs = per_joint[FINGER_MASK]
    print(f"  {'MEAN (fingers)':20s}: {finger_errs.mean():6.2f}")
    print(f"  {'RMS  (fingers)':20s}: {np.sqrt((finger_errs ** 2).mean()):6.2f}")

    chamfer = chamfer_sym(final_mano.detach(), final_ut.detach())
    print(f"\nChamfer distance: {chamfer.item():.2f}  (sqrt: {chamfer.sqrt().item():.2f} mm)")

    # ── Save final GLBs ───────────────────────────────────────────────────

    save_combined_glb(args.output / f"optimized_compare_{side}.glb",
                      final_mano.cpu().numpy(), mano_faces,
                      final_ut.cpu().numpy(), faces_u)
    save_landmark_glb(args.output / f"landmarks_{side}.glb",
                      target_lm.detach().cpu().numpy(),
                      pred_lm_sel.detach().cpu().numpy())
    print(f"\nOutputs in {args.output.resolve()}/")
    print(f"  initial_compare_{side}.glb   — blue=MANO(aligned), red=UmeTrack(rest)")
    print(f"  optimized_compare_{side}.glb — blue=MANO(aligned), red=UmeTrack(optimized)")
    print(f"  landmarks_{side}.glb         — blue=MANO skeleton, red=UmeTrack skeleton")

    # Final angles.
    fa_deg = fa.cpu().numpy() * 180 / np.pi
    ANGLE_NAMES = [
        "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
        "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
        "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
        "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
        "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
    ]
    print("\nFinal joint angles (deg):")
    for n, v in zip(ANGLE_NAMES, fa_deg):
        print(f"  {n:20s}: {v:+7.2f}")


if __name__ == "__main__":
    import argparse
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
