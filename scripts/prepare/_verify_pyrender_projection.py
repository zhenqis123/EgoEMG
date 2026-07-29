"""Verify pyrender can reproduce OpenCV projection.

Key convention differences:
  OpenCV cam: X right, Y down, Z forward  (+Z is forward)
  OpenGL cam: X right, Y up,   Z backward (-Z is forward)

To convert cam-to-world from OpenCV to OpenGL:
  T_W_C_gl = T_W_C_cv @ diag(1, -1, -1, 1)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh

HAND = "right"


def load_mm(manifest, mm_dir, name):
    info = manifest["fields"][name]
    return np.memmap(f"{mm_dir}/{info['filename']}", dtype=np.dtype(info["dtype"]),
                     mode="r", shape=tuple(info["shape"]))


def load_episode_mm(manifest, mm_dir, name):
    info = manifest["episode_fields"][name]
    return np.memmap(f"{mm_dir}/{info['filename']}", dtype=np.dtype(info["dtype"]),
                     mode="r", shape=tuple(info["shape"]))


def get_mano_verts_local(mano, pose_aa, beta, device):
    import torch
    global_orient = torch.zeros(1, 3, dtype=torch.float32, device=device)
    hand_pose = torch.tensor(pose_aa[3:48], dtype=torch.float32, device=device).unsqueeze(0)
    betas = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = mano(global_orient=global_orient, hand_pose=hand_pose, betas=betas)
    return out.vertices[0].cpu().numpy()


def cv_to_gl_pose(T_W_C_cv):
    """Convert camera-to-world from OpenCV to OpenGL convention.

    OpenCV cam: X right, Y down, Z forward
    OpenGL cam: X right, Y up,   Z backward
    """
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return T_W_C_cv @ flip_yz


def project_opencv_nodist(points_world, T_W_C_cv, K):
    """OpenCV pinhole projection (no distortion). T_W_C_cv is camera-to-world in OpenCV."""
    T_C_W = np.linalg.inv(T_W_C_cv)
    R = T_C_W[:3, :3]
    t = T_C_W[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(points_world.astype(np.float64), rvec, t, K, distCoeffs=None)
    proj = proj.reshape(-1, 2).astype(np.float64)
    p_cam = (R @ points_world.T + t).T
    depth_valid = p_cam[:, 2] > 1e-6
    return proj, depth_valid, R, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap_dir", default="data/EgoEMG_memmap")
    parser.add_argument("--data_root", default="data/EgoEMG")
    parser.add_argument("--mano_model_path",
                        default="../WiLoR/mano_data/models")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/pyrender_verify")
    args = parser.parse_args()

    print("=== Pyrender Projection Verification v2 ===\n")

    mm_dir = args.memmap_dir
    manifest = json.load(open(f"{mm_dir}/manifest.json"))
    md = np.load(f"{mm_dir}/metadata.npz", allow_pickle=False)

    cam_tracked_mm = load_mm(manifest, mm_dir, "mocap_webcam_tracked")
    cam_transform_mm = load_mm(manifest, mm_dir, "mocap_webcam_transform")
    mano_pose_mm = load_mm(manifest, mm_dir, f"generated_mano_{HAND}_pose")
    mano_world_mm = load_mm(manifest, mm_dir, f"mocap_mano_{HAND}_world_transform")
    mano_beta_mm = load_episode_mm(manifest, mm_dir, f"generated_mano_{HAND}_beta")
    ep_idx_mm = load_mm(manifest, mm_dir, "episode_index")
    beta_idx_arr = md["episode_beta_idx"]

    with open(Path(args.data_root) / "reprojection_assets" / "GX010023_standard_calibration.json") as f:
        calib = json.load(f)
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    calib_w = int(calib["image_width"])
    calib_h = int(calib["image_height"])

    mano_model = smplx.MANO(
        model_path=str(args.mano_model_path), is_rhand=True,
        flat_hand_mean=False, use_pca=False, num_pca_comps=45,
    ).to(args.device)

    # --- Sample frame ---
    rng = np.random.RandomState(args.seed)
    valid_idx = np.where(cam_tracked_mm == 1)[0]
    global_i = int(rng.choice(valid_idx))
    ep = int(ep_idx_mm[global_i])
    print(f"Sampled: global_i={global_i}, episode={ep}")

    # --- Build world-space vertices ---
    mano_pose = np.asarray(mano_pose_mm[global_i], dtype=np.float64)
    beta_idx = int(beta_idx_arr[ep])
    beta = np.asarray(mano_beta_mm[beta_idx], dtype=np.float64)
    verts_local = get_mano_verts_local(mano_model, mano_pose, beta, args.device)

    t12_world = np.asarray(mano_world_mm[global_i], dtype=np.float64)
    R_world = t12_world[:9].reshape(3, 3)
    t_world = t12_world[9:12]
    verts_world = (R_world @ verts_local.T).T + t_world

    # --- Camera extrinsics (stored as camera-to-world in OpenCV convention) ---
    t12 = np.asarray(cam_transform_mm[global_i], dtype=np.float64)
    T_W_C_cv = np.eye(4, dtype=np.float64)
    T_W_C_cv[:3, :3] = t12[:9].reshape(3, 3)
    T_W_C_cv[:3, 3] = t12[9:12]

    print(f"Vertices: {verts_world.shape}")

    # === Step 1: Verify manual projection with fixed convention ===
    print("\n--- Step 1: Manual projection comparison ---")

    # (a) OpenCV pinhole projection
    proj_cv, valid_cv, R_w2c, t_w2c = project_opencv_nodist(verts_world, T_W_C_cv, K)
    print(f"OpenCV pinhole:     center vertex → px=({proj_cv[0,0]:.1f}, {proj_cv[0,1]:.1f})")

    # (b) pyrender IntrinsicsCamera projection (manual math, with CV→GL conversion)
    P_gl = np.zeros((4, 4), dtype=np.float64)
    P_gl[0, 0] = 2.0 * K[0, 0] / calib_w
    P_gl[1, 1] = 2.0 * K[1, 1] / calib_h
    P_gl[0, 2] = 1.0 - 2.0 * K[0, 2] / calib_w
    P_gl[1, 2] = 2.0 * K[1, 2] / calib_h - 1.0
    P_gl[3, 2] = -1.0

    # CV→GL: X_gl = X_cv, Y_gl = -Y_cv, Z_gl = -Z_cv
    # So R_gl_from_cv = diag(1, -1, -1)
    R_gl_from_cv = np.diag([1.0, -1.0, -1.0])

    # World→Camera in OpenCV, then convert camera coords from CV to GL
    # pts_cv = R_w2c @ pts_w + t_w2c
    # pts_gl = R_gl_from_cv @ pts_cv
    pts_cv = (R_w2c @ verts_world.T + t_w2c).T
    pts_gl = (R_gl_from_cv @ pts_cv.T).T

    N = len(verts_world)
    pts_h = np.concatenate([pts_gl, np.ones((N, 1), dtype=np.float64)], axis=1)
    clip = (P_gl @ pts_h.T).T
    ndc = clip[:, :3] / clip[:, 3:4]

    # NDC → pixels (OpenGL NDC has (0,0) at center, but we map to image coords)
    px_gl = (ndc[:, 0] + 1.0) * 0.5 * calib_w
    py_gl = (1.0 - ndc[:, 1]) * 0.5 * calib_h  # flip Y: GL up → image down

    proj_pyr = np.column_stack([px_gl, py_gl])
    valid_gl = pts_cv[:, 2] > 1e-6  # depth in OpenCV space

    valid = valid_cv & valid_gl
    diff = np.abs(proj_cv[valid] - proj_pyr[valid])

    print(f"OpenCV vs pyrender manual:")
    print(f"  Mean  abs error (px): x={diff[:,0].mean():.4f}  y={diff[:,1].mean():.4f}")
    print(f"  Max   abs error (px): x={diff[:,0].max():.4f}  y={diff[:,1].max():.4f}")
    print(f"  Median abs error (px): x={np.median(diff[:,0]):.4f}  y={np.median(diff[:,1]):.4f}")

    ok = np.median(diff[:, 0]) < 1.0 and np.median(diff[:, 1]) < 1.0
    print(f"  → {'PASS' if ok else 'FAIL'}")

    # === Step 2: End-to-end pyrender offscreen render ===
    print("\n--- Step 2: End-to-end pyrender offscreen render ---")

    faces = mano_model.faces.astype(np.int64)
    mesh = trimesh.Trimesh(vertices=verts_world.astype(np.float32), faces=faces)
    mesh.visual.vertex_colors = [0, 255, 0, 255]

    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyr_mesh)

    # Camera-to-world in OpenGL convention
    T_W_C_gl = cv_to_gl_pose(T_W_C_cv)

    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        znear=0.01, zfar=100.0,
    )
    scene.add(cam, pose=T_W_C_gl)

    # Add point light for shading
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=T_W_C_gl)

    renderer = pyrender.OffscreenRenderer(calib_w, calib_h)
    color, depth = renderer.render(scene)
    renderer.delete()

    print(f"Rendered: color={color.shape}, depth={depth.shape}")
    rendered_mask = depth > 0
    n_rendered = rendered_mask.sum()
    print(f"Rendered pixels: {n_rendered} / {rendered_mask.size} ({100*n_rendered/rendered_mask.size:.1f}%)")

    # === Step 3: Compare bounding boxes (vertex projection vs rendered silhouette) ===
    print("\n--- Step 3: Bounding box comparison ---")

    cv_valid_pts = proj_cv[valid_cv]
    cv_bbox = [cv_valid_pts[:, 0].min(), cv_valid_pts[:, 1].min(),
               cv_valid_pts[:, 0].max(), cv_valid_pts[:, 1].max()]
    print(f"OpenCV vertex bbox: x=[{cv_bbox[0]:.0f}, {cv_bbox[2]:.0f}] y=[{cv_bbox[1]:.0f}, {cv_bbox[3]:.0f}]")

    if n_rendered > 0:
        ys, xs = np.where(rendered_mask)
        pyr_bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        print(f"Pyrender silhouette bbox: x=[{pyr_bbox[0]}, {pyr_bbox[2]}] y=[{pyr_bbox[1]}, {pyr_bbox[3]}]")

        err = [abs(cv_bbox[i] - pyr_bbox[i]) for i in range(4)]
        print(f"Bbox edge errors (px): left={err[0]:.0f} top={err[1]:.0f} right={err[2]:.0f} bottom={err[3]:.0f}")
        max_err = max(err)
        # Silhouette bbox should be slightly inside vertex bbox (rasterization),
        # allow up to 5px tolerance per edge at 3840x3360 resolution
        ok2 = max_err < 5
        print(f"Max edge error: {max_err:.1f} px → {'PASS' if ok2 else 'FAIL'}")
    else:
        print("No pixels rendered — check camera pose / scene setup")

    # Save debug render
    out_path = Path(args.output) / "pyrender_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print(f"\nSaved debug render to {out_path}")

    print("\n=== Verification complete ===")


if __name__ == "__main__":
    main()
