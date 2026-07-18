#!/usr/bin/env python3
"""Compare mesh projections on video frames.

Three columns: saved MANO FK | WiLoR direct pred_verts | UmeTrack FK
"""

from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path
import cv2, numpy as np, torch, torchvision.ops as tv_ops

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# UmeTrack FK
_VIZ_SPEC = importlib.util.spec_from_file_location(
    "vizm", str(_PROJECT_ROOT / "emg2pose" / "visualization.py"))
_VIZ_MOD = importlib.util.module_from_spec(_VIZ_SPEC)
_VIZ_SPEC.loader.exec_module(_VIZ_MOD)
skin_mesh_from_angles = _VIZ_MOD.skin_mesh_from_angles

# MANO
MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")
if str(MANOTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MANOTORCH_ROOT))
from manotorch.manolayer import ManoLayer
MANO_ASSETS = "/home/xiziheng/develop/HandVQVAE/assets/mano"

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from wilor_mini.utils import utils as wilor_utils

SESSIONS = {
    "sess_20260530_140912": {"ep": 4, "name": "xzh_bare"},
    "sess_20260530_143229": {"ep": 5, "name": "dyb_bare"},
    "sess_20260531_142701": {"ep": 6, "name": "xzh_bare_2"},
    "sess_20260531_150809": {"ep": 7, "name": "dyb_bare_2"},
}
DATA_ROOT = Path("/home/xiziheng/develop/emg2pose/data")
FOCAL_LENGTH = 5000
IMAGE_SIZE = 256


def project(verts, pred_cam, center, bbox_size, img_size):
    """Perspective projection (same as sample_wilor_forward_session_mesh.py)."""
    scaled_fl = FOCAL_LENGTH / IMAGE_SIZE * img_size.max()
    cam_t = wilor_utils.cam_crop_to_full(
        pred_cam[None], center[None], bbox_size, img_size[None], scaled_fl)[0]
    v_cam = verts + cam_t[None, :]
    z = v_cam[:, 2]
    w, h = int(img_size[0]), int(img_size[1])
    pts = np.full((v_cam.shape[0], 2), -10000.0, dtype=np.float32)
    ok = z > 1e-6
    pts[ok, 0] = v_cam[ok, 0] / z[ok] * scaled_fl + w / 2.0
    pts[ok, 1] = v_cam[ok, 1] / z[ok] * scaled_fl + h / 2.0
    return pts, ok


def draw(img, pts, in_front, faces, color):
    """Draw wireframe mesh edges."""
    h, w = img.shape[:2]
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (in_front[i0] and in_front[i1] and in_front[i2]):
            continue
        p = pts[[i0, i1, i2]]
        if np.any(~np.isfinite(p)): continue
        p = np.round(p).astype(np.int32)
        if (p[:, 0] < -1000).any() or (p[:, 0] > w + 1000).any() or \
           (p[:, 1] < -1000).any() or (p[:, 1] > h + 1000).any():
            continue
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            cv2.line(img, tuple(p[a]), tuple(p[b]), color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/mano_vs_umetrack"))
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Init MANO FK
    mano_fk = ManoLayer(rot_mode="axisang", side="right",
                        mano_assets_root=MANO_ASSETS, use_pca=False, flat_hand_mean=True)
    mano_faces = mano_fk.th_faces.numpy()

    # Init WiLoR pipeline
    pipe = WiLorHandPose3dEstimationPipeline(
        device=torch.device("cuda:0"), dtype=torch.float16,
        wilor_pretrained_dir="data/pretrained_models")
    model = pipe.wilor_model

    # Load merged memmap
    incre_root = DATA_ROOT / "EgoEMG_incre" / "data_right_merged"
    with open(incre_root / "manifest.json") as f:
        manifest = json.load(f)
    md = np.load(incre_root / "metadata.npz", allow_pickle=True)
    ep_starts = md["episode_start_idx"].astype(np.int64)
    ep_ends = md["episode_end_idx"].astype(np.int64)
    ja_mm = np.memmap(incre_root / "generated_joint_angles_right.dat",
                      dtype=np.float32, mode="r",
                      shape=tuple(manifest["fields"]["generated_joint_angles_right"]["shape"]))
    ts_mm = np.memmap(incre_root / "timestamp_us.dat",
                      dtype=np.int64, mode="r",
                      shape=tuple(manifest["fields"]["timestamp_us"]["shape"]))

    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    for sess_name, info in SESSIONS.items():
        session_dir = DATA_ROOT / sess_name
        if not session_dir.exists(): continue
        wdir = session_dir / "wilor_mano"

        go = np.load(wdir / "mano_global_orient.npy").astype(np.float32)
        hp = np.load(wdir / "mano_hand_pose.npy").astype(np.float32)
        betas = np.load(wdir / "mano_betas.npy").astype(np.float32)
        pred_cams = np.load(wdir / "pred_cam.npy").astype(np.float32)
        wts = np.load(wdir / "timestamps_us.npy").astype(np.int64)
        wvalid = np.load(wdir / "valid.npy").astype(bool)
        bbox_c = np.load(wdir / "bbox_center.npy").astype(np.float32) if (wdir / "bbox_center.npy").exists() else None
        bbox_s = np.load(wdir / "bbox_size.npy").astype(np.float32) if (wdir / "bbox_size.npy").exists() else None

        zed_dirs = list(session_dir.glob("ZED_*"))
        if not zed_dirs: continue
        video_path = zed_dirs[0] / "rgb.mkv"

        valid_vid = np.where(wvalid)[0]
        n = min(args.n_samples, len(valid_vid))
        if n == 0: continue
        sampled = sorted(rng.choice(valid_vid, size=n, replace=False))

        ep_idx, ep_start, ep_end = info["ep"], int(ep_starts[info["ep"]]), int(ep_ends[info["ep"]])
        out_dir = args.output / f"ep{ep_idx:02d}_{info['name']}"
        out_dir.mkdir(parents=True, exist_ok=True)

        import decord
        vr = decord.VideoReader(str(video_path))

        for vid_idx in sampled:
            frame_rgb = vr[vid_idx].asnumpy()
            h, w = frame_rgb.shape[:2]
            img_size = np.array([w, h], dtype=np.float32)

            pred_cam = pred_cams[vid_idx].copy()
            center = bbox_c[vid_idx] if bbox_c is not None else img_size / 2
            bbox_size_val = float(bbox_s[vid_idx]) if bbox_s is not None else float(max(w, h)) * 1.2

            # ── 1. Saved MANO FK projection ──
            pose = np.concatenate([go[vid_idx:vid_idx+1], hp[vid_idx:vid_idx+1]], axis=1)
            with torch.no_grad():
                mout = mano_fk(torch.from_numpy(pose), torch.from_numpy(betas[vid_idx:vid_idx+1]))
            fk_verts = mout.verts[0].numpy().copy()  # meters
            fk_pts, fk_front = project(fk_verts, pred_cam, center, bbox_size_val, img_size)
            # fk_front = np.ones(len(fk_verts), dtype=bool)  # UmeTrack rest pose vertex count

            # ── 2. WiLoR direct forward ──
            ft = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device="cuda:0", dtype=torch.float32)
            half = bbox_size_val / 2
            roi = [[0, center[0]-half, center[1]-half, center[0]+half, center[1]+half]]
            boxes_t = torch.tensor(roi, device="cuda:0", dtype=torch.float32)
            patch = tv_ops.roi_align(ft.unsqueeze(0), boxes_t, (256, 256), 1.0, True)
            patch_nhwc = patch.permute(0, 2, 3, 1).to(dtype=torch.float16)
            with torch.no_grad():
                wout = model(patch_nhwc)
            w_verts = wout["pred_vertices"][0].cpu().float().numpy()
            w_pts, w_front = project(w_verts, pred_cam, center, bbox_size_val, img_size)

            # ── 3. UmeTrack FK ──
            vid_ts = int(wts[vid_idx])
            ep_ts_arr = np.asarray(ts_mm[ep_start:ep_end + 1])
            emg_local = min(np.searchsorted(ep_ts_arr, vid_ts), len(ep_ts_arr) - 1)
            emg_idx = ep_start + emg_local
            ut_angles = np.asarray(ja_mm[emg_idx], dtype=np.float32)
            ut_v, ut_f = skin_mesh_from_angles(joint_angles=ut_angles, flip=False)
            # Align UmeTrack → WiLoR meters space (center at origin)
            ut_v = (ut_v - ut_v.mean(axis=0)) / np.median(ut_v.max(axis=0) - ut_v.min(axis=0)) * 0.05
            ut_pts, ut_front = project(ut_v, pred_cam, center, bbox_size_val, img_size)

            # ── Draw ──
            f_saved = frame_rgb.copy()
            draw(f_saved, fk_pts, fk_front, mano_faces, (255, 80, 80))
            cv2.putText(f_saved, "Saved MANO FK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)

            f_wilor = frame_rgb.copy()
            draw(f_wilor, w_pts, w_front, mano_faces, (255, 200, 80))
            cv2.putText(f_wilor, "WiLoR direct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 80), 2)

            f_ut = frame_rgb.copy()
            draw(f_ut, ut_pts, ut_front, ut_f, (80, 255, 80))
            cv2.putText(f_ut, "UmeTrack FK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 80), 2)

            f_all = frame_rgb.copy()
            draw(f_all, fk_pts, fk_front, mano_faces, (255, 80, 80))
            draw(f_all, w_pts, w_front, mano_faces, (255, 200, 80))
            draw(f_all, ut_pts, ut_front, ut_f, (80, 255, 80))
            cv2.putText(f_all, "Red=SavFK Yellow=WiLoR Green=UT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for name, img in [("saved_fk", f_saved), ("wilor_direct", f_wilor),
                               ("umetrack", f_ut), ("all", f_all)]:
                cv2.imwrite(str(out_dir / f"f{vid_idx:05d}_{name}.jpg"),
                           cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"  vid={vid_idx}: saved")

    print(f"\nDone. {args.output}")


if __name__ == "__main__":
    main()
