"""Evaluate per-sample occlusion vs fusion gain across the entire test set.

For each test sample, computes:
- Self-occlusion score (area-weighted, from GT MANO mesh via z-buffer)
- Vision MAE, EMG MAE, Fusion MAE (degrees)
- Delta MAE (vision - fusion)

Output: CSV with one row per sample.

Usage:
    python scripts/eval/evaluate_occlusion_fusion.py \
        --config-name experiment/fusion/vision_resnet_small_emgfusion_center \
        --checkpoint /path/to/fusion.ckpt \
        --data-location /path/to/EgoEMG_v2_memmap \
        --video-root /path/to/EgoEMG_allintra \
        --output-csv occlusion_fusion_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# ── Project imports ──────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from egoemg.lightning import EmgPredictionModule
from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from egoemg.models.modules.emgformer import Emg2PoseFormer
from egoemg.models.modules.resnet_vision import ResNetVisionPose
from egoemg.models.modules.vit_vision import VisionViTPose
from egoemg.occlusion import compute_self_occlusion

import smplx

CONFIG_DIR = str(_PROJECT_DIR / "config")
MANO_MODEL_PATH = "../WiLoR/mano_data/models"

warnings.filterwarnings(
    "ignore", message="The given NumPy array is not writable", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="enable_nested_tensor is True", category=UserWarning
)

MIRROR_X_3 = np.array([-1.0, 1.0, 1.0], dtype=np.float32)

# ── Caches ───────────────────────────────────────────────────────────────────
_MM_CACHE: dict[str, np.memmap] = {}
_MANIFEST: Optional[dict] = None
_MANO_LAYER = None
_EPISODE_INTRINSICS_CACHE: dict[int, tuple] = {}  # ep_idx -> (K_use, video_w, video_h)


def _load_manifest(memmap_dir: Path) -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        with open(memmap_dir / "manifest.json") as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def _load_mm(memmap_dir: Path, name: str) -> np.memmap:
    if name not in _MM_CACHE:
        mf = _load_manifest(memmap_dir)
        info = mf["fields"][name]
        _MM_CACHE[name] = np.memmap(
            memmap_dir / info["filename"],
            dtype=np.dtype(info["dtype"]),
            mode="r",
            shape=tuple(info["shape"]),
        )
    return _MM_CACHE[name]


def _get_mano_layer(device="cpu"):
    global _MANO_LAYER
    if _MANO_LAYER is None:
        _MANO_LAYER = smplx.MANO(
            model_path=MANO_MODEL_PATH,
            is_rhand=True,
            flat_hand_mean=False,
            use_pca=False,
            num_pca_comps=45,
        ).to(device)
    return _MANO_LAYER


def get_mano_verts_world(
    mano_pose: np.ndarray,
    beta: np.ndarray,
    R_world: np.ndarray,
    t_world: np.ndarray,
    flip: bool,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MANO vertices in world space.

    Returns (verts_world, faces) or (None, None) on failure.
    """
    if not (np.isfinite(mano_pose).all() and np.abs(mano_pose).sum() > 0):
        return None, None

    try:
        mano_layer = _get_mano_layer(str(device))
        global_orient = torch.zeros(1, 3, dtype=torch.float32, device=device)
        hand_pose_aa = mano_pose[3:48].astype(np.float32)
        hp_t = torch.tensor(hand_pose_aa, dtype=torch.float32, device=device).unsqueeze(0)
        betas_t = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out_data = mano_layer(
                global_orient=global_orient, hand_pose=hp_t, betas=betas_t,
            )
        verts_local = out_data.vertices[0].cpu().numpy()
        faces = mano_layer.faces.copy()

        if flip:
            verts_local = verts_local * MIRROR_X_3
            faces = faces[:, [0, 2, 1]]

        verts_world = (R_world @ verts_local.T).T + t_world
        return verts_world.astype(np.float64), faces
    except Exception:
        return None, None


def get_episode_intrinsics(
    ep_idx: int,
    memmap_dir: Path,
    video_root: Path,
    dataset: EgoEmgMemmapDataset,
    calib: dict,
) -> tuple[np.ndarray, int, int]:
    """Get (K_use, video_w, video_h) for an episode, with caching."""
    if ep_idx in _EPISODE_INTRINSICS_CACHE:
        return _EPISODE_INTRINSICS_CACHE[ep_idx]

    # Read first frame of the episode to get dimensions
    raw_video_rel = dataset._episode_webcam_video_path[ep_idx]
    if isinstance(raw_video_rel, (bytes, np.bytes_)):
        raw_video_rel = raw_video_rel.decode("utf-8").rstrip("\x00")
    video_path = str(video_root / str(raw_video_rel).replace(".mp4", "_allintra.mp4"))

    try:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        frame_rgb = vr[0].asnumpy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        video_h, video_w = frame_bgr.shape[:2]
    except Exception:
        # Fallback: assume 1920x1080
        video_w, video_h = 1920, 1080

    K_raw = np.asarray(calib["camera_matrix"], dtype=np.float64)
    dist_raw = np.asarray(calib["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    calib_w = int(calib["image_width"])
    calib_h = int(calib["image_height"])

    # Build K for video frame dimensions (simple scaling, matching
    # compute_video_intrinsics in visualize_egoemg_mesh.py).
    # GoPro 8x7 crop: active_x0 = (video_w - video_h * calib_w / calib_h) / 2
    calib_aspect = calib_w / calib_h
    video_aspect = video_w / video_h
    if abs(video_aspect - calib_aspect) < 0.02:
        active_x0 = 0.0
        active_x1 = float(video_w)
    else:
        active_h = float(video_h)
        active_w = active_h * calib_aspect
        active_x0 = (video_w - active_w) / 2.0
        active_x1 = active_x0 + active_w

    K_vid = np.eye(3, dtype=np.float64)
    K_vid[0, 0] = K_raw[0, 0] * (active_x1 - active_x0) / calib_w
    K_vid[1, 1] = K_raw[1, 1] * video_h / calib_h
    K_vid[0, 2] = K_raw[0, 2] * (active_x1 - active_x0) / calib_w + active_x0
    K_vid[1, 2] = K_raw[1, 2] * video_h / calib_h

    result = (K_vid, video_w, video_h)
    _EPISODE_INTRINSICS_CACHE[ep_idx] = result
    return result


# ── Collate (matching find_fusion_wins.py) ────────────────────────────────────

class _DatasetWithMeta(torch.utils.data.Dataset):
    """Wrapper that adds ep_idx, center_idx, and abs_idx to each sample."""

    def __init__(self, base: EgoEmgMemmapDataset, indices: Optional[list[int]] = None):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx] if self.indices is not None else idx
        sample = self.base[real_idx]
        ep_idx, center_idx = self.base._resolve_index_to_center(real_idx)
        sample["_ep_idx"] = ep_idx
        sample["_center_idx"] = center_idx
        sample["_abs_idx"] = real_idx
        return sample


def _collate_fusion(batch: list[dict]) -> dict:
    from torch.utils.data._utils.collate import default_collate

    emg_batch = []
    for sample in batch:
        emg = sample["emg"]
        if isinstance(emg, np.ndarray):
            emg = torch.as_tensor(emg, dtype=torch.float32)
        ja = sample.get("joint_angles")
        if isinstance(ja, np.ndarray):
            ja = torch.as_tensor(ja, dtype=torch.float32)
        mask = sample.get("label_valid_mask")
        if isinstance(mask, np.ndarray):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        vf = sample.get("vision_features")
        if isinstance(vf, np.ndarray):
            vf = torch.as_tensor(vf, dtype=torch.float32)
        vf_mask = sample.get("vision_valid_mask")
        if isinstance(vf_mask, np.ndarray):
            vf_mask = torch.as_tensor(vf_mask, dtype=torch.bool)
        vi = sample.get("vision_img")
        if isinstance(vi, np.ndarray):
            vi = torch.as_tensor(vi, dtype=torch.float32)

        item = {"emg": emg, "joint_angles": ja, "label_valid_mask": mask}
        if vf is not None:
            item["vision_features"] = vf
        if vf_mask is not None:
            item["vision_valid_mask"] = vf_mask
        if vi is not None:
            item["vision_img"] = vi
        for key in ["_ep_idx", "_center_idx", "_abs_idx", "target_hand",
                     "vision_frame_indices"]:
            if key in sample:
                item[key] = sample[key]
        emg_batch.append(item)

    return default_collate(emg_batch)


# ── Model loading ────────────────────────────────────────────────────────────

def load_module(ckpt_path: str, config) -> EmgPredictionModule:
    module = EmgPredictionModule(
        module_conf=config.module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.get("lr_scheduler"),
        loss_weights=config.get("loss_weights", {}),
        datamodule=config.get("datamodule"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    module.eval()
    return module


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_all_samples(
    module: EmgPredictionModule,
    dataloader: DataLoader,
    device: torch.device,
    inner_model,
    memmap_dir: Path,
    video_root: Path,
    dataset: EgoEmgMemmapDataset,
    calib: dict,
    emg_model: Emg2PoseFormer | None = None,
    vis_model: ResNetVisionPose | VisionViTPose | None = None,
) -> list[dict]:
    """Evaluate all samples: occlusion + vision/EMG/fusion MAE."""
    module.to(device)
    results: list[dict] = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            # ── Fusion forward ────────────────────────────────────────────
            out = module.forward(batch_gpu)
            if isinstance(out, tuple):
                preds, targets, mask = out
            else:
                preds = out
                targets = batch_gpu["joint_angles"]
                mask = batch_gpu["label_valid_mask"]

            preds_fusion = preds.squeeze(-1)
            if targets.ndim == 3:
                targets = targets.squeeze(-1)

            # ── Standalone vision ─────────────────────────────────────────
            if vis_model is not None and "vision_img" in batch_gpu:
                vis_out = vis_model(batch_gpu)
                if isinstance(vis_out, tuple):
                    y_v_all = vis_out[0].squeeze(-1)
                else:
                    y_v_all = vis_out.squeeze(-1)
            else:
                y_v_all = torch.zeros(
                    preds_fusion.shape[0], preds_fusion.shape[1], device=device,
                )

            # ── Standalone EMG ───────────────────────────────────────────
            if emg_model is not None:
                emg_preds_full = emg_model(batch_gpu)
                if isinstance(emg_preds_full, tuple):
                    emg_preds_full = emg_preds_full[0]
                t_center = emg_preds_full.shape[-1] // 2
                y_emg_all = emg_preds_full[:, :, t_center]
            else:
                y_emg_all = torch.zeros(
                    preds_fusion.shape[0], preds_fusion.shape[1], device=device,
                )

        ep_idx_list = batch.get("_ep_idx")
        center_idx_list = batch.get("_center_idx")
        target_hands = batch.get("target_hand")

        for i in range(preds_fusion.shape[0]):
            mae_vision = float((y_v_all[i] - targets[i]).abs().mean())
            mae_fusion = float((preds_fusion[i] - targets[i]).abs().mean())
            mae_emg = float((y_emg_all[i] - targets[i]).abs().mean())

            ep_idx = int(ep_idx_list[i]) if isinstance(ep_idx_list, torch.Tensor) else ep_idx_list[i]
            center_idx = int(center_idx_list[i]) if isinstance(center_idx_list, torch.Tensor) else center_idx_list[i]
            hand = target_hands[i] if isinstance(target_hands, list) else str(target_hands[i])

            results.append({
                "ep_idx": ep_idx,
                "center_idx": center_idx,
                "hand": hand,
                "mae_vision_rad": mae_vision,
                "mae_emg_rad": mae_emg,
                "mae_fusion_rad": mae_fusion,
            })

    # ── Compute occlusion per sample ────────────────────────────────────────
    tracked_mm = _load_mm(memmap_dir, "mocap_webcam_tracked")
    stale_mm = _load_mm(memmap_dir, "image_webcam_stale")
    cam_transform_mm = _load_mm(memmap_dir, "mocap_webcam_transform")

    for r in tqdm(results, desc="Occlusion"):
        ci = r["center_idx"]
        hand = r["hand"]
        flip = (hand == "left")
        ep = r["ep_idx"]

        # Mocap tracking flags
        r["mocap_tracked"] = bool(tracked_mm[ci])
        r["mocap_stale"] = bool(stale_mm[ci])

        # Check marker validity
        try:
            valid_mm = _load_mm(memmap_dir, f"mocap_{hand}_valid")
            r["markers_valid"] = bool(valid_mm[ci].any())
        except Exception:
            r["markers_valid"] = False

        # Camera transform
        try:
            t12 = np.asarray(cam_transform_mm[ci], dtype=np.float64)
            T_W_C = np.eye(4, dtype=np.float64)
            T_W_C[:3, :3] = t12[:9].reshape(3, 3)
            T_W_C[:3, 3] = t12[9:12]
        except Exception:
            r.update({"occlusion_score": np.nan, "visible_ratio": np.nan})
            continue

        # MANO mesh in world space
        try:
            mano_pose_mm = _load_mm(memmap_dir, f"generated_mano_{hand}_pose")
            mano_world_mm = _load_mm(memmap_dir, f"mocap_mano_{hand}_world_transform")
            mano_pose = np.asarray(mano_pose_mm[ci], dtype=np.float64)

            t12_world = np.asarray(mano_world_mm[ci], dtype=np.float64)
            R_world = t12_world[:9].reshape(3, 3)
            t_world = t12_world[9:12]

            # Beta is episode-level
            try:
                md = np.load(memmap_dir / "metadata.npz", allow_pickle=False)
                beta_arr = md[f"generated_mano_{hand}_beta"][ep]
                beta = np.asarray(beta_arr, dtype=np.float32)
            except (KeyError, IndexError):
                beta = np.zeros(10, dtype=np.float32)
        except Exception:
            r.update({"occlusion_score": np.nan, "visible_ratio": np.nan})
            continue

        verts_world, faces = get_mano_verts_world(
            mano_pose, beta, R_world, t_world, flip, device,
        )
        if verts_world is None:
            r.update({"occlusion_score": np.nan, "visible_ratio": np.nan})
            continue

        # Episode intrinsics (cached)
        K_vid, video_w, video_h = get_episode_intrinsics(
            ep, memmap_dir, video_root, dataset, calib,
        )

        # Convert to camera space
        T_C_W = np.linalg.inv(T_W_C)
        R_C_W = T_C_W[:3, :3].astype(np.float64)
        t_C_W = T_C_W[:3, 3].astype(np.float64)
        verts_cam = (R_C_W @ verts_world.T).T + t_C_W

        # Compute occlusion
        occ = compute_self_occlusion(
            verts_cam, faces, K_vid, video_h, video_w,
            depth_eps=0.005, window_half=2,
        )
        r["occlusion_score"] = round(float(occ["occlusion_score"]), 6)
        r["visible_ratio"] = round(float(occ["visible_ratio"]), 6)

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-sample occlusion vs fusion gain on test set")
    parser.add_argument("--config-name",
                        default="experiment/fusion/vision_resnet_small_emgfusion_center")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-location", default="data/EgoEMG_v2_memmap")
    parser.add_argument("--video-root", default="data/EgoEMG_allintra")
    parser.add_argument("--output-csv", default="./occlusion_fusion_results.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+", default=["user", "gesture", "both"])
    parser.add_argument("--hands", nargs="+", default=["left", "right"])
    parser.add_argument("--max-samples-per-split", type=int, default=0,
                        help="Limit samples per split (0=all)")
    parser.add_argument("--emg-checkpoint", default=None,
                        help="Pretrained EMG checkpoint for EMG-only head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-occlusion", action="store_true",
                        help="Skip occlusion computation (faster, MAE only)")
    args = parser.parse_args()

    device = torch.device(args.device)
    memmap_dir = Path(args.data_location)
    video_root = Path(args.video_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────────
    from hydra import compose, initialize_config_dir

    exp_name = args.config_name
    for prefix in ("experiment/", "experiment\\"):
        if exp_name.startswith(prefix):
            exp_name = exp_name[len(prefix):]
            break

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        try:
            cfg = compose(config_name=args.config_name)
        except Exception:
            cfg = compose(config_name="base", overrides=[f"experiment={exp_name}"])

    print(f"Config: {args.config_name}")
    print(f"Device: {device}")

    # ── Load fusion model ───────────────────────────────────────────────────
    print(f"\nLoading fusion checkpoint: {args.checkpoint}")
    module = load_module(args.checkpoint, cfg)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    # ── Load standalone EMG model ──────────────────────────────────────────
    emg_ckpt_path = args.emg_checkpoint or cfg.get("pretrained_emg_checkpoint")
    emg_model = None
    if emg_ckpt_path and Path(emg_ckpt_path).exists():
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
            try:
                emg_cfg = compose(
                    config_name="experiment/emgformer/regression_emgformer_small_aggressive_egoemg")
            except Exception:
                emg_cfg = compose(
                    config_name="base",
                    overrides=["experiment=emgformer/regression_emgformer_small_aggressive_egoemg"])
        emg_model = Emg2PoseFormer(
            featurizer=instantiate(emg_cfg.module.featurizer),
            decoder=instantiate(emg_cfg.module.decoder),
            head=instantiate(emg_cfg.module.head),
            out_channels=emg_cfg.module.get("out_channels", 22),
            provide_initial_pos=emg_cfg.module.get("provide_initial_pos", False),
        )
        emg_ckpt = torch.load(emg_ckpt_path, map_location="cpu", weights_only=False)
        emg_sd = emg_ckpt.get("state_dict", emg_ckpt)
        emg_sd_remapped = {k[6:] if k.startswith("model.") else k: v
                           for k, v in emg_sd.items()}
        emg_model.load_state_dict(emg_sd_remapped, strict=True)
        emg_model.to(device)
        emg_model.eval()
        n_p = sum(p.numel() for p in emg_model.parameters())
        print(f"Loaded standalone EMG model: {n_p:,} params")
    else:
        print("No standalone EMG model loaded (EMG MAE will be 0)")

    # ── Load standalone vision model ───────────────────────────────────────
    vis_ckpt_key = (
        "vision_vit_checkpoint"
        if cfg.module.get("vision_backbone_type", "").startswith("vit")
        else "vision_resnet_checkpoint"
    )
    vis_ckpt_path = cfg.get(vis_ckpt_key)
    vis_model = None
    if vis_ckpt_path and Path(vis_ckpt_path).exists():
        vis_ckpt = torch.load(vis_ckpt_path, map_location="cpu", weights_only=False)
        vis_sd = vis_ckpt.get("state_dict", vis_ckpt)
        vis_sd_remapped = {k[6:] if k.startswith("model.") else k: v
                           for k, v in vis_sd.items()}
        vis_backbone = cfg.module.get("vision_backbone_type", "resnet18")
        VisModelClass = VisionViTPose if vis_backbone.startswith("vit") else ResNetVisionPose
        vis_model = VisModelClass(
            out_channels=22,
            backbone_type=vis_backbone,
            pretrained=False,
            head_hidden=512,
            head_dropout=0.1,
        )
        vis_model.load_state_dict(vis_sd_remapped, strict=True)
        vis_model.to(device)
        vis_model.eval()
        n_p = sum(p.numel() for p in vis_model.parameters())
        print(f"Loaded standalone vision model: {n_p:,} params")
    else:
        print("No standalone vision model loaded (vision MAE will be 0)")

    # ── Monkey-patch fusion forward (matching find_fusion_wins.py) ─────────
    original_forward_cs = module.model._forward_center_supervised

    def patched_forward_cs(self, batch, vision_features, emg):
        emg_features = self.featurizer(emg)
        decoded = self.decoder(emg_features)
        attn_scores = self.temporal_attn(decoded.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=1)
        emg_pooled = (decoded * attn_weights.squeeze(-1).unsqueeze(1)).sum(dim=-1)
        if "vision_valid_mask" in batch:
            vision_valid = batch["vision_valid_mask"]
            if vision_valid.ndim > 1:
                vision_valid = vision_valid.any(dim=1)
            vision_features = vision_features * vision_valid[:, None].to(vision_features.dtype)
        y_v = self.head_vision(vision_features)
        vis_feat = self.vision_proj(vision_features)
        fused = torch.cat([emg_pooled, vis_feat], dim=-1).unsqueeze(-1)
        fused = self.fusion_proj(fused)
        delta = self.head(fused)
        self._last_delta = delta
        preds = y_v.unsqueeze(-1) + delta
        if "joint_angles" in batch and "label_valid_mask" in batch:
            ja = batch["joint_angles"]
            mask = batch["label_valid_mask"]
            if ja.shape[-1] == 1:
                return preds, ja, mask[..., :1] if mask.ndim >= 2 else mask.unsqueeze(-1)
            half_ctx = self.left_context // 2
            right_stop = -half_ctx if half_ctx > 0 else None
            targets_full = ja[..., half_ctx:right_stop]
            mask_full = mask[..., half_ctx:right_stop]
            center = targets_full.shape[-1] // 2
            targets = targets_full[:, :, center:center + 1]
            mask_out = mask_full[:, :, center:center + 1] if mask_full.ndim >= 3 else mask_full[..., center:center + 1]
            return preds, targets, mask_out
        return preds

    module.model._forward_center_supervised = types.MethodType(
        patched_forward_cs, module.model)
    inner_model = module.model

    # ── Load calibration ─────────────────────────────────────────────────────
    calib_path = (_PROJECT_DIR / "data" / "EgoEMG" / "reprojection_assets" /
                  "GX010023_standard_calibration.json")
    with open(calib_path) as f:
        calib = json.load(f)

    # ── Dataset params ───────────────────────────────────────────────────────
    emg_layout = cfg.get("egoemg_emg_layout", "emg2pose_interpolate16")
    channel_indices = cfg.get("egoemg_emg2pose_channel_indices",
                              [10, 12, 0, 1, 2, 4, 5, 6])
    channel_interpolate = cfg.get("egoemg_channel_interpolate", False)
    norm_stats_path = cfg.datamodule.get("per_dataset_norm_stats_path")
    val_window = cfg.datamodule.get("val_test_window_length", 7790)
    val_stride = cfg.datamodule.get("val_test_stride", val_window) // 4
    vit_features_dir = cfg.get("cached_vit_features_dir")
    crops_dir = cfg.get("per_episode_crops_dir")
    skip_emg = cfg.get("skip_emg_loading", False)

    # ── Iterate splits × hands ─────────────────────────────────────────────
    all_results: list[dict] = []
    base_dataset = None  # keep reference to last dataset for episode metadata

    for split in args.splits:
        for hand in args.hands:
            print(f"\n{'='*50}")
            print(f"Split={split}, hand={hand}")

            dataset = EgoEmgMemmapDataset(
                memmap_dir=str(memmap_dir),
                window_length=int(val_window),
                stride=int(val_stride),
                allowed_splits=[split],
                modalities=["emg", "joint_angles", "labels"],
                target_hand=hand,
                emg_field_preference="filtered",
                emg_layout=emg_layout,
                emg2pose_channel_indices=channel_indices,
                channel_interpolate=channel_interpolate,
                norm_mode="per-dataset",
                norm_stats_path=norm_stats_path,
                dataset_name="egoemg",
                jitter=False,
                cached_vit_features_dir=vit_features_dir,
                per_episode_crops_dir=crops_dir,
                vision_num_frames=cfg.get("vision_num_frames", 0),
                vision_frame_selection=cfg.get("vision_frame_selection", "center"),
                vision_patch_size=cfg.get("vision_patch_size", 256),
                center_target_only=cfg.get("center_target_only", False),
                skip_emg_loading=skip_emg,
            )
            base_dataset = dataset

            n_total = len(dataset)
            print(f"  Samples: {n_total:,}")

            rng = np.random.default_rng(args.seed)
            if args.max_samples_per_split > 0 and n_total > args.max_samples_per_split:
                indices = sorted(rng.choice(
                    n_total, size=args.max_samples_per_split, replace=False).tolist())
            else:
                indices = list(range(n_total))

            ds_meta = _DatasetWithMeta(dataset, indices)
            dataloader = DataLoader(
                ds_meta, batch_size=args.batch_size, shuffle=False,
                num_workers=4, collate_fn=_collate_fusion, pin_memory=True,
            )

            split_results = evaluate_all_samples(
                module, dataloader, device, inner_model,
                memmap_dir=memmap_dir,
                video_root=video_root,
                dataset=dataset,
                calib=calib,
                emg_model=emg_model,
                vis_model=vis_model,
            )
            for r in split_results:
                r["split"] = split
            all_results.extend(split_results)
            print(f"  Evaluated: {len(split_results)} samples")

    # ── Restore original forward ─────────────────────────────────────────────
    module.model._forward_center_supervised = original_forward_cs

    # ── Build CSV ───────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        rows.append({
            "split": r.get("split", "?"),
            "hand": r.get("hand", "?"),
            "ep_idx": r.get("ep_idx", -1),
            "center_idx": r.get("center_idx", -1),
            "mocap_tracked": int(r.get("mocap_tracked", False)),
            "mocap_stale": int(r.get("mocap_stale", False)),
            "markers_valid": int(r.get("markers_valid", False)),
            "occlusion_score": r.get("occlusion_score", np.nan),
            "visible_ratio": r.get("visible_ratio", np.nan),
            "mae_vision_deg": round(r["mae_vision_rad"] * 57.3, 4),
            "mae_emg_deg": round(r["mae_emg_rad"] * 57.3, 4),
            "mae_fusion_deg": round(r["mae_fusion_rad"] * 57.3, 4),
            "delta_fusion_deg": round(
                (r["mae_vision_rad"] - r["mae_fusion_rad"]) * 57.3, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to {output_csv}")

    # ── Summary statistics ───────────────────────────────────────────────────
    valid_occ = df[df["occlusion_score"].notna()]
    print(f"\nSamples with valid occlusion: {len(valid_occ)} / {len(df)}")
    if len(valid_occ) > 0:
        print(f"Occlusion score range: [{valid_occ['occlusion_score'].min():.4f}, "
              f"{valid_occ['occlusion_score'].max():.4f}]")
        print(f"Occlusion score mean: {valid_occ['occlusion_score'].mean():.4f}")

    valid_mae = df[df["mae_vision_deg"] > 0.01]
    if len(valid_mae) > 0:
        print(f"\nMAE stats ({len(valid_mae)} samples with predictions):")
        print(f"  Vision: {valid_mae['mae_vision_deg'].mean():.2f}°")
        print(f"  EMG:    {valid_mae['mae_emg_deg'].mean():.2f}°")
        print(f"  Fusion: {valid_mae['mae_fusion_deg'].mean():.2f}°")
        print(f"  Delta:  {valid_mae['delta_fusion_deg'].mean():.2f}° "
              f"({(valid_mae['delta_fusion_deg'] > 0).mean()*100:.1f}% positive)")

    # ── Occlusion-binned analysis ──────────────────────────────────────────
    if len(valid_occ) > 0 and len(valid_mae) > 0:
        merged = valid_occ[valid_occ["mae_vision_deg"] > 0.01]
        merged = merged.copy()
        merged["occ_bin"] = pd.cut(
            merged["occlusion_score"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        )
        print(f"\nOcclusion-binned fusion gain:")
        print(f"{'Bin':<12} {'N':<8} {'Vis(°)':<10} {'EMG(°)':<10} "
              f"{'Fus(°)':<10} {'Δ(°)':<10}")
        print("-" * 60)
        for bin_label, grp in merged.groupby("occ_bin", observed=True):
            print(f"{str(bin_label):<12} {len(grp):<8} "
                  f"{grp['mae_vision_deg'].mean():<10.2f} "
                  f"{grp['mae_emg_deg'].mean():<10.2f} "
                  f"{grp['mae_fusion_deg'].mean():<10.2f} "
                  f"{grp['delta_fusion_deg'].mean():<10.2f}")


if __name__ == "__main__":
    main()
