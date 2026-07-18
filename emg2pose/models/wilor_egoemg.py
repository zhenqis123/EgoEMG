"""WiLoR wrapper for EgoEMG fine-tuning.

The upstream WiLoR model is reused as-is for outputs and losses, but this
wrapper fixes a few training-entry incompatibilities for local fine-tuning:
- optional adversarial branch can be disabled cleanly;
- the debug `torch.save(...)` in upstream forward_step is removed;
- validation/test loss is explicitly logged for Lightning callbacks;
- scalar loss logging is decoupled from the mesh renderer so TensorBoard
  always receives per-component loss values even without pyrender;
- optional JointAngleHead for angle regression from MANO output.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


WILOR_PATH = Path(__file__).resolve().parents[2] / ".." / "WiLoR"
if str(WILOR_PATH) not in sys.path:
    sys.path.insert(0, str(WILOR_PATH))

from wilor.models.wilor import WiLoR
from wilor.utils.geometry import perspective_projection

from emg2pose.models.vit_freeze import apply_vit_freeze, get_vit_param_groups

log = logging.getLogger(__name__)

# 20D emg2pose joint angle names and (MANO joint index, Euler axis index).
ANGLE_NAMES = [
    "thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe",
    "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe",
    "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe",
    "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe",
    "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe",
]
EULER_TO_EMG2POSE = {
    "thumb_cmc_fe": (13, 2), "thumb_cmc_aa": (13, 1),
    "thumb_mcp_fe": (14, 2), "thumb_ip_fe": (15, 2),
    "index_mcp_aa": (1, 1), "index_mcp_fe": (1, 2),
    "index_pip_fe": (2, 2), "index_dip_fe": (3, 2),
    "middle_mcp_aa": (4, 1), "middle_mcp_fe": (4, 2),
    "middle_pip_fe": (5, 2), "middle_dip_fe": (6, 2),
    "ring_mcp_aa": (10, 1), "ring_mcp_fe": (10, 2),
    "ring_pip_fe": (11, 2), "ring_dip_fe": (12, 2),
    "pinky_mcp_aa": (7, 1), "pinky_mcp_fe": (7, 2),
    "pinky_pip_fe": (8, 2), "pinky_dip_fe": (9, 2),
}
_EULER_INDICES = [EULER_TO_EMG2POSE[n] for n in ANGLE_NAMES]

MANO_ASSETS_ROOT = Path("/home/xiziheng/develop/HandVQVAE/assets/mano")
MANOTORCH_ROOT = Path("/home/xiziheng/develop/manotorch")


def _aa_to_rotmat_batch(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (B*N, 3) to rotation matrices (B*N, 3, 3)."""
    from wilor.utils.geometry import aa_to_rotmat
    return aa_to_rotmat(aa)


def _rotmat_to_aa(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (... , 3, 3) to axis-angle (..., 3)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    skew = R - R.transpose(-2, -1)
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / axis_norm
    return axis * angle.unsqueeze(-1)


@torch.no_grad()
def _root_rel_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Root-relative MPJPE in mm. pred: (B,J,3), gt: (B,J,4) xyz+conf."""
    with torch.cuda.amp.autocast(enabled=False):
        pred_rel = pred.float() - pred[:, 0:1].float()
        gt_xyz = gt[..., :3].float() - gt[:, 0:1, :3].float()
        conf = gt[..., 3]
        err = (pred_rel - gt_xyz).norm(dim=-1)
        valid = conf > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (err * valid).sum() / valid.sum() * 1000


@torch.no_grad()
def _pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-aligned MPJPE in mm. pred: (B,J,3), gt: (B,J,4) xyz+conf."""
    with torch.cuda.amp.autocast(enabled=False):
        gt_xyz = gt[..., :3].float()
        conf = gt[..., 3]
        valid = conf > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        B = pred.shape[0]
        errors = []
        for b in range(B):
            v = valid[b]
            if v.sum() < 3:
                continue
            p = pred[b, v].float()
            g = gt_xyz[b, v].float()
            p_c = p - p.mean(0)
            g_c = g - g.mean(0)
            H = p_c.T @ g_c
            U, _, Vh = torch.linalg.svd(H)
            d = torch.det(Vh.T @ U.T).sign()
            Vh[-1] = Vh[-1] * d
            R = Vh.T @ U.T
            p_aligned = p_c @ R.T + g.mean(0)
            errors.append((p_aligned - g).norm(dim=-1).mean())
        if not errors:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(errors).mean() * 1000


@torch.no_grad()
def _reproj_error(
    pred_2d: torch.Tensor, gt_2d: torch.Tensor
) -> torch.Tensor:
    """Mean 2D reprojection error in pixels. gt_2d: (B,J,3) xy+conf."""
    with torch.cuda.amp.autocast(enabled=False):
        gt_xy = gt_2d[..., :2].float()
        conf = gt_2d[..., 2]
        err = (pred_2d.float() - gt_xy).norm(dim=-1)
        valid = conf > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_2d.device)
        return (err * valid).sum() / valid.sum()


class JointAngleHead(nn.Module):
    """Simple MLP: concatenated MANO params -> joint angles."""

    def __init__(
        self,
        in_dim: int = 154,
        num_joints: int = 22,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_joints),
        )

    def forward(self, mano_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        go = mano_params["global_orient"].flatten(1)
        hp = mano_params["hand_pose"].flatten(1)
        betas = mano_params["betas"].flatten(1)
        x = torch.cat([go, hp, betas], dim=-1)
        return self.mlp(x)


class EgoEMGWiLoRModule(WiLoR):
    """WiLoR fine-tuning module with safer local training defaults."""

    IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        cfg,
        joint_angle_head: bool = False,
        num_joints: int = 22,
        joint_angle_loss_weight: float = 1.0,
        backbone_freeze: dict | None = None,
        num_log_images: int = 4,
    ):
        super().__init__(cfg, init_renderer=False)
        self.backbone_freeze_cfg = backbone_freeze or {}
        self.joint_angle_head = (
            JointAngleHead(
                in_dim=9 + 135 + 10,
                num_joints=num_joints,
            )
            if joint_angle_head
            else None
        )
        self.joint_angle_loss_weight = joint_angle_loss_weight
        self._num_log_images = num_log_images
        self._vis_logged_count = 0

        if self.backbone_freeze_cfg:
            apply_vit_freeze(self.backbone, **self.backbone_freeze_cfg)
            frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
            total = sum(1 for p in self.backbone.parameters())
            log.info("ViT: %d/%d param tensors frozen", frozen, total)

    def on_after_backward(self):
        pass

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        x = batch["img"]
        batch_size = x.shape[0]

        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(
            x[:, :, :, 32:-32]
        )

        device = temp_mano_params["hand_pose"].device
        dtype = temp_mano_params["hand_pose"].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            batch_size,
            2,
            device=device,
            dtype=dtype,
        )

        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(
            batch_size, -1, 3, 3,
        )
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(
            batch_size, -1, 3, 3,
        )
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)
        temp_vertices = self.mano(
            **{k: v.float() for k, v in temp_mano_params.items()},
            pose2rot=False,
        ).vertices

        pred_mano_params, pred_cam = self.refine_net(
            vit_out,
            temp_vertices,
            pred_cam,
            pred_mano_feats,
            focal_length,
        )

        output = {
            "pred_cam": pred_cam,
            "pred_mano_params": {k: v.clone() for k, v in pred_mano_params.items()},
        }
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length

        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(
            batch_size, -1, 3, 3,
        )
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(
            batch_size, -1, 3, 3,
        )
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        mano_output = self.mano(
            **{k: v.float() for k, v in pred_mano_params.items()},
            pose2rot=False,
        )
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
        )
        output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        if self.joint_angle_head is not None:
            output["pred_joint_angles"] = self.joint_angle_head(pred_mano_params)

        return output

    @pl.utilities.rank_zero.rank_zero_only
    def _log_loss_scalars(
        self,
        output: Dict[str, Any],
        step_count: int,
        mode: str,
    ) -> None:
        if "losses" not in output:
            return
        summary_writer = self.logger.experiment
        for loss_name, val in output["losses"].items():
            summary_writer.add_scalar(
                f"{mode}/{loss_name}", val.detach().item(), step_count
            )

    def tensorboard_logging(
        self,
        batch: Dict[str, Any],
        output: Dict[str, Any],
        step_count: int,
        train: bool = True,
        write_to_summary_writer: bool = True,
    ):
        """Override parent to skip mesh rendering (renderer not initialized)."""
        mode = "train" if train else "val"
        self._log_loss_scalars(output, step_count, mode)

    def configure_optimizers(self):
        lr = self.cfg.TRAIN.LR
        wd = self.cfg.TRAIN.WEIGHT_DECAY
        param_groups: list[dict] = []

        # ViT backbone groups
        if self.backbone_freeze_cfg:
            vit_groups = get_vit_param_groups(
                self.backbone, **self.backbone_freeze_cfg
            )
            param_groups.extend(vit_groups)
            vit_group_ids = set()
            for g in vit_groups:
                vit_group_ids.update(id(p) for p in g["params"])
            other_bb = [
                p for p in self.backbone.parameters()
                if p.requires_grad and id(p) not in vit_group_ids
            ]
            if other_bb:
                param_groups.append({"params": other_bb, "lr": lr})
        else:
            bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if bb_params:
                param_groups.append({"params": bb_params, "lr": lr})

        # RefineNet at main LR
        refine_params = [p for p in self.refine_net.parameters() if p.requires_grad]
        if refine_params:
            param_groups.append({"params": refine_params, "lr": lr})

        # JointAngleHead at main LR
        if self.joint_angle_head is not None:
            head_params = list(self.joint_angle_head.parameters())
            if head_params:
                param_groups.append({"params": head_params, "lr": lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer_disc = torch.optim.AdamW(
                params=self.discriminator.parameters(),
                lr=lr,
                weight_decay=wd,
            )
            return optimizer, optimizer_disc
        return optimizer

    @staticmethod
    def _project_cam_points_to_patch(
        points_cam_np: np.ndarray,
        K_np: np.ndarray,
        dist_np: np.ndarray,
        cam_crop_params_np: np.ndarray,
        box_center_np: np.ndarray,
        box_size_np: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        """Project camera-space 3D points to patch pixel coordinates.

        Applies calibrated-camera projection → raw video mapping → bbox-to-patch.
        All inputs are numpy arrays (detached from any computation graph).

        Args:
            points_cam_np: (B, N, 3) camera-space 3D points.
            K_np: (B, 3, 3) camera intrinsic matrices.
            dist_np: (B, D) distortion coefficients.
            cam_crop_params_np: (B, 6) [proc_w, proc_h, crop_w, crop_h, crop_x0, crop_y0].
            box_center_np: (B, 2) bbox centers in raw video coords.
            box_size_np: (B,) bbox sizes.
            patch_size: output patch size (square).
        Returns:
            (B, N, 2) patch pixel coordinates.
        """
        B = points_cam_np.shape[0]
        proj_list = []
        for b in range(B):
            pts, _ = cv2.projectPoints(
                points_cam_np[b], np.zeros(3), np.zeros(3),
                K_np[b], dist_np[b].reshape(-1, 1),
            )
            proj_list.append(pts.reshape(-1, 2))
        proj_2d = np.stack(proj_list, axis=0)

        cp = cam_crop_params_np
        pw, ph = cp[:, 0:1, None], cp[:, 1:2, None]
        cw, ch = cp[:, 2:3, None], cp[:, 3:4, None]
        x0, y0 = cp[:, 4:5, None], cp[:, 5:6, None]
        raw_2d = np.concatenate([
            (proj_2d[:, :, 0:1] / pw) * cw + x0,
            (proj_2d[:, :, 1:2] / ph) * ch + y0,
        ], axis=-1)

        bc = box_center_np
        bs = box_size_np
        s = patch_size / bs
        patch_2d = (raw_2d - bc[:, None, :]) * s[:, None, None] + patch_size / 2.0
        return patch_2d

    @torch.no_grad()
    def _forward_gt_mano(
        self, batch: Dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward GT MANO params to get joints/vertices in the same space as pred.

        Uses global_orient=0 in MANO and applies the camera-space rotation externally
        as a rigid transform.  This avoids the LBS root-joint offset that occurs when
        MANO rotates around J[0] instead of the origin.

        Returns:
            gt_keypoints_3d: (B, J, 3) joints with external rotation applied
            gt_keypoints_2d: (B, J, 2) projected 2D joints in normalized coords
            gt_vertices: (B, V, 3) vertices with external rotation applied
        """
        batch_size = batch["img"].shape[0]
        device = batch["img"].device

        gt_mano = dict(batch["mano_params"])
        hand_mean = self.mano.hand_mean.to(
            device=device, dtype=gt_mano["hand_pose"].dtype,
        )
        gt_mano["hand_pose"] = gt_mano["hand_pose"] + hand_mean.view(1, -1)

        # Extract camera-space rotation for external application.
        R_cam = _aa_to_rotmat_batch(
            gt_mano["global_orient"].reshape(-1, 3)
        ).reshape(batch_size, 3, 3)

        # Pass identity global_orient to MANO; we apply the rotation externally.
        gt_mano["global_orient"] = (
            torch.eye(3, device=device, dtype=gt_mano["hand_pose"].dtype)
            .unsqueeze(0).unsqueeze(0)
            .expand(batch_size, 1, 3, 3)
            .contiguous()
        )
        gt_mano["hand_pose"] = _aa_to_rotmat_batch(
            gt_mano["hand_pose"].reshape(-1, 3)
        ).reshape(batch_size, -1, 3, 3)
        gt_mano["betas"] = gt_mano["betas"].reshape(batch_size, -1)

        mano_out = self.mano(**{k: v.float() for k, v in gt_mano.items()}, pose2rot=False)
        kp_local = mano_out.joints.reshape(batch_size, -1, 3)
        verts_local = mano_out.vertices.reshape(batch_size, -1, 3)

        # External rigid rotation (around origin, not root joint).
        gt_keypoints_3d = torch.bmm(
            kp_local, R_cam.transpose(1, 2),
        )
        gt_vertices = torch.bmm(
            verts_local, R_cam.transpose(1, 2),
        )

        # Project GT joints through calibrated camera → raw video → patch → normalized.
        wrist_pos_cam = batch["wrist_pos_cam"].to(device)
        patch_size = self.cfg.MODEL.IMAGE_SIZE

        joints_cam = gt_keypoints_3d + wrist_pos_cam.unsqueeze(1)
        patch_2d = self._project_cam_points_to_patch(
            joints_cam.detach().cpu().float().numpy(),
            batch["cam_K"].detach().cpu().float().numpy(),
            batch["cam_dist"].detach().cpu().float().numpy(),
            batch["cam_crop_params"].detach().cpu().float().numpy(),
            batch["box_center"].detach().cpu().float().numpy(),
            batch["box_size"].detach().cpu().float().numpy(),
            patch_size,
        )

        gt_keypoints_2d = torch.from_numpy(
            (patch_2d / patch_size - 0.5).astype(np.float32),
        ).to(device)

        return gt_keypoints_3d, gt_keypoints_2d, gt_vertices

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        batch_for_loss = dict(batch)
        mano_params = dict(batch["mano_params"])
        hand_mean = self.mano.hand_mean.to(
            device=mano_params["hand_pose"].device,
            dtype=mano_params["hand_pose"].dtype,
        )
        mano_params["hand_pose"] = mano_params["hand_pose"] + hand_mean.view(1, -1)
        batch_for_loss["mano_params"] = mano_params

        # Replace mocap-marker GT keypoints with MANO-forwarded GT joints.
        gt_kp3d, gt_kp2d, _ = output.get("gt_mano", (None, None, None))
        if gt_kp3d is None:
            gt_kp3d, gt_kp2d, _ = self._forward_gt_mano(batch)
        label_valid = batch["has_mano_params"]["global_orient"].unsqueeze(-1).unsqueeze(-1).float()
        batch_for_loss["keypoints_3d"] = torch.cat(
            [gt_kp3d, label_valid.expand_as(gt_kp3d[..., :1])], dim=-1,
        )
        batch_for_loss["keypoints_2d"] = torch.cat(
            [gt_kp2d, label_valid.expand_as(gt_kp2d[..., :1])], dim=-1,
        )

        loss = super().compute_loss(batch_for_loss, output, train=train)

        if self.joint_angle_head is not None and "pred_joint_angles" in output:
            gt_angles = batch["joint_angles"]
            pred_angles = output["pred_joint_angles"]
            n = min(pred_angles.shape[-1], gt_angles.shape[-1])
            loss_ja = torch.nn.functional.l1_loss(pred_angles[..., :n], gt_angles[..., :n])
            loss = loss + self.joint_angle_loss_weight * loss_ja

        return loss

    def _init_angle_converter(self):
        """Lazy-init manotorch components for angle extraction."""
        if hasattr(self, "_angle_mano"):
            return
        if str(MANOTORCH_ROOT) not in sys.path:
            sys.path.insert(0, str(MANOTORCH_ROOT))
        from manotorch.axislayer import AxisLayerFK
        from manotorch.manolayer import ManoLayer
        device = next(self.parameters()).device
        self._angle_mano = ManoLayer(
            rot_mode="axisang", side="right",
            mano_assets_root=str(MANO_ASSETS_ROOT),
            use_pca=False, flat_hand_mean=False,
        ).to(device)
        self._angle_axis = AxisLayerFK(
            side="right",
            mano_assets_root=str(MANO_ASSETS_ROOT),
        ).to(device)

    @torch.no_grad()
    def _pred_to_angles(self, pred_mano_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert predicted MANO rotmats to 20D emg2pose angles via manotorch."""
        self._init_angle_converter()
        device = pred_mano_params["hand_pose"].device
        with torch.cuda.amp.autocast(enabled=False):
            hp_aa = _rotmat_to_aa(
                pred_mano_params["hand_pose"].flatten(0, 1)
            ).reshape(-1, 45)
            # global_orient doesn't affect finger angles; use zeros.
            go_aa = torch.zeros(hp_aa.shape[0], 3, device=device, dtype=hp_aa.dtype)
            full_pose = torch.cat([go_aa, hp_aa], dim=-1)
            betas = pred_mano_params["betas"].reshape(-1, 10).float()
            mano_out = self._angle_mano(full_pose, betas)
            ee = self._angle_axis(mano_out.transforms_abs)[2]
            angles = torch.stack(
                [ee[:, ji, ai] for ji, ai in _EULER_INDICES], dim=1,
            )
            return angles  # (B, 20)

    def _log_pose_metrics(
        self, batch: Dict, output: Dict, stage: str
    ) -> None:
        pred_kp3d = output["pred_keypoints_3d"]  # (B, 21, 3)
        pred_kp2d = output["pred_keypoints_2d"]  # (B, 21, 2)

        gt_kp3d, gt_kp2d, _ = output.get("gt_mano", (None, None, None))
        if gt_kp3d is None:
            gt_kp3d, gt_kp2d, _ = self._forward_gt_mano(batch)
        label_valid = batch["has_mano_params"]["global_orient"].unsqueeze(-1).unsqueeze(-1).float()
        gt_kp3d_conf = torch.cat([gt_kp3d, label_valid.expand_as(gt_kp3d[..., :1])], dim=-1)
        gt_kp2d_conf = torch.cat([gt_kp2d, label_valid.expand_as(gt_kp2d[..., :1])], dim=-1)

        mpjpe = _root_rel_mpjpe(pred_kp3d, gt_kp3d_conf)
        pa = _pa_mpjpe(pred_kp3d, gt_kp3d_conf)
        reproj = _reproj_error(pred_kp2d, gt_kp2d_conf)

        bs = pred_kp3d.shape[0]
        self.log(f"{stage}/mpjpe_mm", mpjpe, on_epoch=True, prog_bar=True,
                 batch_size=bs)
        self.log(f"{stage}/pa_mpjpe_mm", pa, on_epoch=True, prog_bar=True,
                 batch_size=bs)
        self.log(f"{stage}/reproj_px", reproj, on_epoch=True, batch_size=bs)

        # Sanity check prints — verify GT keypoints come from MANO forward
        # and the pretrained model already has baseline prediction ability.
        if getattr(self.trainer, "sanity_checking", False):
            gt_xyz = gt_kp3d_conf[..., :3]
            gt_conf = gt_kp3d_conf[..., 3]
            gt_min = gt_xyz.min().item()
            gt_max = gt_xyz.max().item()
            pred_min = pred_kp3d.min().item()
            pred_max = pred_kp3d.max().item()
            loss_str = ", ".join(
                f"{k}={v.item():.4f}"
                for k, v in output.get("losses", {}).items()
            )
            log.info(
                "[sanity_check] bs=%d | mpjpe=%.1f mm, pa_mpjpe=%.1f mm, "
                "reproj=%.1f px | GT range[%.1f, %.1f] Pred range[%.1f, %.1f] | "
                "GT conf mean=%.2f valid=%.1f%% | %s",
                bs,
                float(mpjpe), float(pa), float(reproj),
                gt_min, gt_max, pred_min, pred_max,
                float(gt_conf.mean()),
                float((gt_conf > 0).float().mean() * 100),
                loss_str,
            )

        # Per-joint angle MAE from MANO params.
        gt_angles_20 = batch["joint_angles"][:, :20].float()
        pred_angles_20 = self._pred_to_angles(output["pred_mano_params"])
        mae_per_joint = (pred_angles_20 - gt_angles_20).abs()  # (B, 20)
        mae_deg = mae_per_joint.mean() * 180.0 / 3.14159265
        self.log(f"{stage}/angle_mae_deg", mae_deg, on_epoch=True,
                 prog_bar=True, batch_size=bs)
        for i, name in enumerate(ANGLE_NAMES):
            self.log(f"{stage}/angle_{name}", mae_per_joint[:, i].mean() * 180.0 / 3.14159265,
                     on_epoch=True, batch_size=bs)

    def _denormalize_patch(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized (3,H,W) tensor back to BGR uint8 image."""
        img = img_tensor.cpu().float()
        mean = self.IMAGE_MEAN.to(img.device).view(3, 1, 1) * 255
        std = self.IMAGE_STD.to(img.device).view(3, 1, 1) * 255
        img = img * std + mean
        img = img.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _project_vertices_to_patch(
        vertices: torch.Tensor,
        pred_cam_t: torch.Tensor,
        focal_length: torch.Tensor,
        image_size: int,
        patch_size: int,
    ) -> np.ndarray:
        """Project 3D vertices to 2D patch coordinates. Returns (B, N, 2) float."""
        B = vertices.shape[0]
        proj = perspective_projection(
            vertices,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / image_size,
        )
        coords = proj.reshape(B, -1, 2).float().cpu().numpy()
        coords = (coords + 0.5) * patch_size
        return coords

    @staticmethod
    def _project_gt_verts_to_patch(
        gt_vertices: torch.Tensor,
        wrist_pos_cam: torch.Tensor,
        cam_K: torch.Tensor,
        cam_dist: torch.Tensor,
        cam_crop_params: torch.Tensor,
        box_center: torch.Tensor,
        box_size: torch.Tensor,
        patch_size: int,
    ) -> np.ndarray:
        """Project GT MANO vertices through calibrated camera to patch pixels."""
        v_cam = gt_vertices + wrist_pos_cam.unsqueeze(1)
        return EgoEMGWiLoRModule._project_cam_points_to_patch(
            v_cam.detach().cpu().float().numpy(),
            cam_K.detach().cpu().float().numpy(),
            cam_dist.detach().cpu().float().numpy(),
            cam_crop_params.detach().cpu().float().numpy(),
            box_center.detach().cpu().float().numpy(),
            box_size.detach().cpu().float().numpy(),
            patch_size,
        )

    @pl.utilities.rank_zero.rank_zero_only
    def _save_vis_images(
        self,
        batch: Dict[str, Any],
        output: Dict[str, Any],
    ) -> None:
        """Save GT/Pred mesh overlay images during validation."""
        if self._vis_logged_count >= self._num_log_images:
            return

        vis_dir = self._get_vis_dir()
        patch_size = batch["img"].shape[-1]

        B = batch["img"].shape[0]
        n_save = min(self._num_log_images - self._vis_logged_count, B)

        pred_vertices = output["pred_vertices"].reshape(B, -1, 3)
        pred_cam_t = output["pred_cam_t"]
        focal_length = output["focal_length"]
        gt_has_mano = batch["has_mano_params"]["global_orient"]

        _, _, gt_vertices = output.get("gt_mano", (None, None, None))
        if gt_vertices is None:
            _, _, gt_vertices = self._forward_gt_mano(batch)

        image_size = self.cfg.MODEL.IMAGE_SIZE
        pred_proj = self._project_vertices_to_patch(
            pred_vertices, pred_cam_t, focal_length, image_size, patch_size,
        )

        has_cam = "wrist_pos_cam" in batch and "cam_K" in batch
        if has_cam:
            gt_proj = self._project_gt_verts_to_patch(
                gt_vertices,
                batch["wrist_pos_cam"],
                batch["cam_K"],
                batch["cam_dist"],
                batch["cam_crop_params"],
                batch["box_center"],
                batch["box_size"],
                patch_size,
            )
        else:
            gt_proj = self._project_vertices_to_patch(
                gt_vertices, pred_cam_t, focal_length, image_size, patch_size,
            )

        for i in range(n_save):
            idx = self._vis_logged_count + i
            if not gt_has_mano[i]:
                continue

            patch_bgr = self._denormalize_patch(batch["img"][i])
            gt_canvas = patch_bgr.copy()
            pred_canvas = patch_bgr.copy()

            # GT vertices projected
            gt_v2d = gt_proj[i]
            for j in range(0, gt_v2d.shape[0], 3):
                x, y = int(gt_v2d[j, 0]), int(gt_v2d[j, 1])
                if 0 <= x < patch_size and 0 <= y < patch_size:
                    cv2.circle(gt_canvas, (x, y), 1, (0, 255, 0), -1)

            # Pred vertices projected
            pred_v2d = pred_proj[i]
            for j in range(0, pred_v2d.shape[0], 3):
                x, y = int(pred_v2d[j, 0]), int(pred_v2d[j, 1])
                if 0 <= x < patch_size and 0 <= y < patch_size:
                    cv2.circle(pred_canvas, (x, y), 1, (0, 0, 255), -1)

            combined = np.concatenate([gt_canvas, pred_canvas], axis=1)
            cv2.putText(combined, "GT", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, "Pred", (patch_size + 10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(vis_dir / f"val_{idx:04d}.jpg"), combined)

        self._vis_logged_count += n_save

    def on_validation_epoch_start(self) -> None:
        self._vis_logged_count = 0
        self._vis_logged_dir = None

    def _get_vis_dir(self) -> Path:
        """Create a new timestamped folder per validation epoch so outputs
        are never overwritten across epochs."""
        if getattr(self, "_vis_logged_dir", None) is not None:
            return self._vis_logged_dir
        epoch = self.current_epoch
        step = self.global_step
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(os.getcwd())
        vis_dir = base / f"vis_val_epoch{epoch}_step{step}_{ts}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        self._vis_logged_dir = vis_dir
        return vis_dir

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict:
        output = self.forward_step(batch, train=False)
        gt_kp3d, gt_kp2d, gt_verts = self._forward_gt_mano(batch)
        output["gt_mano"] = (gt_kp3d, gt_kp2d, gt_verts)
        loss = self.compute_loss(batch, output, train=False)
        output["loss"] = loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._log_pose_metrics(batch, output, "val")
        self.tensorboard_logging(batch, output, self.global_step, train=False)
        self._save_vis_images(batch, output)
        return output

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output["loss"] = loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._log_pose_metrics(batch, output, "test")
        return output
