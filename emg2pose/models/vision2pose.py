"""Vision-to-pose Lightning module using WiLoR ViT backbone.

Supports two supervision targets:
  - "angle": MLP head predicts joint angles (B, 22), L1 loss.
  - "mano": Uses ViT's built-in MANO decode heads for MANO param regression,
            delegates loss to EgoEMGWiLoRModule-style compute_loss.

Supports two freeze strategies via vision_freeze config:
  - "tiered": Freeze blocks[0:30], fine-tune blocks[30:32]+last_norm at low LR.
  - "simple": Freeze entire ViT, only train the head at uniform LR.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from emg2pose.models.vit_freeze import apply_vit_freeze, get_vit_param_groups

log = logging.getLogger(__name__)

WILOR_PATH = Path(__file__).resolve().parents[2] / ".." / "WiLoR"
if str(WILOR_PATH) not in sys.path:
    sys.path.insert(0, str(WILOR_PATH))


class JointAngleHead(nn.Module):
    """MLP head: ViT pooled features -> joint angles."""

    def __init__(
        self,
        feat_dim: int = 1280,
        num_joints: int = 22,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feat_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else num_joints
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Vision2PoseModule(pl.LightningModule):
    """Vision-to-pose: WiLoR ViT backbone + regression head.

    Args:
        num_joints: Number of output joint angles (default 22).
        lr: Learning rate for head and new layers.
        weight_decay: AdamW weight decay.
        pretrained_backbone_path: Path to WiLoR checkpoint.
        supervision_target: "angle" for joint angle regression, "mano" for MANO param regression.
        vision_freeze: Dict with freeze strategy config (strategy, frozen_block_end, etc.).
        max_epochs: For cosine scheduler T_max.
        mano_model_path: Path to MANO data dir (only needed for supervision_target="mano").
    """

    def __init__(
        self,
        num_joints: int = 22,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        pretrained_backbone_path: str | None = None,
        supervision_target: str = "angle",
        vision_freeze: dict[str, Any] | None = None,
        max_epochs: int = 100,
        mano_model_path: str | None = None,
        head_hidden_dim: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_joints = num_joints
        self.lr = lr
        self.weight_decay = weight_decay
        self.supervision_target = supervision_target
        self.vision_freeze_cfg = vision_freeze or {
            "strategy": "simple",
            "frozen_block_end": 32,
            "freeze_patch_embed": True,
            "freeze_pos_embed": True,
            "freeze_mano_tokens": True,
            "finetune_block_start": 32,
            "finetune_block_end": 32,
            "finetune_last_norm": False,
            "vit_finetune_lr": None,
        }
        self.max_epochs = max_epochs

        self._build_backbone(pretrained_backbone_path)
        self._build_head()
        self._apply_freeze()

    def _build_backbone(self, pretrained_path: str | None) -> None:
        from wilor.configs import get_config

        model_config_path = str(WILOR_PATH / "pretrained_models" / "model_config.yaml")
        cfg = get_config(model_config_path, merge=True, update_cachedir=False)

        mano_data_dir = self.hparams.mano_model_path or str(WILOR_PATH / "mano_data")
        cfg.defrost()
        cfg.MANO.DATA_DIR = mano_data_dir
        cfg.MANO.MODEL_PATH = mano_data_dir
        cfg.MANO.MEAN_PARAMS = str(Path(mano_data_dir) / "mano_mean_params.npz")
        cfg.freeze()

        from wilor.models.backbones import vit
        self.backbone = vit(cfg)

        if pretrained_path and Path(pretrained_path).exists():
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    backbone_state[k[len("backbone."):]] = v
            if backbone_state:
                missing, unexpected = self.backbone.load_state_dict(
                    backbone_state, strict=False
                )
                log.info(
                    "Loaded ViT backbone: %d keys, %d missing, %d unexpected",
                    len(backbone_state), len(missing), len(unexpected),
                )
            else:
                log.warning("No backbone keys found in checkpoint: %s", pretrained_path)
        else:
            log.warning("No pretrained backbone path or file not found.")

    def _build_head(self) -> None:
        if self.supervision_target == "angle":
            self.head = JointAngleHead(
                feat_dim=1280,
                num_joints=self.num_joints,
                hidden_dim=self.hparams.head_hidden_dim,
            )
        # For "mano", the ViT's built-in decpose/decshape/deccam are the head.

    def _apply_freeze(self) -> None:
        apply_vit_freeze(self.backbone, **self.vision_freeze_cfg)
        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        log.info("ViT: %d/%d param tensors frozen", frozen, total)

    # ── Feature extraction ──────────────────────────────────────────────────

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pooled ViT features.

        Args:
            images: (B, C, H, W) or (B, T, C, H, W).
        Returns:
            (B, 1280) or (B, T, 1280).
        """
        has_time = images.ndim == 5
        if has_time:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)

        out = self.backbone(images)
        if isinstance(out, tuple):
            for item in reversed(out):
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    feat = item
                    break
            else:
                raise ValueError("ViT backbone returned no 4D tensor")
        else:
            feat = out
        feat = feat.mean(dim=[-2, -1])  # (BT, 1280)

        if has_time:
            feat = feat.view(B, T, -1)
        return feat

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for angle supervision.

        Args:
            images: (B, C, H, W) or (B, T, C, H, W).
        Returns:
            joint_angles: (B, num_joints) or (B, T, num_joints).
        """
        feat = self._extract_features(images)
        return self.head(feat)

    def forward_mano(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for MANO supervision — uses ViT's built-in MANO heads."""
        out = self.backbone(images)
        pred_mano_params, pred_cam, pred_mano_feats, img_feat = out
        return {
            "pred_mano_params": pred_mano_params,
            "pred_cam": pred_cam,
            "pred_mano_feats": pred_mano_feats,
            "img_feat": img_feat,
        }

    # ── Training / validation / test steps ──────────────────────────────────

    def _angle_step(
        self, batch: dict[str, Any], stage: str
    ) -> torch.Tensor:
        images = batch["img"]
        targets = batch["joint_angles"]
        valid = batch.get(
            "label_valid_mask",
            torch.ones_like(targets[..., 0], dtype=torch.bool),
        )

        pred = self(images)

        if pred.shape[-1] != targets.shape[-1]:
            n = min(pred.shape[-1], targets.shape[-1])
            pred = pred[..., :n]
            targets = targets[..., :n]

        if valid.ndim < targets.ndim:
            valid = valid.unsqueeze(-1).expand_as(targets)
        valid = valid & torch.isfinite(targets).all(dim=-1, keepdim=True)

        loss = F.l1_loss(pred[valid], targets[valid])
        mae = (pred[valid] - targets[valid]).abs().mean()

        bs = images.shape[0]
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}/mae", mae, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def _mano_step(
        self, batch: dict[str, Any], stage: str
    ) -> torch.Tensor:
        images = batch["img"]

        output = self.forward_mano(images)

        gt_mano = batch["mano_params"]
        loss = self._mano_loss(output, gt_mano)

        bs = images.shape[0]
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def _mano_loss(
        self,
        output: dict[str, Any],
        gt_mano: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred_params = output["pred_mano_params"]
        loss = torch.tensor(0.0, device=self.device)
        if "hand_pose" in gt_mano and "hand_pose" in pred_params:
            loss = loss + F.mse_loss(
                pred_params["hand_pose"], gt_mano["hand_pose"]
            )
        if "global_orient" in gt_mano and "global_orient" in pred_params:
            loss = loss + F.mse_loss(
                pred_params["global_orient"], gt_mano["global_orient"]
            )
        if "betas" in gt_mano and "betas" in pred_params:
            loss = loss + 0.001 * F.mse_loss(
                pred_params["betas"], gt_mano["betas"]
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.supervision_target == "mano":
            return self._mano_step(batch, "train")
        return self._angle_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        if self.supervision_target == "mano":
            self._mano_step(batch, "val")
        else:
            self._angle_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        if self.supervision_target == "mano":
            self._mano_step(batch, "test")
        else:
            self._angle_step(batch, "test")

    # ── Optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict[str, Any]:
        # Collect head params (always trainable at self.lr)
        if self.supervision_target == "angle":
            head_params = list(self.head.parameters())
        else:
            head_params = []

        # ViT finetune param group (tiered strategy only)
        vit_groups = get_vit_param_groups(self.backbone, **self.vision_freeze_cfg)

        # Remaining trainable backbone params not in vit_groups
        vit_group_ids = set()
        for g in vit_groups:
            vit_group_ids.update(id(p) for p in g["params"])
        head_ids = {id(p) for p in head_params}

        other_backbone_params = [
            p
            for p in self.backbone.parameters()
            if p.requires_grad
            and id(p) not in vit_group_ids
            and id(p) not in head_ids
        ]

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": self.lr})
        if other_backbone_params:
            param_groups.append({"params": other_backbone_params, "lr": self.lr})
        param_groups.extend(vit_groups)

        if not param_groups:
            log.warning(
                "Vision2PoseModule: no trainable parameters found. "
                "All ViT params are frozen and no head is attached. "
                "The optimizer will have nothing to optimize."
            )
            return {"optimizer": torch.optim.AdamW([torch.zeros(1, requires_grad=True)])}

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
