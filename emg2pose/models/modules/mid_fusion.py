from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from emg2pose.models.modules.base import BaseModule
from emg2pose.models.modules._vision_constants import DINOV2_VARIANTS as _DINOV2_VARIANTS, RESNET_DIMS as _RESNET_DIMS
from emg2pose.models.modules._pooling import TemporalAttentionPool
from emg2pose.models.vit_freeze import apply_vit_freeze


log = logging.getLogger(__name__)

WILOR_PATH = Path(__file__).resolve().parents[3] / ".." / "WiLoR"
if str(WILOR_PATH) not in sys.path:
    sys.path.insert(0, str(WILOR_PATH))

# DINOv2 ViT variants from timm — shared with VisionViTPose

# ResNet backbone → feature dimension


def _init_residual_branch(head: nn.Module) -> None:
    """Zero-initialize EMG residual head last layer so fusion starts at vision baseline.

    head (MLPHead): last Linear layer → near-zero weights, zero bias → Δy_emg ≈ 0
    With Δy_emg ≈ 0 at init, preds = y_v + Δy_emg ≈ y_v.
    """
    head_net = getattr(head, "net", None)
    if head_net is not None:
        last_linear = head_net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-5)
            nn.init.zeros_(last_linear.bias)


class MidFusionPoseFormer(BaseModule):
    """Residual fusion: vision single-pose baseline + EMG residual.

    Three fusion modes:
    - ``fusion``: full-window supervision.  vision → y_v, EMG → Δy_emg(T),
      preds = y_v + Δy_emg, supervised at all time steps.
    - ``emg_only`` / ``vision_only``: single-modality baselines.
    - ``center_supervised``: full EMG temporal window → attention-pooled →
      fused with vision → single center-frame prediction.  Same supervision
      target as vision-only (center frame only), fair cross-modal comparison.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        head: nn.Module,
        out_channels: int = 22,
        provide_initial_pos: bool = False,
        vision_embed_dim: int = 1280,
        emg_feat_dim: int | None = None,
        fusion_proj_dim: int = 256,
        vision_pretrained_checkpoint: str | None = None,
        mano_model_path: str | None = None,
        vision_freeze: dict[str, Any] | None = None,
        fusion_mode: str = "fusion",
        skip_vision_backbone: bool = False,
        vision_backbone_type: str = "vit",
        moddrop_prob: float = 0.0,
        input_emg_length: int | None = None,
        force_zero_emg: bool = False,
    ) -> None:
        super().__init__(
            featurizer=featurizer,
            decoder=decoder,
            out_channels=out_channels,
            provide_initial_pos=provide_initial_pos,
        )
        self.head = head
        self.vision_backbone_type = vision_backbone_type
        self.vision_freeze_cfg = vision_freeze or {
            "strategy": "simple",
            "frozen_block_end": 32,
            "freeze_patch_embed": True,
            "freeze_pos_embed": True,
            "freeze_mano_tokens": True,
            "finetune_last_norm": False,
        }
        if skip_vision_backbone:
            self.vision_backbone = nn.Identity()
        else:
            self.vision_backbone = self._build_vision_backbone(
                pretrained_path=vision_pretrained_checkpoint,
                mano_model_path=mano_model_path,
            )
            if vision_backbone_type == "vit" and vision_backbone_type not in _DINOV2_VARIANTS:
                apply_vit_freeze(self.vision_backbone, **self.vision_freeze_cfg)

        feat_dim = emg_feat_dim or getattr(self.featurizer, "out_channels", None)
        if feat_dim is None and hasattr(self.decoder, "input_proj"):
            input_proj = self.decoder.input_proj
            if isinstance(input_proj, nn.Linear):
                feat_dim = int(input_proj.in_features)
            elif isinstance(input_proj, nn.Identity) and hasattr(self.decoder, "output_proj"):
                feat_dim = int(self.decoder.output_proj.in_features)
        if feat_dim is None:
            raise ValueError("featurizer must expose out_channels for fusion")

        # ── vision_proj: backbone → fusion space (for fusion branch + gate only) ──
        self.vision_proj = nn.Linear(int(vision_embed_dim), int(fusion_proj_dim))
        fusion_in = int(feat_dim) + int(fusion_proj_dim)
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(fusion_in, fusion_in, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(fusion_in, int(feat_dim), kernel_size=1),
        )
        if fusion_mode not in {"fusion", "emg_only", "vision_only", "center_supervised"}:
            raise ValueError(
                f"fusion_mode must be fusion/emg_only/vision_only/center_supervised, "
                f"got {fusion_mode}"
            )
        self.fusion_mode = fusion_mode
        self.moddrop_prob = float(moddrop_prob)
        self.force_zero_emg = force_zero_emg

        if input_emg_length is not None and fusion_mode == "vision_only":
            self._vision_T = self.featurizer.get_output_length(int(input_emg_length))
        else:
            self._vision_T: int | None = None

        # ── head_vision: backbone features → y_v (single-pose baseline) ──
        # If a pretrained ResNetVisionPose checkpoint was loaded, build head_vision
        # from its head weights (same architecture, pretrained parameters).
        pretrained_head_sd = getattr(self, "_pretrained_head_state", None)
        if pretrained_head_sd is not None:
            head_hidden = pretrained_head_sd["0.weight"].shape[0]
            self.head_vision = nn.Sequential(
                nn.Linear(int(vision_embed_dim), head_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden, int(out_channels)),
            )
            self.head_vision.load_state_dict(pretrained_head_sd)
            del self._pretrained_head_state
            log.info("Loaded pretrained head_vision from vision checkpoint")
        else:
            self.head_vision = nn.Sequential(
                nn.Linear(int(vision_embed_dim), 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, int(out_channels)),
            )

        _init_residual_branch(self.head)

        # ── center_supervised: temporal attention pooling over EMG decoder ──
        if fusion_mode == "center_supervised":
            self.temporal_attn = TemporalAttentionPool(feat_dim)

    def _forward_center_supervised(
        self,
        batch: dict[str, torch.Tensor],
        vision_features: torch.Tensor,
        emg: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Center-frame-supervised fusion with full-EMG temporal context.

        EMG → featurizer → decoder → temporal attention pool → emg_pooled (B, C)
        vision → vision_proj → vis_feat (B, D)
        concat[emg_pooled, vis_feat] → fusion_proj → head → delta (B, 22, 1)
        preds = y_v + delta   ← supervised at center frame only
        """
        emg_features = self.featurizer(emg)
        decoded = self.decoder(emg_features)  # (B, C_emg, T')

        # Temporal attention pooling: learn which time steps inform center frame
        attn_scores = self.temporal_attn(decoded.transpose(1, 2))  # (B, T', 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T', 1)
        emg_pooled = (decoded * attn_weights.squeeze(-1).unsqueeze(1)).sum(dim=-1)  # (B, C)

        # Vision validity
        if "vision_valid_mask" in batch:
            vision_valid = batch["vision_valid_mask"]
            if vision_valid.ndim > 1:
                vision_valid = vision_valid.any(dim=1)
            vision_features = vision_features * vision_valid[:, None].to(
                vision_features.dtype
            )

        y_v = self.head_vision(vision_features)  # (B, out_channels)
        vis_feat = self.vision_proj(vision_features)  # (B, fusion_proj_dim)

        # Fuse pooled EMG with vision → single-frame delta
        fused = torch.cat([emg_pooled, vis_feat], dim=-1)  # (B, C + D)
        fused = fused.unsqueeze(-1)  # (B, C + D, 1)
        fused = self.fusion_proj(fused)  # (B, C_emg, 1)
        delta = self.head(fused)  # (B, 22, 1)
        self._last_delta = delta

        preds = y_v.unsqueeze(-1) + delta  # (B, 22, 1)

        if "joint_angles" in batch and "label_valid_mask" in batch:
            ja = batch["joint_angles"]
            mask = batch["label_valid_mask"]

            if ja.shape[-1] == 1:
                # Center-only labels (from center_supervised fast path)
                targets = ja  # (B, 22, 1)
                if mask.ndim >= 2:
                    mask_out = mask[..., :1]  # (B, 1)
                else:
                    mask_out = mask.unsqueeze(-1)
                return preds, targets, mask_out

            half_ctx = self.left_context // 2
            right_stop = -half_ctx if half_ctx > 0 else None
            targets_full = ja[..., half_ctx:right_stop]
            mask_full = mask[..., half_ctx:right_stop]

            center = targets_full.shape[-1] // 2
            targets = targets_full[:, :, center:center + 1]  # (B, 22, 1)

            if mask_full.ndim >= 3:
                mask_out = mask_full[:, :, center:center + 1]  # (B, C, 1)
            else:
                mask_out = mask_full[..., center:center + 1]  # (B, 1)

            return preds, targets, mask_out

        return preds

    def _forward_vision_only(
        self, batch: dict[str, torch.Tensor], vision_features: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch.nn.functional import interpolate as F_interpolate

        vision_valid = batch.get("vision_valid_mask")
        if vision_valid is not None:
            if vision_valid.ndim > 1:
                vision_valid = vision_valid.any(dim=1)
            vision_features = vision_features * vision_valid[:, None].to(
                vision_features.dtype
            )

        y_v = self.head_vision(vision_features)  # (B, out_channels)
        # vision_proj is still computed so its parameters exist for checkpoint loading,
        # but its output isn't used in vision_only mode.
        _ = self.vision_proj(vision_features)

        if self._vision_T is None:
            dummy = torch.zeros(1, batch["emg"].shape[1], batch["emg"].shape[2],
                                device=batch["emg"].device, dtype=batch["emg"].dtype)
            was_training = self.featurizer.training
            self.featurizer.eval()
            try:
                with torch.no_grad():
                    self._vision_T = self.featurizer(dummy).shape[-1]
            finally:
                if was_training:
                    self.featurizer.train()
        T = self._vision_T

        preds = y_v.unsqueeze(-1).expand(-1, -1, T)

        if "joint_angles" not in batch or "label_valid_mask" not in batch:
            return preds

        ja = batch["joint_angles"]
        mask = batch["label_valid_mask"]

        if ja.shape[-1] == 1:
            targets = torch.zeros(
                ja.shape[0], ja.shape[1], T,
                device=ja.device, dtype=ja.dtype,
            )
            center = T // 2
            targets[:, :, center] = ja.squeeze(-1)

            mask_out = torch.zeros(
                ja.shape[0], T, device=ja.device, dtype=torch.float32,
            )
            mask_out[:, center] = mask.squeeze(-1).float()

            return preds, targets, mask_out

        half_ctx = self.left_context // 2
        right_stop = -half_ctx if half_ctx > 0 else None
        targets = ja[..., half_ctx:right_stop]
        mask = mask[..., half_ctx:right_stop]

        targets = F_interpolate(targets, size=T, mode="linear", align_corners=False)
        mask = self.align_mask(mask, T)

        center = T // 2
        mask = torch.zeros_like(mask)
        if vision_valid is not None:
            mask[vision_valid, center] = 1.0
        else:
            mask[..., center] = 1.0

        return preds, targets, mask

    def _build_vision_backbone(
        self,
        *,
        pretrained_path: str | None,
        mano_model_path: str | None,
    ) -> nn.Module:
        # ── DINOv2 ViT backbone (timm) ──────────────────────────────────
        if self.vision_backbone_type in _DINOV2_VARIANTS:
            timm_name, backbone_dim = _DINOV2_VARIANTS[self.vision_backbone_type]
            import timm
            backbone = timm.create_model(
                timm_name, pretrained=True, num_classes=0, img_size=256,
            )
            self._dino_backbone_dim = backbone_dim
            if pretrained_path and Path(pretrained_path).exists():
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                backbonesd = {}
                headsd = {}
                for key, value in state_dict.items():
                    stripped = key[6:] if key.startswith("model.") else key
                    if stripped.startswith("backbone."):
                        backbonesd[stripped[len("backbone."):]] = value
                    elif stripped.startswith("head."):
                        headsd[stripped] = value
                if backbonesd:
                    missing, unexpected = backbone.load_state_dict(backbonesd, strict=False)
                    log.info(
                        "Loaded DINOv2 %s backbone from %s (%d missing, %d unexpected)",
                        self.vision_backbone_type, pretrained_path, len(missing), len(unexpected),
                    )
                if headsd:
                    self._pretrained_head_state = {
                        k[len("head."):]: v for k, v in headsd.items()
                    }
                    log.info(
                        "Found %d pretrained head keys in %s",
                        len(headsd), pretrained_path,
                    )
                else:
                    log.warning(
                        "No head weights found in %s — head_vision will be randomly initialized",
                        pretrained_path,
                    )
            return backbone

        if self.vision_backbone_type in _RESNET_DIMS:
            import torchvision.models as tv_models

            resnet_name = self.vision_backbone_type
            weights_cls = f"ResNet{resnet_name[6:].upper()}_Weights"
            weights = getattr(tv_models, weights_cls).IMAGENET1K_V1
            rn = getattr(tv_models, resnet_name)(weights=weights)
            backbone = nn.Sequential(
                rn.conv1, rn.bn1, rn.relu, rn.maxpool,
                rn.layer1, rn.layer2, rn.layer3, rn.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            if pretrained_path and Path(pretrained_path).exists():
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                backbonesd = {}
                headsd = {}
                for key, value in state_dict.items():
                    stripped = key[6:] if key.startswith("model.") else key
                    if stripped.startswith("backbone."):
                        backbonesd[stripped[len("backbone."):]] = value
                    elif stripped.startswith("head."):
                        headsd[stripped] = value
                if backbonesd:
                    missing, unexpected = backbone.load_state_dict(backbonesd, strict=False)
                    log.info(
                        "Loaded %s vision backbone from %s (%d missing, %d unexpected)",
                        resnet_name, pretrained_path, len(missing), len(unexpected),
                    )
                if headsd:
                    self._pretrained_head_state = {
                        k[len("head."):]: v for k, v in headsd.items()
                    }
                    log.info(
                        "Found %d pretrained head keys in %s",
                        len(headsd), pretrained_path,
                    )
            return backbone

        # ── ViT vision backbone ──────────────────────────────────────────
        from wilor.configs import get_config
        from wilor.models.backbones import vit

        model_config_path = str(WILOR_PATH / "pretrained_models" / "model_config.yaml")
        cfg = get_config(model_config_path, merge=True, update_cachedir=False)
        mano_data_dir = mano_model_path or str(WILOR_PATH / "mano_data")
        cfg.defrost()
        cfg.MANO.DATA_DIR = mano_data_dir
        cfg.MANO.MODEL_PATH = mano_data_dir
        cfg.MANO.MEAN_PARAMS = str(Path(mano_data_dir) / "mano_mean_params.npz")
        cfg.freeze()

        backbone = vit(cfg)
        if pretrained_path and Path(pretrained_path).exists():
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            backbone_state = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    backbone_state[key[len("backbone."):]] = value
            if backbone_state:
                missing, unexpected = backbone.load_state_dict(
                    backbone_state, strict=False
                )
                log.info(
                    "Loaded fusion vision backbone from %s (%d missing, %d unexpected)",
                    pretrained_path, len(missing), len(unexpected),
                )
        return backbone

    def _extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        has_time = images.ndim == 5
        if has_time:
            bsz, num_frames, channels, height, width = images.shape
            images = images.view(bsz * num_frames, channels, height, width)
        if self.vision_backbone_type == "vit":
            images = images[:, :, :, 32:-32]
        out = self.vision_backbone(images)
        # DINOv2 ViT returns (B, D) directly; WiLoR ViT returns tuple of feature maps
        if self.vision_backbone_type in _DINOV2_VARIANTS:
            feat = out  # (B, backbone_dim) from timm ViT
        elif isinstance(out, tuple):
            feat = None
            for item in reversed(out):
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    feat = item
                    break
            if feat is None:
                raise ValueError("vision backbone returned no 4D feature map")
            feat = feat.mean(dim=[-2, -1])
        else:
            feat = out
            if feat.ndim == 4:
                feat = feat.mean(dim=[-2, -1])
        if has_time:
            feat = feat.view(bsz, num_frames, -1).mean(dim=1)
        return feat

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "vision_features" in batch:
            vision_features = batch["vision_features"]
        elif "vision_img" in batch:
            vision_features = self._extract_vision_features(batch["vision_img"])
        else:
            raise ValueError("Batch must contain 'vision_features' or 'vision_img'")

        if self.fusion_mode == "vision_only":
            return self._forward_vision_only(batch, vision_features)

        emg = batch["emg"]

        if self.fusion_mode == "center_supervised":
            if self.force_zero_emg:
                emg = torch.zeros_like(emg)
            return self._forward_center_supervised(batch, vision_features, emg)

        if self.fusion_mode == "emg_only":
            vision_features = torch.zeros_like(vision_features)
        elif self.force_zero_emg:
            emg = torch.zeros_like(emg)
        elif self.training and self.moddrop_prob > 0:
            mod_mask = torch.rand(emg.shape[0], device=emg.device) < self.moddrop_prob
            emg = emg.clone()
            emg[mod_mask] = 0

        emg_features = self.featurizer(emg)
        decoded = self.decoder(emg_features)  # (B, C_emg, T)

        if "vision_valid_mask" in batch:
            vision_valid = batch["vision_valid_mask"]
            if vision_valid.ndim > 1:
                vision_valid = vision_valid.any(dim=1)
            vision_features = vision_features * vision_valid[:, None].to(
                dtype=vision_features.dtype
            )

        # Vision baseline: head_vision takes raw backbone features → y_v
        y_v = self.head_vision(vision_features)  # (B, out_channels)

        # Fusion branch: vision_proj projects backbone → fusion space
        vis_feat = self.vision_proj(vision_features)  # (B, fusion_proj_dim)

        # EMG residual: concat vision context with EMG decoding, fuse, predict delta
        vis_feat_t = vis_feat.unsqueeze(-1).expand(-1, -1, decoded.shape[-1])
        fused = torch.cat([decoded, vis_feat_t], dim=1)
        fused = self.fusion_proj(fused)
        delta = self.head(fused)
        self._last_delta = delta

        if self.fusion_mode == "emg_only":
            preds = delta
        else:
            preds = y_v.unsqueeze(-1) + delta

        if "joint_angles" in batch and "label_valid_mask" in batch:
            from torch.nn.functional import interpolate as F_interpolate

            ja = batch["joint_angles"]
            mask = batch["label_valid_mask"]
            half_ctx = self.left_context // 2
            right_stop = -half_ctx if half_ctx > 0 else None
            targets = ja[..., half_ctx:right_stop]
            mask = mask[..., half_ctx:right_stop]
            n_time = preds.shape[-1]
            targets = F_interpolate(targets, size=n_time, mode="linear", align_corners=False)
            mask = self.align_mask(mask, n_time)
            return preds, targets, mask

        return preds
