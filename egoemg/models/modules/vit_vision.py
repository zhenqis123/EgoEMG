"""ViT vision-only baseline for hand pose estimation.

Structurally identical to ResNetVisionPose: backbone → head → 22 joint angles.
Uses DINOv2-pretrained ViTs from timm.
One image predicts one pose — the dataset returns center-frame-only targets ``(B, 22, 1)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from egoemg.models.modules._vision_constants import DINOV2_VARIANTS as _DINOV2_VARIANTS


# DINOv2 ViT variants from timm


class VisionViTPose(nn.Module):
    """DINOv2 ViT → MLP head → 22 joint angles.

    Vision-only baseline. One image predicts one pose.
    """

    def __init__(
        self,
        out_channels: int = 22,
        pretrained: bool = True,
        backbone_type: str = "vit_base",
        freeze_backbone: bool = False,
        head_hidden: int = 512,
        head_dropout: float = 0.1,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.backbone_type = backbone_type

        if backbone_type not in _DINOV2_VARIANTS:
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}. "
                f"Choose from {list(_DINOV2_VARIANTS)}."
            )

        timm_name, backbone_dim = _DINOV2_VARIANTS[backbone_type]

        import timm
        from huggingface_hub import try_to_load_from_cache

        local_weights = None
        if pretrained and pretrained_path:
            local_weights = Path(pretrained_path).expanduser()
            if not local_weights.is_file():
                raise FileNotFoundError(local_weights)
        elif pretrained:
            cfg = timm.models.get_pretrained_cfg(timm_name)
            if cfg.hf_hub_id:
                for filename in ("model.safetensors", "pytorch_model.bin"):
                    cached = try_to_load_from_cache(cfg.hf_hub_id, filename)
                    if isinstance(cached, str) and Path(cached).is_file():
                        local_weights = Path(cached)
                        break
        kwargs = {}
        if local_weights is not None:
            logging.getLogger(__name__).info(
                "Loading %s pretrained weights locally: %s", timm_name, local_weights
            )
            # Keep timm's checkpoint filter, including positional interpolation
            # from the pretrained resolution to the 256px crop resolution.
            kwargs["pretrained_cfg_overlay"] = {"file": str(local_weights)}
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, img_size=256, **kwargs
        )

        self.left_context = 0
        self.right_context = 0
        # Session-v2 transports compact RGB bytes; normalize once on the device.
        # Nonpersistent buffers preserve compatibility with released checkpoints.
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(backbone_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, out_channels),
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch["vision_img"]  # (B, 3, H, W)

        if img.dtype == torch.uint8:
            img = img.float().div_(255).sub_(self.image_mean).div_(self.image_std)

        if img.ndim == 5:
            img = img.mean(dim=1)  # (B, T_img, 3, H, W) → (B, 3, H, W)

        feat = self.backbone(img)  # (B, backbone_dim)
        preds = self.head(feat).unsqueeze(-1)  # (B, 22, 1)

        if "joint_angles" not in batch:
            return preds

        targets = batch["joint_angles"]
        if targets.ndim == 2:
            targets = targets.unsqueeze(-1)

        mask = batch["label_valid_mask"]
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)

        return preds, targets, mask

    @staticmethod
    def align_mask(mask: torch.Tensor, n_time: int) -> torch.Tensor:
        return mask
