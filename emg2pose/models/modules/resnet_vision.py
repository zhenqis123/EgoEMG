"""ResNet vision-only baselines for hand pose estimation.

Accepts pre-cropped RGB patches from the dataset, runs them through
a pretrained ResNet backbone, and predicts 22 joint angles.
"""

from __future__ import annotations

import torch
from torch import nn


_RESNET_BUILDERS = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
}


class ResNetVisionPose(nn.Module):
    """ResNet → Global Pool → MLP → 22 joint angles.

    Vision-only baseline. One image predicts one pose — the dataset already
    returns center-frame-only targets ``(B, 22, 1)``.
    """

    def __init__(
        self,
        out_channels: int = 22,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        backbone_type: str = "resnet18",
        head_hidden: int = 512,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.backbone_type = backbone_type

        if backbone_type not in _RESNET_BUILDERS:
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}. "
                f"Choose from {list(_RESNET_BUILDERS)}."
            )

        import torchvision.models as tv_models

        resnet_fn, weights_cls = _RESNET_BUILDERS[backbone_type]
        weights = getattr(tv_models, weights_cls).IMAGENET1K_V1 if pretrained else None
        resnet = getattr(tv_models, resnet_fn)(weights=weights)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        backbone_dim = _RESNET_DIMS[backbone_type]

        self.left_context = 0
        self.right_context = 0

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
        img = batch["vision_img"]  # (B, 3, H, W) already ImageNet-normalized

        if img.ndim == 5:
            img = img.mean(dim=1)  # (B, T_img, 3, H, W) → (B, 3, H, W)

        feat = self.backbone(img)
        feat = self.avgpool(feat).flatten(1)
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


# Backward-compat alias
ResNet18VisionPose = ResNetVisionPose
