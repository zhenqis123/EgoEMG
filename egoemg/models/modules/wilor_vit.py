"""WiLoR ViT vision-only baseline for hand pose estimation.

Accepts pre-cropped RGB patches from the dataset, runs them through
a pretrained WiLoR ViT backbone, and predicts 22 joint angles.

Follows the same pattern as ResNetVisionPose — one image predicts one pose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger(__name__)

# Resolve the WiLoR dependency.  Prefer an explicit override via the WILOR_PATH
# environment variable; otherwise fall back to a sibling directory next to this
# repository (``../WiLoR``), which is the documented install layout.
_DEFAULT_WILOR = Path(__file__).resolve().parents[3] / ".." / "WiLoR"
WILOR_PATH = Path(os.environ.get("WILOR_PATH", str(_DEFAULT_WILOR)))

_MANO_COMPONENTS = [
    "pose_emb", "shape_emb", "cam_emb", "decpose", "decshape", "deccam",
]


class WiLoRViTPose(nn.Module):
    """WiLoR ViT → Global Pool → MLP → 22 joint angles.

    Vision-only baseline. One image predicts one pose — the dataset already
    returns center-frame-only targets ``(B, 22, 1)``.
    """

    def __init__(
        self,
        out_channels: int = 22,
        pretrained_path: str | None = None,
        head_hidden: int = 512,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.left_context = 0
        self.right_context = 0

        self.backbone = self._build_backbone(pretrained_path)
        self._freeze_mano_components()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(1280, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, out_channels),
        )

    def _build_backbone(self, pretrained_path: str | None) -> nn.Module:
        import sys
        if str(WILOR_PATH) not in sys.path:
            sys.path.insert(0, str(WILOR_PATH))

        from wilor.configs import get_config
        from wilor.models.backbones import vit

        cfg_path = WILOR_PATH / "pretrained_models" / "model_config.yaml"
        cfg = get_config(str(cfg_path), merge=True, update_cachedir=False)
        cfg.defrost()
        cfg.MANO.DATA_DIR = str(WILOR_PATH / "mano_data")
        cfg.MANO.MODEL_PATH = str(WILOR_PATH / "mano_data")
        cfg.MANO.MEAN_PARAMS = str(WILOR_PATH / "mano_data" / "mano_mean_params.npz")
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
                    backbone_state, strict=False,
                )
                log.info(
                    "Loaded WiLoR ViT backbone (matched %d/%d, %d missing, %d unexpected)",
                    len(backbone_state) - len(missing),
                    len(backbone.state_dict()),
                    len(missing),
                    len(unexpected),
                )
        return backbone

    def _freeze_mano_components(self) -> None:
        for name in _MANO_COMPONENTS:
            mod = getattr(self.backbone, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

    def _extract_feature(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[2] < 64:
            raise ValueError(
                "WiLoR expects 256px-tall inputs (crops 32px top/bottom to the "
                f"192px PE grid); got height {images.shape[2]}"
            )
        images = images[:, :, 32:-32, :]  # 256→192 height (WiLoR ViT PE grid)
        out = self.backbone(images)
        # out is (pred_mano_params, pred_cam, pred_mano_feats, img_feat)
        if isinstance(out, tuple):
            feat = None
            for item in reversed(out):
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    feat = item
                    break
            if feat is None:
                raise ValueError("WiLoR ViT backbone returned no 4D feature map")
        else:
            feat = out
        return self.avgpool(feat).flatten(1)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch["vision_img"]  # (B, 3, 256, 256) already ImageNet-normalized

        if img.ndim == 5:
            img = img.mean(dim=1)  # (B, T_img, 3, 256, 256) → (B, 3, 256, 256)

        feat = self._extract_feature(img)
        preds = self.head(feat).unsqueeze(-1)  # (B, 22, 1)

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
