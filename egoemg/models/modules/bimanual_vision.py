"""One full RGB frame, one shared DINO backbone, ordered left/right angle head."""

from egoemg.models.modules.vit_vision import VisionViTPose
from egoemg.models.modules.resnet_vision import ResNetVisionPose


class BimanualVisionViTPose(VisionViTPose):
    bimanual_output = True

    def __init__(self, **kwargs):
        kwargs["out_channels"] = 44
        super().__init__(**kwargs)

    def forward(self, batch):
        out = super().forward(batch)
        if isinstance(out, tuple):
            predictions, targets, mask = out
            return (predictions.reshape(-1, 22, 1), targets.reshape(-1, 22, 1),
                    mask.reshape(-1, 22, 1))
        return out.reshape(-1, 22, 1)


class BimanualVisionResNetPose(ResNetVisionPose):
    """ResNet with the same ordered 44-angle contract as ViT."""

    bimanual_output = True

    def __init__(self, **kwargs):
        kwargs['out_channels'] = 44
        super().__init__(**kwargs)
        # Fusion consumes backbone(image) directly, so include global pooling
        # in the backbone and retain an identity pool for the inherited forward.
        from torch import nn

        self.backbone = nn.Sequential(*self.backbone.children(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.avgpool = nn.Identity()

    def forward(self, batch):
        out = super().forward(batch)
        if isinstance(out, tuple):
            return tuple(value.reshape(-1, 22, 1) for value in out)
        return out.reshape(-1, 22, 1)


def expand_bimanual_metadata(batch):
    """Match [L0,R0,L1,R1,...] outputs without treating hands as time samples."""
    import torch

    result = {}
    for key, value in batch.items():
        if key in {"vision_img", "joint_angles", "label_valid_mask"}:
            continue
        if isinstance(value, torch.Tensor):
            if key == "emg" and value.ndim == 4:
                result[key] = value.flatten(0, 1)
            elif key in {"split_id", "object_id", "emg_valid_mask",
                       "emg_start_row", "window_offset_samples"}:
                result[key] = value.reshape(-1)
            else:
                result[key] = value.repeat_interleave(2, dim=0)
        elif isinstance(value, (list, tuple)):
            result[key] = [item for item in value for _ in range(2)]
        else:
            result[key] = value
    return result
