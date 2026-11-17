"""Single source of truth for vision backbone variant → (timm name, embed dim).

Imported by mid_fusion.py, vit_vision.py, resnet_vision.py so the variant
mappings stay consistent across all fusion / vision modules.
"""

# DINOv2 ViT variants. Keyed by the short name used in configs
# (e.g. `vision_backbone_type: vit_small`). Value: (timm model name, embed dim).
DINOV2_VARIANTS = {
    "vit_small": ("vit_small_patch14_dinov2", 384),
    "vit_base": ("vit_base_patch14_dinov2", 768),
    "vit_large": ("vit_large_patch14_dinov2", 1024),
    "vit_huge": ("vit_giant_patch14_dinov2", 1536),
}

# ResNet variants → output embed dim.
RESNET_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet152": 2048,
}
