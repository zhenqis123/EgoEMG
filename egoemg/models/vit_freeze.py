"""Shared ViT freeze and param-group utilities for vision baseline and fusion models.

WiLoR ViT attribute map (from wilor/models/backbones/vit.py):
  - patch_embed: PatchEmbed (Conv2d projection)
  - pos_embed: nn.Parameter (1, num_patches+1, embed_dim)
  - blocks: nn.ModuleList of Block, len=depth (32 for WiLoR)
  - last_norm: LayerNorm(embed_dim)
  - pose_emb, shape_emb, cam_emb: nn.Linear (MANO token embeddings)
  - decpose, decshape, deccam: nn.Linear (MANO decode heads)
  - init_hand_pose, init_betas, init_cam: registered buffers
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

log = logging.getLogger(__name__)


def apply_vit_freeze(
    backbone: nn.Module,
    strategy: str = "tiered",
    frozen_block_end: int = 30,
    freeze_patch_embed: bool = True,
    freeze_pos_embed: bool = True,
    freeze_mano_tokens: bool = True,
    finetune_last_norm: bool = True,
    **kwargs: Any,
) -> None:
    """Apply freeze strategy to a WiLoR ViT backbone in-place.

    Args:
        backbone: WiLoR ViT module.
        strategy: "tiered" or "simple".
        frozen_block_end: Freeze blocks[0:frozen_block_end].
        freeze_patch_embed: Freeze patch_embed conv.
        freeze_pos_embed: Freeze pos_embed parameter.
        freeze_mano_tokens: Freeze pose_emb, shape_emb, cam_emb, decpose, decshape, deccam.
        finetune_last_norm: If False, freeze last_norm too (used by simple strategy).
    """
    frozen_count = 0
    total_count = 0

    # --- Patch embedding ---
    if freeze_patch_embed and hasattr(backbone, "patch_embed"):
        for p in backbone.patch_embed.parameters():
            p.requires_grad = False
            frozen_count += p.numel()

    # --- Positional embedding ---
    if freeze_pos_embed and hasattr(backbone, "pos_embed") and backbone.pos_embed is not None:
        backbone.pos_embed.requires_grad = False
        frozen_count += backbone.pos_embed.numel()

    # --- MANO token embeddings and decode heads ---
    mano_modules = ["pose_emb", "shape_emb", "cam_emb", "decpose", "decshape", "deccam"]
    if freeze_mano_tokens:
        for name in mano_modules:
            mod = getattr(backbone, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
                    frozen_count += p.numel()

    # --- Transformer blocks ---
    blocks = getattr(backbone, "blocks", [])
    for i, block in enumerate(blocks):
        if i < frozen_block_end:
            for p in block.parameters():
                p.requires_grad = False
                frozen_count += p.numel()

    # --- last_norm ---
    if not finetune_last_norm and hasattr(backbone, "last_norm"):
        for p in backbone.last_norm.parameters():
            p.requires_grad = False
            frozen_count += p.numel()

    for p in backbone.parameters():
        total_count += p.numel()

    trainable = total_count - frozen_count
    log.info(
        "ViT freeze [%s]: %d/%d params frozen, %d trainable "
        "(blocks[0:%d] frozen, last_norm %s)",
        strategy,
        frozen_count,
        total_count,
        trainable,
        frozen_block_end,
        "trainable" if finetune_last_norm else "frozen",
    )


def get_vit_param_groups(
    backbone: nn.Module,
    strategy: str = "tiered",
    frozen_block_end: int = 30,
    finetune_block_start: int = 30,
    finetune_block_end: int = 32,
    finetune_last_norm: bool = True,
    vit_finetune_lr: float | None = 1e-5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Build optimizer param groups for the ViT backbone.

    For "tiered" strategy, returns a param group for the fine-tuned top blocks
    and last_norm at vit_finetune_lr. For "simple", returns an empty list
    (all trainable ViT params are frozen, so nothing to add).

    Only returns params with requires_grad=True.
    """
    if strategy == "simple" or vit_finetune_lr is None:
        return []

    finetune_params: list[nn.Parameter] = []
    blocks = getattr(backbone, "blocks", [])
    for i in range(finetune_block_start, min(finetune_block_end, len(blocks))):
        for p in blocks[i].parameters():
            if p.requires_grad:
                finetune_params.append(p)

    if finetune_last_norm and hasattr(backbone, "last_norm"):
        for p in backbone.last_norm.parameters():
            if p.requires_grad:
                finetune_params.append(p)

    if not finetune_params:
        return []

    param_count = sum(p.numel() for p in finetune_params)
    log.info(
        "ViT finetune group: %d params at lr=%.1e (blocks[%d:%d]%s)",
        param_count,
        vit_finetune_lr,
        finetune_block_start,
        finetune_block_end,
        " + last_norm" if finetune_last_norm else "",
    )
    return [{"params": finetune_params, "lr": vit_finetune_lr}]
