from __future__ import annotations

import logging
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from egoemg.models.modules.base import BaseModule
from egoemg.models.modules._vision_constants import DINOV2_VARIANTS as _DINOV2_VARIANTS, RESNET_DIMS as _RESNET_DIMS
from egoemg.models.modules._pooling import TemporalAttentionPool
from egoemg.models.vit_freeze import apply_vit_freeze


log = logging.getLogger(__name__)

# Resolve the WiLoR dependency.  Prefer an explicit override via the WILOR_PATH
# environment variable; otherwise fall back to a sibling directory next to this
# repository (``../WiLoR``), which is the documented install layout.
_DEFAULT_WILOR = Path(__file__).resolve().parents[3] / ".." / "WiLoR"
WILOR_PATH = Path(os.environ.get("WILOR_PATH", str(_DEFAULT_WILOR)))
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


class JointTokenFusionEncoder(nn.Module):
    """Fuse temporal EMG tokens and spatial CNN tokens with self-attention.

    EMG tokens receive sinusoidal temporal positions, while CNN tokens receive
    factorized 2-D positions.  A learned pose token reads the fused sequence.
    Keeping the two position systems separate avoids imposing an artificial
    temporal ordering on image patches.
    """

    def __init__(
        self,
        emg_dim: int,
        vision_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        max_vision_grid_size: int = 16,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by token_num_heads")
        self.emg_proj = (
            nn.Identity() if emg_dim == token_dim else nn.Linear(emg_dim, token_dim)
        )
        self.vision_proj = nn.Linear(vision_dim, token_dim)
        self.pose_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.modality_embed = nn.Parameter(torch.zeros(2, 1, token_dim))
        self.vision_row_embed = nn.Parameter(
            torch.zeros(max_vision_grid_size, token_dim)
        )
        self.vision_col_embed = nn.Parameter(
            torch.zeros(max_vision_grid_size, token_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pose_token, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)
        nn.init.normal_(self.vision_row_embed, std=0.02)
        nn.init.normal_(self.vision_col_embed, std=0.02)

    @staticmethod
    def _temporal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10_000.0) / max(dim, 1))
        )
        encoding = torch.zeros(length, dim, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position[:, None] * frequencies)
        encoding[:, 1::2] = torch.cos(position[:, None] * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(
        self,
        emg_features: torch.Tensor,
        vision_map: torch.Tensor,
        vision_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the fused pose token.

        Args:
            emg_features: ``(B, C_emg, T_emg)``.
            vision_map: ``(B, C_vision, H, W)`` from ResNet layer4.
            vision_valid: optional ``(B,)`` boolean mask.
        """
        batch_size, _, emg_length = emg_features.shape
        _, _, height, width = vision_map.shape
        if height > self.vision_row_embed.shape[0] or width > self.vision_col_embed.shape[0]:
            raise ValueError(
                f"vision grid {height}x{width} exceeds configured maximum "
                f"{self.vision_row_embed.shape[0]}x{self.vision_col_embed.shape[0]}"
            )

        emg_tokens = self.emg_proj(emg_features.transpose(1, 2))
        emg_tokens = emg_tokens + self._temporal_positions(
            emg_length, emg_tokens.shape[-1], emg_tokens.device, emg_tokens.dtype
        ).unsqueeze(0)
        emg_tokens = emg_tokens + self.modality_embed[0]

        vision_tokens = vision_map.flatten(2).transpose(1, 2)
        vision_tokens = self.vision_proj(vision_tokens)
        vision_pos = (
            self.vision_row_embed[:height, None, :]
            + self.vision_col_embed[None, :width, :]
        ).reshape(1, height * width, -1)
        vision_tokens = vision_tokens + vision_pos + self.modality_embed[1]

        pose_token = self.pose_token.expand(batch_size, -1, -1)
        tokens = torch.cat((pose_token, emg_tokens, vision_tokens), dim=1)
        padding_mask = None
        if vision_valid is not None:
            invalid_vision = (~vision_valid.bool())[:, None].expand(-1, height * width)
            padding_mask = torch.cat(
                (
                    torch.zeros(batch_size, 1 + emg_length, device=tokens.device, dtype=torch.bool),
                    invalid_vision,
                ),
                dim=1,
            )
        return self.norm(self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 0])


class VisualCrossAttentionRefinementBlock(nn.Module):
    """One pre-norm visual cross-attention refinement of EMG tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emg_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.cross_attention.out_proj.weight, std=1e-5)
        nn.init.zeros_(self.cross_attention.out_proj.bias)
        nn.init.normal_(self.ffn[-2].weight, std=1e-5)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(
        self,
        emg_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        vision_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            self.emg_norm(emg_tokens),
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        if vision_valid is not None:
            attended = attended * vision_valid[:, None, None].to(attended.dtype)
        fused = emg_tokens + self.attention_dropout(attended)
        return fused + self.ffn(self.ffn_norm(fused))


class FrozenVisualCrossAttention(nn.Module):
    """Inject frozen multi-scale ResNet features into temporal EMG tokens.

    Layer3 is pooled to the layer4 grid, both maps are projected to the EMG
    width, and the EMG sequence cross-attends to the resulting visual tokens.
    The attention output projection starts near zero, so this block initially
    preserves the pretrained EMG representation while remaining trainable.
    """

    def __init__(
        self,
        emg_dim: int,
        layer3_dim: int,
        layer4_dim: int,
        fusion_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_vision_grid_size: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if fusion_dim != emg_dim:
            raise ValueError(
                "frozen_cross_attention requires fusion_dim == emg_feat_dim"
            )
        if fusion_dim % num_heads:
            raise ValueError("fusion_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.layer3_proj = nn.Conv2d(layer3_dim, fusion_dim, kernel_size=1)
        self.layer4_proj = nn.Conv2d(layer4_dim, fusion_dim, kernel_size=1)
        self.visual_norm = nn.LayerNorm(fusion_dim)
        self.emg_norm = nn.LayerNorm(fusion_dim)
        self.vision_row_embed = nn.Parameter(
            torch.zeros(max_vision_grid_size, fusion_dim)
        )
        self.vision_col_embed = nn.Parameter(
            torch.zeros(max_vision_grid_size, fusion_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            fusion_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, fusion_dim),
            nn.Dropout(dropout),
        )
        # Keep the original first-block parameter names checkpoint-compatible.
        # Additional blocks start close to identity through near-zero output
        # projections, so a deeper variant initially preserves the proven
        # single-block fusion path.
        self.extra_blocks = nn.ModuleList(
            VisualCrossAttentionRefinementBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers - 1)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.vision_row_embed, std=0.02)
        nn.init.normal_(self.vision_col_embed, std=0.02)
        # Preserve the pretrained EMG path at initialization without blocking
        # gradients as an exactly-zero projection would.
        nn.init.normal_(self.cross_attention.out_proj.weight, std=1e-5)
        nn.init.zeros_(self.cross_attention.out_proj.bias)
        last_linear = self.ffn[-2]
        nn.init.normal_(last_linear.weight, std=1e-5)
        nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        emg_features: torch.Tensor,
        layer3_map: torch.Tensor,
        layer4_map: torch.Tensor,
        vision_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return visually conditioned EMG tokens with shape ``(B, C, T)``."""
        height, width = layer4_map.shape[-2:]
        max_grid = self.vision_row_embed.shape[0]
        if height > max_grid or width > max_grid:
            raise ValueError(
                f"vision grid {height}x{width} exceeds configured maximum "
                f"{max_grid}x{max_grid}"
            )

        layer3 = nn.functional.adaptive_avg_pool2d(
            self.layer3_proj(layer3_map), (height, width)
        )
        visual = layer3 + self.layer4_proj(layer4_map)
        visual = visual.flatten(2).transpose(1, 2)
        visual_pos = (
            self.vision_row_embed[:height, None, :]
            + self.vision_col_embed[None, :width, :]
        ).reshape(1, height * width, -1)
        visual = self.visual_norm(visual + visual_pos)

        emg_tokens = emg_features.transpose(1, 2)
        attended, _ = self.cross_attention(
            self.emg_norm(emg_tokens),
            visual,
            visual,
            need_weights=False,
        )
        if vision_valid is not None:
            attended = attended * vision_valid[:, None, None].to(attended.dtype)
        fused = emg_tokens + self.attention_dropout(attended)
        fused = fused + self.ffn(self.ffn_norm(fused))
        for block in self.extra_blocks:
            fused = block(fused, visual, vision_valid)
        return fused.transpose(1, 2)


def make_invalid_anchor_emg(
    emg: torch.Tensor,
    target_hand_index: torch.Tensor | None,
    shuffle_fraction: float,
) -> torch.Tensor:
    """Build zero/mismatched EMG negatives without crossing hand layouts."""
    if not 0.0 <= shuffle_fraction <= 1.0:
        raise ValueError("shuffle_fraction must be in [0, 1]")
    anchor_emg = torch.zeros_like(emg)
    if shuffle_fraction == 0.0 or emg.shape[0] <= 1:
        return anchor_emg

    identity = torch.arange(emg.shape[0], device=emg.device)
    permutation = identity.clone()
    if target_hand_index is None:
        permutation = torch.roll(permutation, shifts=1)
    else:
        hands = target_hand_index.reshape(-1)
        for hand in hands.unique():
            indices = torch.where(hands == hand)[0]
            if indices.numel() > 1:
                permutation[indices] = torch.roll(indices, shifts=1)
    use_shuffled = (
        torch.rand(emg.shape[0], device=emg.device) < shuffle_fraction
    ) & (permutation != identity)
    anchor_emg[use_shuffled] = emg[permutation[use_shuffled]]
    return anchor_emg


class MidFusionPoseFormer(BaseModule):
    """Residual fusion: vision single-pose baseline + EMG residual.

    Fusion modes:
    - ``fusion``: full-window supervision.  vision → y_v, EMG → Δy_emg(T),
      preds = y_v + Δy_emg, supervised at all time steps.
    - ``emg_only`` / ``vision_only``: single-modality baselines.
    - ``center_supervised``: full EMG temporal window → attention-pooled →
      fused with vision → single center-frame prediction.  Same supervision
      target as vision-only (center frame only), fair cross-modal comparison.
    - ``token_self_attention``: TDS EMG tokens and ResNet layer4 spatial
      tokens are concatenated and processed by a shared Transformer.
    - ``frozen_cross_attention``: temporal EMG tokens query spatial vision
      tokens before the EMG decoder; ResNets use layer3/layer4 maps while ViTs
      use their patch-token grid.  A center-token residual is added to the
      unchanged vision-only prediction.
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
        freeze_vision_branch: bool = False,
        vision_trainable_prefixes: list[str] | None = None,
        freeze_vision_head: bool = False,
        lock_vision_batch_norm: bool = False,
        token_dim: int = 256,
        token_num_heads: int = 8,
        token_num_layers: int = 3,
        token_ffn_dim: int = 512,
        token_dropout: float = 0.1,
        token_max_vision_grid_size: int = 16,
        cross_attention_dim: int = 256,
        cross_attention_num_heads: int = 8,
        cross_attention_ffn_dim: int = 512,
        cross_attention_dropout: float = 0.1,
        cross_attention_max_vision_grid_size: int = 16,
        cross_attention_num_layers: int = 1,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__(
            featurizer=featurizer,
            decoder=decoder,
            out_channels=out_channels,
            provide_initial_pos=provide_initial_pos,
        )
        self.head = head
        self.vision_backbone_type = vision_backbone_type
        if fusion_mode not in {
            "fusion",
            "emg_only",
            "vision_only",
            "center_supervised",
            "token_self_attention",
            "frozen_cross_attention",
        }:
            raise ValueError(
                "fusion_mode must be fusion/emg_only/vision_only/center_supervised/"
                f"token_self_attention/frozen_cross_attention, got {fusion_mode}"
            )
        if (
            fusion_mode == "token_self_attention"
            and vision_backbone_type not in _RESNET_DIMS
        ):
            raise ValueError("token_self_attention currently requires a ResNet backbone")
        spatial_fusion_modes = {"token_self_attention", "frozen_cross_attention"}
        if fusion_mode in spatial_fusion_modes and skip_vision_backbone:
            raise ValueError(f"{fusion_mode} requires vision images, not cached global features")
        self.fusion_mode = fusion_mode
        self.freeze_vision_branch = bool(freeze_vision_branch)
        self.vision_trainable_prefixes = vision_trainable_prefixes
        self.freeze_vision_head = bool(freeze_vision_head)
        self.lock_vision_batch_norm = bool(lock_vision_batch_norm)
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
        self.moddrop_prob = float(moddrop_prob)
        self.force_zero_emg = force_zero_emg
        self.residual_scale = float(residual_scale)
        if self.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")

        if input_emg_length is not None and fusion_mode == "vision_only":
            self._vision_T = self.featurizer.get_output_length(int(input_emg_length))
        else:
            self._vision_T: int | None = None

        # The token-fusion variant is a single, direct pose decoder: the joint
        # pose token is decoded by ``head`` with no vision-only bypass.
        # Other variants retain the vision baseline plus EMG residual design.
        if fusion_mode == "token_self_attention":
            self.head_vision = nn.Identity()
        else:
            # ── head_vision: backbone features → y_v (single-pose baseline) ──
            # If a pretrained ResNetVisionPose checkpoint was loaded, build
            # head_vision from its head weights (same architecture).
            pretrained_head_sd = getattr(self, "_pretrained_head_state", None)
            if pretrained_head_sd is not None:
                head_hidden = pretrained_head_sd["0.weight"].shape[0]
                # Preserve the exact pretrained vision-head architecture.
                # WiLoRViTPose uses GELU, while ResNetVisionPose and the
                # DINOv2 VisionViTPose baselines use ReLU.
                vision_head_activation: nn.Module = (
                    nn.GELU()
                    if self.vision_backbone_type == "vit"
                    else nn.ReLU()
                )
                self.head_vision = nn.Sequential(
                    nn.Linear(int(vision_embed_dim), head_hidden),
                    vision_head_activation,
                    nn.Dropout(0.1),
                    nn.Linear(head_hidden, int(out_channels)),
                )
                self.head_vision.load_state_dict(pretrained_head_sd)
                del self._pretrained_head_state
                log.info("Loaded pretrained head_vision from vision checkpoint")
            else:
                vision_head_activation = (
                    nn.GELU()
                    if self.vision_backbone_type == "vit"
                    else nn.ReLU()
                )
                self.head_vision = nn.Sequential(
                    nn.Linear(int(vision_embed_dim), 512),
                    vision_head_activation,
                    nn.Dropout(0.1),
                    nn.Linear(512, int(out_channels)),
                )
            _init_residual_branch(self.head)

        if self.freeze_vision_branch:
            if fusion_mode == "token_self_attention":
                raise ValueError(
                    "freeze_vision_branch is only valid for residual fusion modes"
                )
            for module in (self.vision_backbone, self.head_vision):
                for parameter in module.parameters():
                    parameter.requires_grad = False
                # Keep ResNet BatchNorm statistics and vision-head dropout fixed
                # even after Lightning calls model.train() for fusion training.
                module.eval()
            log.info(
                "Frozen vision backbone and head_vision; their BatchNorm/dropout "
                "state remains in eval mode during fusion training"
            )
        elif self.vision_trainable_prefixes is not None:
            prefixes = tuple(self.vision_trainable_prefixes)
            for name, parameter in self.vision_backbone.named_parameters():
                parameter.requires_grad = name.startswith(prefixes)
            trainable = sum(
                parameter.numel()
                for parameter in self.vision_backbone.parameters()
                if parameter.requires_grad
            )
            if trainable == 0:
                raise ValueError(
                    "vision_trainable_prefixes matched no backbone parameters: "
                    f"{self.vision_trainable_prefixes}"
                )
            log.info(
                "Selectively unfroze vision prefixes %s (%d parameters)",
                self.vision_trainable_prefixes,
                trainable,
            )

        if self.freeze_vision_head and not self.freeze_vision_branch:
            for parameter in self.head_vision.parameters():
                parameter.requires_grad = False
            self.head_vision.eval()
            log.info("Frozen vision pose head and locked it in eval mode")

        if self.lock_vision_batch_norm:
            self._lock_vision_batch_norm_eval()
            log.info("Locked all vision BatchNorm modules in eval mode")

        # ── center_supervised: temporal attention pooling over EMG decoder ──
        if fusion_mode == "center_supervised":
            self.temporal_attn = TemporalAttentionPool(feat_dim)
        elif fusion_mode == "frozen_cross_attention":
            layer3_dims = {
                "resnet18": 256,
                "resnet34": 256,
                "resnet50": 1024,
                "resnet152": 1024,
            }
            early_vision_dim = layer3_dims.get(
                vision_backbone_type, int(vision_embed_dim)
            )
            self.early_fusion = FrozenVisualCrossAttention(
                emg_dim=int(feat_dim),
                layer3_dim=early_vision_dim,
                layer4_dim=int(vision_embed_dim),
                fusion_dim=int(cross_attention_dim),
                num_heads=int(cross_attention_num_heads),
                ffn_dim=int(cross_attention_ffn_dim),
                dropout=float(cross_attention_dropout),
                max_vision_grid_size=int(cross_attention_max_vision_grid_size),
                num_layers=int(cross_attention_num_layers),
            )
            # Multi-scale token fusion replaces the old global-feature concat.
            self.vision_proj = nn.Identity()
            self.fusion_proj = nn.Identity()
        elif fusion_mode == "token_self_attention":
            self.token_fusion = JointTokenFusionEncoder(
                emg_dim=int(feat_dim),
                vision_dim=int(vision_embed_dim),
                token_dim=int(token_dim),
                num_heads=int(token_num_heads),
                num_layers=int(token_num_layers),
                ffn_dim=int(token_ffn_dim),
                dropout=float(token_dropout),
                max_vision_grid_size=int(token_max_vision_grid_size),
            )
            if int(token_dim) != int(feat_dim):
                raise ValueError(
                    "token_self_attention currently requires token_dim == emg_feat_dim "
                    "so the existing pose head can consume the pose token"
                )
            # The joint Transformer replaces the standalone EMG decoder and
            # legacy late-fusion projections.  Do not leave unused parameters
            # in DDP/optimizer state.
            self.decoder = nn.Identity()
            self.vision_proj = nn.Identity()
            self.fusion_proj = nn.Identity()

    def train(self, mode: bool = True) -> MidFusionPoseFormer:
        """Keep a protected pretrained vision branch in inference mode.

        Freezing parameters alone does not prevent BatchNorm running-stat updates
        or head dropout during training.  Re-applying eval here preserves the
        loaded vision-only predictor exactly while the residual branch learns.
        """
        super().train(mode)
        if self.freeze_vision_branch:
            self.vision_backbone.eval()
            self.head_vision.eval()
        else:
            if self.freeze_vision_head:
                self.head_vision.eval()
            if self.lock_vision_batch_norm:
                self._lock_vision_batch_norm_eval()
        return self

    def _lock_vision_batch_norm_eval(self) -> None:
        """Keep pretrained vision BatchNorm parameters and buffers unchanged."""
        for module in self.vision_backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

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
        emg_pooled = self.temporal_attn(decoded)  # (B, C)

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

        targets = F_interpolate(targets, size=T, mode="linear", align_corners=False)

        # Vision-only supervision is single-frame: only the center frame is
        # valid (the earlier align_mask result was discarded, so don't compute
        # it — this is the sole validity mask used downstream).
        center = T // 2
        mask = torch.zeros(
            mask.shape[0], T, device=mask.device, dtype=torch.float32
        )
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
            has_local_pretrained = bool(
                pretrained_path and Path(pretrained_path).is_file()
            )
            import os

            _offline = os.environ.get("EGOEMG_NO_PRETRAINED_DOWNLOAD", "") not in ("", "0")
            backbone = timm.create_model(
                timm_name,
                pretrained=not has_local_pretrained and not _offline,
                num_classes=0,
                img_size=256,
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
            has_local_pretrained = bool(
                pretrained_path and Path(pretrained_path).is_file()
            )
            weights = (
                None
                if has_local_pretrained
                else getattr(tv_models, weights_cls).IMAGENET1K_V1
            )
            rn = getattr(tv_models, resnet_name)(weights=weights)
            layers: list[nn.Module] = [
                rn.conv1, rn.bn1, rn.relu, rn.maxpool,
                rn.layer1, rn.layer2, rn.layer3, rn.layer4,
            ]
            if self.fusion_mode not in {"token_self_attention", "frozen_cross_attention"}:
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            backbone = nn.Sequential(*layers)
            if pretrained_path and Path(pretrained_path).exists():
                ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                backbonesd = {}
                headsd = {}
                for key, value in state_dict.items():
                    stripped = key[6:] if key.startswith("model.") else key
                    if stripped.startswith("backbone."):
                        backbonesd[stripped[len("backbone."):]] = value
                    elif stripped.startswith("vision_backbone."):
                        # Full fusion checkpoints retain a trained ResNet
                        # branch under ``vision_backbone``.  This makes that
                        # vision branch reusable by a new fusion architecture.
                        backbonesd[stripped[len("vision_backbone."):]] = value
                    elif stripped.startswith("head_vision."):
                        headsd[stripped[len("head_vision."):]] = value
                    elif stripped.startswith("head."):
                        # Store pre-stripped like the head_vision family so a
                        # single (already-stripped) key space is consumed below.
                        headsd[stripped[len("head."):]] = value
                if backbonesd:
                    missing, unexpected = backbone.load_state_dict(backbonesd, strict=False)
                    log.info(
                        "Loaded %s vision backbone from %s (%d missing, %d unexpected)",
                        resnet_name, pretrained_path, len(missing), len(unexpected),
                    )
                if headsd:
                    self._pretrained_head_state = headsd
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
            head_state = {}
            for key, value in state_dict.items():
                stripped = key[6:] if key.startswith("model.") else key
                if stripped.startswith("backbone."):
                    backbone_state[stripped[len("backbone."):]] = value
                elif stripped.startswith("head."):
                    head_state[stripped[len("head."):]] = value
            if backbone_state:
                missing, unexpected = backbone.load_state_dict(
                    backbone_state, strict=False
                )
                log.info(
                    "Loaded fusion vision backbone from %s (%d missing, %d unexpected)",
                    pretrained_path, len(missing), len(unexpected),
                )
            if head_state:
                self._pretrained_head_state = head_state
        return backbone

    def _extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        has_time = images.ndim == 5
        if has_time:
            bsz, num_frames, channels, height, width = images.shape
            images = images.view(bsz * num_frames, channels, height, width)
        if self.vision_backbone_type == "vit":
            # WiLoR's positional-embedding grid expects a 192x256 input.
            # Keep this identical to WiLoRViTPose._extract_feature: crop the
            # vertical dimension, not the horizontal dimension.
            images = images[:, :, 32:-32, :]
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

    def _extract_resnet_token_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return ResNet layer4 features as a ``(B, C, H, W)`` feature map."""
        if images.ndim == 5:
            batch_size, num_frames, channels, height, width = images.shape
            images = images.view(batch_size * num_frames, channels, height, width)
        else:
            batch_size, num_frames = images.shape[0], 1
        feature_map = self.vision_backbone(images)
        if feature_map.ndim != 4:
            raise ValueError(
                "token_self_attention expects a 4-D ResNet layer4 feature map"
            )
        if num_frames > 1:
            feature_map = feature_map.view(
                batch_size, num_frames, *feature_map.shape[1:]
            ).mean(dim=1)
        return feature_map

    def _extract_resnet_multiscale(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen ResNet layer3 and layer4 spatial feature maps."""
        if images.ndim == 5:
            batch_size, num_frames, channels, height, width = images.shape
            images = images.reshape(
                batch_size * num_frames, channels, height, width
            )
        else:
            batch_size, num_frames = images.shape[0], 1

        # The backbone is intentionally a Sequential of stem, layer1..layer4.
        vision_context = (
            torch.no_grad() if self.freeze_vision_branch else nullcontext()
        )
        with vision_context:
            features = images
            for layer in self.vision_backbone[:6]:
                features = layer(features)
            layer3_map = self.vision_backbone[6](features)
            layer4_map = self.vision_backbone[7](layer3_map)
        if num_frames > 1:
            layer3_map = layer3_map.reshape(
                batch_size, num_frames, *layer3_map.shape[1:]
            ).mean(dim=1)
            layer4_map = layer4_map.reshape(
                batch_size, num_frames, *layer4_map.shape[1:]
            ).mean(dim=1)
        return layer3_map, layer4_map

    def _extract_vit_spatial_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact pooled ViT feature and its spatial patch map.

        DINOv2/timm exposes the final token sequence through
        ``forward_features`` and applies its original pooling through
        ``forward_head``.  WiLoR already returns a final 4-D image feature map.
        Keeping the pooled path unchanged preserves the vision-only pose while
        the patch map supplies key/value tokens to the EMG cross-attention.
        """
        if images.ndim == 5:
            batch_size, num_frames, channels, height, width = images.shape
            images = images.reshape(
                batch_size * num_frames, channels, height, width
            )
        else:
            batch_size, num_frames = images.shape[0], 1

        vision_context = (
            torch.no_grad() if self.freeze_vision_branch else nullcontext()
        )
        with vision_context:
            if self.vision_backbone_type in _DINOV2_VARIANTS:
                token_sequence = self.vision_backbone.forward_features(images)
                vision_features = self.vision_backbone.forward_head(
                    token_sequence, pre_logits=True
                )
                if not isinstance(token_sequence, torch.Tensor):
                    raise ValueError(
                        "DINOv2 forward_features must return a token tensor"
                    )
                num_prefix = int(
                    getattr(self.vision_backbone, "num_prefix_tokens", 1)
                )
                patch_tokens = token_sequence[:, num_prefix:]
                grid_height, grid_width = self.vision_backbone.patch_embed.grid_size
                if patch_tokens.shape[1] != grid_height * grid_width:
                    raise ValueError(
                        "DINOv2 patch-token count does not match patch grid: "
                        f"{patch_tokens.shape[1]} vs {grid_height}x{grid_width}"
                    )
                spatial_map = patch_tokens.transpose(1, 2).reshape(
                    patch_tokens.shape[0], patch_tokens.shape[2],
                    grid_height, grid_width,
                )
            elif self.vision_backbone_type == "vit":
                # WiLoR was trained on the centered 192-pixel-wide crop.
                output = self.vision_backbone(images[:, :, :, 32:-32])
                spatial_map = None
                if isinstance(output, tuple):
                    for item in reversed(output):
                        if isinstance(item, torch.Tensor) and item.ndim == 4:
                            spatial_map = item
                            break
                elif isinstance(output, torch.Tensor) and output.ndim == 4:
                    spatial_map = output
                if spatial_map is None:
                    raise ValueError("WiLoR ViT returned no spatial feature map")
                vision_features = spatial_map.mean(dim=(-2, -1))
            else:
                raise ValueError(
                    "ViT spatial extraction requires a DINOv2 or WiLoR backbone"
                )

        if num_frames > 1:
            vision_features = vision_features.reshape(
                batch_size, num_frames, -1
            ).mean(dim=1)
            spatial_map = spatial_map.reshape(
                batch_size, num_frames, *spatial_map.shape[1:]
            ).mean(dim=1)
        return vision_features, spatial_map

    @staticmethod
    def _center_targets(
        batch: dict[str, torch.Tensor],
        preds: torch.Tensor,
        left_context: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "joint_angles" not in batch or "label_valid_mask" not in batch:
            return preds
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]
        if joint_angles.shape[-1] == 1:
            mask_out = mask[..., :1] if mask.ndim >= 2 else mask.unsqueeze(-1)
            return preds, joint_angles, mask_out

        half_context = left_context // 2
        right_stop = -half_context if half_context > 0 else None
        targets_full = joint_angles[..., half_context:right_stop]
        mask_full = mask[..., half_context:right_stop]
        center = targets_full.shape[-1] // 2
        targets = targets_full[:, :, center:center + 1]
        if mask_full.ndim >= 3:
            mask_out = mask_full[:, :, center:center + 1]
        else:
            mask_out = mask_full[..., center:center + 1]
        return preds, targets, mask_out

    def _forward_frozen_cross_attention(
        self,
        batch: dict[str, torch.Tensor],
        layer3_map: torch.Tensor,
        layer4_map: torch.Tensor,
        emg: torch.Tensor,
        vision_features: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict a center-frame residual from early multi-scale fusion."""
        vision_valid = batch.get("vision_valid_mask")
        if vision_valid is not None and vision_valid.ndim > 1:
            vision_valid = vision_valid.any(dim=1)

        # Both modules are frozen and held in eval mode, so the baseline branch
        # is computed without retaining an autograd graph.
        vision_context = (
            torch.no_grad() if self.freeze_vision_branch else nullcontext()
        )
        with vision_context:
            if vision_features is None:
                vision_features = layer4_map.mean(dim=(-2, -1))
            if vision_valid is not None:
                vision_features = vision_features * vision_valid[:, None].to(
                    vision_features.dtype
                )
            vision_pose = self.head_vision(vision_features)

        emg_features = self.featurizer(emg)
        fused_features = self.early_fusion(
            emg_features, layer3_map, layer4_map, vision_valid
        )
        decoded = self.decoder(fused_features)
        center = decoded.shape[-1] // 2
        delta = self.head(decoded[..., center:center + 1])
        scaled_delta = self.residual_scale * delta
        self._last_delta = scaled_delta
        preds = vision_pose.unsqueeze(-1) + scaled_delta
        return self._center_targets(batch, preds, self.left_context)

    def compute_anchor_emg_delta(
        self,
        batch: dict[str, torch.Tensor],
        shuffle_fraction: float = 0.0,
    ) -> torch.Tensor | None:
        """Center residual for a zero/mismatched EMG anchor batch.

        A fraction of samples receive EMG from another sample with the same
        target hand; the remainder receive zero EMG.  Penalizing this residual
        prevents the model from using arbitrary nonzero EMG as a gate for a
        vision-only correction while preserving hand-specific channel
        statistics in the hard negatives.
        """
        if self.fusion_mode != "frozen_cross_attention":
            return None
        if "vision_img" not in batch:
            return None
        emg = batch["emg"]
        anchor_emg = make_invalid_anchor_emg(
            emg, batch.get("target_hand_index"), shuffle_fraction
        )

        if self.vision_backbone_type in _RESNET_DIMS:
            layer3_map, layer4_map = self._extract_resnet_multiscale(
                batch["vision_img"]
            )
        else:
            _, spatial_map = self._extract_vit_spatial_features(
                batch["vision_img"]
            )
            # ViTs expose one final spatial grid through their stable public
            # interface.  Feed that same grid through the two learned visual
            # projections so the downstream cross-attention block remains
            # architecturally identical to the ResNet path.
            layer3_map = spatial_map
            layer4_map = spatial_map
        emg_features = self.featurizer(anchor_emg)
        vision_valid = batch.get("vision_valid_mask")
        if vision_valid is not None and vision_valid.ndim > 1:
            vision_valid = vision_valid.any(dim=1)
        fused_features = self.early_fusion(
            emg_features, layer3_map, layer4_map, vision_valid
        )
        decoded = self.decoder(fused_features)
        center = decoded.shape[-1] // 2
        return self.head(decoded[..., center:center + 1])

    def compute_zero_emg_delta(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Backward-compatible pure-zero anchor helper."""
        return self.compute_anchor_emg_delta(batch, shuffle_fraction=0.0)

    def _forward_token_self_attention(
        self,
        batch: dict[str, torch.Tensor],
        vision_map: torch.Tensor,
        emg: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the center pose from jointly attended EMG and CNN tokens."""
        vision_valid = batch.get("vision_valid_mask")
        if vision_valid is not None and vision_valid.ndim > 1:
            vision_valid = vision_valid.any(dim=1)

        emg_features = self.featurizer(emg)
        pose_feature = self.token_fusion(emg_features, vision_map, vision_valid)
        preds = self.head(pose_feature.unsqueeze(-1))
        self._last_delta = None

        if "joint_angles" not in batch or "label_valid_mask" not in batch:
            return preds
        joint_angles = batch["joint_angles"]
        mask = batch["label_valid_mask"]
        if joint_angles.shape[-1] == 1:
            mask_out = mask[..., :1] if mask.ndim >= 2 else mask.unsqueeze(-1)
            return preds, joint_angles, mask_out

        half_context = self.left_context // 2
        right_stop = -half_context if half_context > 0 else None
        targets_full = joint_angles[..., half_context:right_stop]
        mask_full = mask[..., half_context:right_stop]
        center = targets_full.shape[-1] // 2
        targets = targets_full[:, :, center:center + 1]
        if mask_full.ndim >= 3:
            mask_out = mask_full[:, :, center:center + 1]
        else:
            mask_out = mask_full[..., center:center + 1]
        return preds, targets, mask_out

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "frozen_cross_attention":
            if "vision_img" not in batch:
                raise ValueError(
                    "frozen_cross_attention requires batch['vision_img']"
                )
            emg = batch["emg"]
            if self.force_zero_emg:
                emg = torch.zeros_like(emg)
            vision_features = None
            if self.vision_backbone_type in _RESNET_DIMS:
                layer3_map, layer4_map = self._extract_resnet_multiscale(
                    batch["vision_img"]
                )
            else:
                vision_features, spatial_map = self._extract_vit_spatial_features(
                    batch["vision_img"]
                )
                layer3_map = spatial_map
                layer4_map = spatial_map
            return self._forward_frozen_cross_attention(
                batch, layer3_map, layer4_map, emg, vision_features
            )

        if self.fusion_mode == "token_self_attention":
            if "vision_img" not in batch:
                raise ValueError("token_self_attention requires batch['vision_img']")
            emg = batch["emg"]
            if self.force_zero_emg:
                emg = torch.zeros_like(emg)
            return self._forward_token_self_attention(
                batch, self._extract_resnet_token_map(batch["vision_img"]), emg
            )

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
