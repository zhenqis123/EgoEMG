"""July-style temporal-attention residual on current paired data."""

from __future__ import annotations

import torch
from torch import nn

from egoemg.models.heads.mlp import MLPHead
from egoemg.models.modules._pooling import TemporalAttentionPool
from egoemg.models.modules.bimanual_fusion import BimanualResidualFusion


class BimanualSimpleResidualFusion(BimanualResidualFusion):
    """Train EMG + temporal pool + projections, retaining trained vision pose.

    EMG decoder tokens -> learned time pooling (256 ->64 ->1 scores).
    Visual global feature -> Linear256. Concatenate the two per-hand vectors
    and hand ID -> Linear512/GELU/Dropout/Linear256 -> MLP512 ->22 correction.
    The original EMG pose head is frozen/unused. Missing EMG is exact vision.
    """

    optimizer_component_prefixes = {
        "vision": ("vision.",),
        "emg": ("emg_model.featurizer.", "emg_model.decoder."),
        "fusion": ("temporal_attn.", "vision_proj.", "fusion_proj.", "head."),
    }

    def __init__(
        self,
        vision_checkpoint,
        emg_checkpoint,
        fusion_dim=256,
        fusion_hidden_dim=512,
        head_hidden_dim=512,
        dropout=0.1,
        residual_init_std=1e-5,
        freeze_emg=False,
        freeze_vision=True,
        emg_config=None,
        independent_hand_heads=False,
        require_causal_inputs=False,
    ):
        if residual_init_std < 0:
            raise ValueError("residual_init_std must be nonnegative")
        super().__init__(
            vision_checkpoint,
            emg_checkpoint,
            freeze_emg=freeze_emg,
            freeze_vision=freeze_vision,
            emg_config=emg_config,
        )
        self.require_causal_inputs = bool(require_causal_inputs)
        if self.require_causal_inputs:
            if not getattr(self.vision, 'causal', False):
                raise ValueError('Causal fusion requires causal temporal vision')
            if not getattr(self.emg_model, 'model_causal', False):
                raise ValueError('Causal fusion requires a model-causal EMG checkpoint')
        vision_dim = self.vision.head[0].in_features
        self.temporal_attn = TemporalAttentionPool(self.emg_dim)
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.emg_dim + fusion_dim + 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, self.emg_dim),
        )
        self.independent_hand_heads = independent_hand_heads
        heads = [MLPHead(self.emg_dim, 22, hidden_sizes=[head_hidden_dim],
                         dropout=dropout)
                 for _ in range(2 if independent_hand_heads else 1)]
        for head in heads:
            nn.init.normal_(head.net[-1].weight, std=residual_init_std)
            nn.init.zeros_(head.net[-1].bias)
        self.head = nn.ModuleList(heads) if independent_hand_heads else heads[0]

    def _pool_emg(self, features):
        return self.temporal_attn(features)

    def _predict_delta(self, visual, pooled, hand_id):
        projected = self.vision_proj(visual).repeat_interleave(2, dim=0)
        fused = self.fusion_proj(torch.cat((pooled, projected, hand_id), dim=-1))
        if self.independent_hand_heads:
            fused = fused.reshape(len(visual), 2, self.emg_dim)
            # The flattened output order remains image0-left, image0-right, ...
            corrections = [head(fused[:, side].unsqueeze(-1))
                           for side, head in enumerate(self.head)]
            return torch.stack(corrections, dim=1).flatten(0, 1)
        return self.head(fused.unsqueeze(-1))
