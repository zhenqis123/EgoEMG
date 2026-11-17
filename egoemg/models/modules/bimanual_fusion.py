"""Protected ViT-S + per-hand EMGFormer residual fusion."""

from __future__ import annotations

from egoemg.checkpoints import load_branch as load_branch

from contextlib import nullcontext

import torch
from hydra.utils import instantiate
from torch import nn


class BimanualResidualFusion(nn.Module):
    bimanual_output = True

    def __init__(
        self,
        vision_checkpoint,
        emg_checkpoint,
        hidden_dim=256,
        dropout=0.2,
        freeze_emg=True,
        freeze_vision=True,
        emg_config=None,
    ):
        super().__init__()
        self.freeze_emg = bool(freeze_emg)
        self.freeze_vision = bool(freeze_vision)
        self.vision = load_branch(vision_checkpoint)
        if emg_checkpoint is not None:
            if emg_config is not None:
                raise ValueError("Specify an EMG checkpoint or random config, not both")
            self.emg_model = load_branch(emg_checkpoint)
        elif emg_config is not None:
            self.emg_model = (
                emg_config
                if isinstance(emg_config, nn.Module)
                else instantiate(emg_config, _convert_="all")
            )
        else:
            raise ValueError("Random EMG initialization requires emg_config")
        if not getattr(self.vision, "bimanual_output", False):
            raise ValueError("Require a bimanual vision checkpoint")
        vision_dim = self.vision.head[0].in_features
        # Output feature width is the EMG decoder's output projection width.
        emg_dim = self.emg_model.decoder.output_proj.out_features
        self.emg_dim = emg_dim
        self.head = nn.Sequential(
            nn.LayerNorm(vision_dim + emg_dim + 2),
            nn.Linear(vision_dim + emg_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 22),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.vision.requires_grad_(not self.freeze_vision)
        self.emg_model.requires_grad_(not self.freeze_emg)
        # The original EMG pose head is retained for restoration/diagnostics;
        # fusion consumes decoder features, so that unused head stays frozen.
        if hasattr(self.emg_model, "head"):
            self.emg_model.head.requires_grad_(False)
        self.train(self.training)
        self.left_context = self.right_context = 0
        self._last_delta = None
        self._last_delta_l1 = None
        self.force_zero_correction = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision:
            self.vision.eval()
        if self.freeze_emg:
            self.emg_model.eval()
        if hasattr(self.emg_model, "head"):
            self.emg_model.head.eval()
        return self

    def _pool_emg(self, features):
        return features.mean(-1)

    def _predict_delta(self, visual, pooled, hand_id):
        fused = torch.cat((visual.repeat_interleave(2, 0), pooled, hand_id), dim=-1)
        return self.head(fused).unsqueeze(-1)

    def _validate_causal_batch(self, batch):
        if not getattr(self, "require_causal_inputs", False):
            return
        anchor = batch["anchor_timestamp_us"].reshape(-1)
        times = batch["temporal_timestamps_us"]
        if (times > anchor[:, None]).any() or not torch.equal(times[:, -1], anchor):
            raise ValueError("Future/misaligned visual frame in causal fusion")
        available = batch["emg_valid_mask"].bool()
        if ((batch["emg_last_timestamp_us"] > anchor[:, None]) & available).any():
            raise ValueError("Future EMG in causal fusion")

    def forward(self, batch):
        self._validate_causal_batch(batch)
        image = batch["vision_img"]
        batch_size = image.shape[0]
        with torch.no_grad() if self.freeze_vision else nullcontext():
            if hasattr(self.vision, "forward_features"):
                visual = self.vision.forward_features(batch)
            else:
                if image.dtype == torch.uint8:
                    image = (
                        image.float()
                        .div(255)
                        .sub(self.vision.image_mean)
                        .div(self.vision.image_std)
                    )
                visual = self.vision.backbone(image)
            baseline = self.vision.head(visual).reshape(-1, 22, 1)
        self._last_vision_prediction = baseline.detach()
        signal = batch["emg"].reshape(-1, 8, batch["emg"].shape[-1])
        available = batch["emg_valid_mask"].reshape(-1).bool()
        # Missing windows must not influence a trainable EMG branch's
        # BatchNorm statistics through fabricated zero/noise inputs.
        indices = available.nonzero(as_tuple=True)[0]
        pooled = visual.new_zeros((signal.shape[0], self.emg_dim))
        if indices.numel():
            with torch.no_grad() if self.freeze_emg else nullcontext():
                features = self.emg_model.decoder(
                    self.emg_model.featurizer(signal[indices])
                )
            # Pooling may contain new trainable layers even with frozen experts.
            pooled = pooled.index_copy(
                0, indices, self._pool_emg(features).to(pooled.dtype)
            )
        hand_id = torch.eye(2, device=visual.device, dtype=visual.dtype).repeat(
            batch_size, 1
        )
        valid_emg = batch["emg_valid_mask"].reshape(-1, 1, 1)
        delta = self._predict_delta(visual, pooled, hand_id)
        delta = torch.where(valid_emg, delta, 0.0)
        if self.force_zero_correction:
            delta = delta * 0
        self._last_delta = delta
        if "joint_angles" not in batch:
            return baseline + delta
        targets = batch["joint_angles"].reshape(-1, 22, 1)
        mask = batch["label_valid_mask"].reshape(-1, 22, 1).bool()
        active = mask & valid_emg
        self._last_delta_l1 = delta.masked_fill(
            ~active, 0
        ).abs().sum() / active.sum().clamp_min(1)
        return baseline + delta, targets, mask

    @staticmethod
    def align_mask(mask, n_time):
        return mask
