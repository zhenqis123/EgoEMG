from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn.functional import interpolate


class EmgformerPretrain(nn.Module):
    """Shared TDS + Transformer backbone with multi-task pretraining heads."""

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        recon_head: nn.Module,
        angle_head: nn.Module,
        gesture_head: nn.Module,
        keystroke_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.decoder = decoder
        self.recon_head = recon_head
        self.angle_head = angle_head
        self.gesture_head = gesture_head
        self.keystroke_head = keystroke_head

        self.left_context = getattr(featurizer, "left_context", 0)
        self.right_context = getattr(featurizer, "right_context", 0)

    def align_predictions(self, pred: torch.Tensor, n_time: int) -> torch.Tensor:
        """Temporally resamples predictions to match the length of targets."""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int) -> torch.Tensor:
        """Temporally resample mask to match the length of targets."""
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            emg = batch["emg"]
        except KeyError:
            raise KeyError("batch must contain 'emg'") from None
        features = self.featurizer(emg)  # (B, C, T)
        decoded = self.decoder(features)
        outputs = {}
        if self.recon_head is not None:
            outputs["recon"] = self.recon_head(decoded)
        if self.angle_head is not None:
            outputs["angles"] = self.angle_head(decoded)
        if self.gesture_head is not None:
            outputs["gesture_logits"] = self.gesture_head(decoded)
        if self.keystroke_head is not None:
            # Output shape: (T, B, C) for CTC loss
            keystroke_logits = self.keystroke_head(decoded)  # (B, C, T)
            keystroke_logits = keystroke_logits.permute(2, 0, 1)  # (T, B, C)
            outputs["keystroke_logits"] = keystroke_logits
        return outputs
