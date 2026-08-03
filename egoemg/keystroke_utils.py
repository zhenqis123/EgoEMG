# Copyright (c) Meta Platforms, Inc. and affiliates.
# Simplified CTC utilities for keystroke supervision in pretraining

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import Levenshtein
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric


@dataclass
class KeystrokeData:
    """Simple container for keystroke label sequences."""

    labels: list[int]
    text: str = ""

    @classmethod
    def from_labels(cls, labels: list[int] | np.ndarray, charset: Any) -> KeystrokeData:
        """Create KeystrokeData from label indices."""
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        # Convert labels to text using charset
        text = ""
        for label in labels:
            try:
                key = charset.label_to_key(label)
                text += chr(charset.key_to_unicode(key))
            except (IndexError, KeyError, ValueError):
                pass  # Skip invalid labels

        return cls(labels=labels, text=text)

    def __len__(self) -> int:
        return len(self.text)


class CTCGreedyDecoder:
    """Simple CTC greedy decoder for keystroke sequences."""

    def __init__(self, charset: Any, blank_idx: int = 0):
        self.charset = charset
        self.blank_idx = blank_idx

    def decode_batch(
        self,
        emissions: np.ndarray,
        emission_lengths: np.ndarray,
    ) -> list[KeystrokeData]:
        """Decode a batch of emission logits.

        Args:
            emissions: (T, N, num_classes) emission log probabilities
            emission_lengths: (N,) valid lengths for each sequence

        Returns:
            List of KeystrokeData, one per batch item
        """
        assert emissions.ndim == 3  # (T, N, num_classes)
        N = emissions.shape[1]

        decodings = []
        for i in range(N):
            # Get emissions for this batch item
            emis = emissions[: emission_lengths[i], i]  # (T, num_classes)

            # Greedy decode: argmax at each timestep
            labels = emis.argmax(axis=-1)  # (T,)

            # CTC collapse: remove blanks and repeated labels
            decoded = []
            prev_label = self.blank_idx
            for label in labels:
                if label != self.blank_idx and label != prev_label:
                    decoded.append(int(label))
                prev_label = label

            decodings.append(
                KeystrokeData.from_labels(decoded, self.charset)
            )

        return decodings


class KeystrokeErrorRates(Metric):
    """Character-level error rates based on Levenshtein distance.

    Computes:
    - CER: Character Error Rate
    - IER: Insertion Error Rate
    - DER: Deletion Error Rate
    - SER: Substitution Error Rate
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.add_state("insertions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("deletions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("substitutions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_len", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: KeystrokeData, target: KeystrokeData) -> None:
        """Update metrics with a prediction-target pair."""
        # Use Levenshtein.editops to break down errors
        editops = Levenshtein.editops(prediction.text, target.text)
        edits = Counter(op for op, _, _ in editops)

        # Update running counts
        self.insertions += edits["insert"]
        self.deletions += edits["delete"]
        self.substitutions += edits["replace"]
        self.target_len += len(target)

    def compute(self) -> dict[str, float]:
        """Compute error rates as percentages."""
        if self.target_len == 0:
            return {
                "CER": 0.0,
                "IER": 0.0,
                "DER": 0.0,
                "SER": 0.0,
            }

        def _error_rate(errors: torch.Tensor) -> float:
            return float(errors.item() / self.target_len.item() * 100.0)

        return {
            "CER": _error_rate(self.insertions + self.deletions + self.substitutions),
            "IER": _error_rate(self.insertions),
            "DER": _error_rate(self.deletions),
            "SER": _error_rate(self.substitutions),
        }


def compute_ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
) -> torch.Tensor:
    """Compute CTC loss for keystroke sequences.

    Args:
        logits: (T, N, num_classes) or (N, T, num_classes) model outputs
        targets: (N, S) target label sequences (padded)
        input_lengths: (N,) valid lengths of input sequences
        target_lengths: (N,) valid lengths of target sequences
        blank_idx: index of blank label

    Returns:
        Scalar CTC loss
    """
    # Ensure logits are (T, N, num_classes)
    if logits.dim() == 3 and logits.shape[0] < logits.shape[1]:
        logits = logits.transpose(0, 1)

    # Apply log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # Flatten targets to 1D for CTCLoss
    # targets should be (N, S) -> concatenate to 1D
    targets_flat = []
    for i in range(targets.shape[0]):
        targets_flat.append(targets[i, :target_lengths[i]])
    targets_concat = torch.cat(targets_flat)

    # Compute CTC loss
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    loss = ctc_loss(
        log_probs=log_probs,
        targets=targets_concat,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )

    return loss
