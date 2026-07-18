from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GumbelVectorQuantizer(nn.Module):
    """Gumbel-Softmax vector quantizer for wav2vec 2.0 style training.

    Splits input_dim into ``num_groups`` groups, each selecting from
    ``num_codes`` codebook entries via Gumbel-Softmax (differentiable).

    Supports linear temperature annealing (wav2vec 2.0 paper) or exponential
    decay. Linear is default: temp decreases linearly from max to min over
    ``temp_anneal_steps`` training steps.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_groups: int = 2,
        num_codes: int = 320,
        temperature: tuple[float, float, float] = (2.0, 0.5, 0.999995),
        temp_anneal_steps: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_codes = num_codes

        # Temperature annealing params
        self.max_temp, self.min_temp, self.temp_decay = temperature
        self.curr_temp = self.max_temp
        # If temp_anneal_steps > 0, use linear annealing; else exponential
        self.temp_anneal_steps = temp_anneal_steps
        self._temp_step = 0

        self.code_dim = input_dim // num_groups
        assert input_dim % num_groups == 0, (
            f"input_dim ({input_dim}) must be divisible by "
            f"num_groups ({num_groups})"
        )

        # Project input to logits over codebook entries per group
        self.proj = nn.Linear(input_dim, num_groups * num_codes)

        # Codebook: (G * V, code_dim)
        self.codebook = nn.Parameter(
            torch.empty(num_groups * num_codes, self.code_dim)
        )
        nn.init.uniform_(self.codebook, -0.5, 0.5)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, T) latent features from featurizer.
        Returns:
            quantized: (B, D, T) quantized representation.
            diversity_loss: scalar, encourages uniform codebook usage.
            perplexity: scalar, effective number of codes used.
        """
        B, D, T = z.shape
        # (B, D, T) -> (B*T, D)
        x = z.permute(0, 2, 1).reshape(B * T, D)

        # Project to logits: (B*T, G * V)
        logits = self.proj(x)
        logits = logits.view(B * T, self.num_groups, self.num_codes)

        if self.training:
            probs = F.gumbel_softmax(
                logits.float(), tau=self.curr_temp, hard=True
            ).type_as(z)
        else:
            # Argmax at eval
            indices = logits.argmax(dim=-1)
            probs = F.one_hot(indices, self.num_codes).float().type_as(z)

        # Compute diversity loss per group: encourage uniform usage
        # Average probs across batch: (G, V)
        avg_probs = torch.softmax(logits.float(), dim=-1).mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()
        # Normalize by total capacity (num_groups * num_codes) so loss ∈ [0, 1]
        total_codes = self.num_groups * self.num_codes
        diversity_loss = (total_codes - perplexity) / total_codes

        # Look up codebook: (B*T, G, V) x (G*V, code_dim) -> (B*T, G, code_dim)
        cb = self.codebook.view(self.num_groups, self.num_codes, self.code_dim)
        quantized = torch.einsum("bgv,gvd->bgd", probs, cb)

        # Concat groups: (B*T, G, code_dim) -> (B*T, D)
        quantized = quantized.reshape(B * T, D)

        # Reshape back to (B, D, T)
        quantized = quantized.view(B, T, D).permute(0, 2, 1)

        return quantized, diversity_loss, perplexity

    def update_temperature(self) -> None:
        """Anneal temperature by one step.

        Linear annealing if temp_anneal_steps > 0, else exponential decay.
        """
        if self.temp_anneal_steps > 0:
            # Linear: from max_temp to min_temp over temp_anneal_steps
            self._temp_step += 1
            frac = min(1.0, self._temp_step / self.temp_anneal_steps)
            self.curr_temp = self.max_temp - frac * (
                self.max_temp - self.min_temp
            )
        else:
            # Exponential decay (legacy)
            self.curr_temp = max(
                self.min_temp, self.curr_temp * self.temp_decay
            )
