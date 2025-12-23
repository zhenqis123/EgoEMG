from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from emg2pose.models.quantizers.vq import VectorQuantizer


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float):
        super().__init__()
        mods: list[nn.Module] = [
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            mods.append(nn.Dropout(dropout_p))
        mods.append(nn.Linear(dim, dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        embed_dim: int,
        dropout: float = 0.0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        if use_residual and len(hidden_dims) > 0:
            if len(set(hidden_dims)) != 1:
                raise ValueError("use_residual=True requires all hidden_dims to be equal.")
            hidden_dim = hidden_dims[0]
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            for _ in range(len(hidden_dims) - 1):
                layers.append(ResidualBlock(hidden_dim, dropout))
            layers.append(nn.Linear(hidden_dim, embed_dim))
        else:
            for hidden in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, embed_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = embed_dim
        if use_residual and len(hidden_dims) > 0:
            if len(set(hidden_dims)) != 1:
                raise ValueError("use_residual=True requires all hidden_dims to be equal.")
            hidden_dim = hidden_dims[0]
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            for _ in range(len(hidden_dims) - 1):
                layers.append(ResidualBlock(hidden_dim, dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            for hidden in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class JointAngleVQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Sequence[int] = (128, 128),
        embed_dim: int = 64,
        num_codes: int = 512,
        commitment_cost: float = 0.25,
        dropout: float = 0.0,
        output_dim: int | None = None,
        quantizer: nn.Module | None = None,
        encoder_residual: bool = False,
        decoder_residual: bool = False,
    ) -> None:
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout,
            use_residual=encoder_residual,
        )
        self.quantizer = (
            quantizer
            if quantizer is not None
            else VectorQuantizer(
                num_codes=num_codes, embed_dim=embed_dim, commitment_cost=commitment_cost
            )
        )
        self.decoder = MLPDecoder(
            embed_dim=embed_dim,
            hidden_dims=tuple(hidden_dims)[::-1],
            output_dim=output_dim,
            dropout=dropout,
            use_residual=decoder_residual,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, indices, vq_loss, codebook_loss, commit_loss = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss, codebook_loss, commit_loss



