from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from emg2pose.geometry import (
    axis_angle_to_angles,
    get_joint_rotation_axes,
    rot6d_to_axis_angle,
)
from emg2pose.models.quantizers import ResidualVectorQuantizer


class RVQHead(nn.Module):
    """Residual VQ head producing quantized angle predictions."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_joints: int = 20,
        hidden_dims: Sequence[int] = (256, 256),
        angle_representation: Literal["angle", "axis_angle", "rot6d"] = "angle",
        num_levels: int = 2,
        num_codes: int = 256,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.angle_representation = angle_representation
        self.num_joints = num_joints
        self.repr_dim = self._get_repr_dim()

        self.proj = nn.Linear(in_channels, embed_dim)
        self.quantizer = ResidualVectorQuantizer(
            num_levels=num_levels,
            num_codes=num_codes,
            embed_dim=embed_dim,
            commitment_cost=commitment_cost,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
        )

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.repr_dim))
        self.decoder = nn.Sequential(*layers)

        if self.angle_representation != "angle":
            axes = get_joint_rotation_axes(self.num_joints)
            self.register_buffer("joint_axes", axes, persistent=False)

    def _get_repr_dim(self) -> int:
        if self.angle_representation == "angle":
            return self.num_joints
        if self.angle_representation == "axis_angle":
            return self.num_joints * 3
        if self.angle_representation == "rot6d":
            return self.num_joints * 6
        raise ValueError(f"Unknown representation {self.angle_representation}")

    def _repr_to_angles(self, repr_tensor: torch.Tensor) -> torch.Tensor:
        if self.angle_representation == "angle":
            return repr_tensor
        axes = self.joint_axes.to(dtype=repr_tensor.dtype, device=repr_tensor.device)
        if self.angle_representation == "axis_angle":
            axis_angle = repr_tensor.view(-1, self.num_joints, 3)
        else:  # rot6d
            rot6d = repr_tensor.view(-1, self.num_joints, 6)
            axis_angle = rot6d_to_axis_angle(rot6d.view(-1, 6)).view(-1, self.num_joints, 3)
        return axis_angle_to_angles(axis_angle, axes)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, C, T)
        b, _, t = x.shape
        x_flat = x.transpose(1, 2).reshape(b * t, -1)
        proj = self.proj(x_flat)

        quantized, indices, vq_loss, codebook_loss, commit_loss = self.quantizer(proj)

        decoded_repr = self.decoder(quantized)
        angles = self._repr_to_angles(decoded_repr)
        angles = angles.view(b, t, self.num_joints).transpose(1, 2)  # (B, J, T)

        perplexity = ResidualVectorQuantizer.compute_perplexity(
            indices, num_codes=self.quantizer.num_codes
        )

        aux = {
            "indices_pred": indices,  # (L, N)
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
        }

        return angles, aux
