"""Runtime loader for compact MANO-theta to UmeTrack-angle mapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ManoToUmeTrackMapper(nn.Module):
    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(45, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RuntimeManoToUmeTrackMapper:
    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> None:
        payload = torch.load(checkpoint_path, map_location=device)
        hidden_dim = int(payload["hidden_dim"])
        self.device = torch.device(device)
        self.model = ManoToUmeTrackMapper(hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.x_mean = payload["x_mean"].to(self.device).float()
        self.x_std = payload["x_std"].to(self.device).float()
        self.y_mean = payload["y_mean"].to(self.device).float()
        self.y_std = payload["y_std"].to(self.device).float()

    @torch.no_grad()
    def predict(self, mano_theta: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map MANO hand pose to 20D UmeTrack finger angles.

        Args:
            mano_theta: Either shape (45,), (B, 45), full MANO pose (48,), or
                (B, 48). Full poses are sliced as pose[..., 3:48].
        """
        is_numpy = isinstance(mano_theta, np.ndarray)
        x = torch.from_numpy(mano_theta).float() if is_numpy else mano_theta.float()
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.shape[-1] == 48:
            x = x[..., 3:48]
        if x.shape[-1] != 45:
            raise ValueError(f"Expected MANO theta dim 45 or 48, got {tuple(x.shape)}")
        x = x.to(self.device)
        y_norm = self.model((x - self.x_mean) / self.x_std)
        y = y_norm * self.y_std + self.y_mean
        if squeeze:
            y = y.squeeze(0)
        return y.detach().cpu().numpy()
