import collections
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


class SequentialLSTM(nn.Module):
    """
    LSTM where each forward() call computes only a single time step, to be compatible
    looping over time manually.

    NOTE: Need to manually reset the state in outer context after each trajectory!
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers, batch_first=True)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.mlp_out = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(hidden_size, out_channels)
        )
        self.scale = scale

    def reset_state(self):
        self.hidden = None

    def forward(self, x):
        """Forward pass for a single time step, where x is (batch, channel.)"""

        if self.hidden is None:
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            device = x.device
            size = (self.num_layers, batch_size, self.hidden_size)
            self.hidden = (torch.zeros(*size).to(device), torch.zeros(*size).to(device))

        out, self.hidden = self.lstm(x[:, None], self.hidden)
        return self.mlp_out(out[:, 0]) * self.scale

    def _non_sequential_forward(self, x):
        """Non-sequential forward pass, where x is (batch, time, channel)."""
        return self.mlp_out(self.lstm(x)[0]) * self.scale
