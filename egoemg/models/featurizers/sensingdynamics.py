"""SensingDynamics feature extractor adapted to ring-electrode sEMG."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SMU(nn.Module):
    """Learnable Smooth Maximum Unit activation.

    The implementation follows Biswas et al. (CVPR 2022):

    ``((1 + alpha) * x + (1 - alpha) * x * erf(mu*(1-alpha)*x)) / 2``.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        mu: float = 25.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha)), requires_grad=learnable
        )
        self.mu = nn.Parameter(torch.tensor(float(mu)), requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp(0.0, 1.0)
        mu = self.mu.clamp_min(0.01)
        linear = (1.0 + alpha) * x
        smooth = (1.0 - alpha) * x * torch.erf(mu * (1.0 - alpha) * x)
        return 0.5 * (linear + smooth)


def butterworth_lowpass(
    x: torch.Tensor,
    sample_rate: float,
    cutoff_hz: float,
    order: int,
) -> torch.Tensor:
    """Apply a zero-phase Butterworth-magnitude low-pass along time."""
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate / 2.0:
        raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")
    if order <= 0:
        raise ValueError("order must be positive")

    spectrum = torch.fft.rfft(x.float(), dim=-1)
    frequencies = torch.fft.rfftfreq(
        x.shape[-1], d=1.0 / sample_rate, device=x.device
    )
    response = torch.rsqrt(
        1.0 + (frequencies / cutoff_hz).pow(2 * order)
    )
    filtered = torch.fft.irfft(spectrum * response, n=x.shape[-1], dim=-1)
    return filtered.to(dtype=x.dtype)


class CircularPad2d(nn.Module):
    """Circularly pad only the ring-electrode dimension."""

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, self.padding, self.padding), mode="circular")


class SensingDynamicsConvBlock(nn.Module):
    """Conv-BN-SMU-dropout block over electrode position and time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        dropout_rate: float = 0.25,
        circular_pad: bool = True,
    ) -> None:
        super().__init__()
        kernel_electrode, kernel_time = kernel_size
        dilation_electrode, dilation_time = dilation
        electrode_padding = dilation_electrode * (kernel_electrode - 1) // 2
        time_padding = dilation_time * (kernel_time - 1) // 2

        self.circular_pad = (
            CircularPad2d(electrode_padding)
            if circular_pad
            else nn.Identity()
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, time_padding),
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = SMU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.circular_pad(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.dropout(x)


class SensingDynamicsFeaturizer(nn.Module):
    """2D SensingDynamics encoder for an eight-electrode wristband.

    The original model used 3D convolutions over five high-density electrode
    patches. EgoEMG has one circular ring, so this adaptation uses 2D
    convolutions over electrode position and time, matching the adaptation in
    the emg2pose benchmark. Broadband EMG and its 20 Hz low-pass component are
    stacked as the two input feature planes.
    """

    def __init__(
        self,
        conv_blocks: Sequence[SensingDynamicsConvBlock],
        out_channels: int,
        sample_rate: float = 2000.0,
        lowpass_hz: float = 20.0,
        lowpass_order: int = 4,
        include_lowpass: bool = True,
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.out_channels = int(out_channels)
        self.sample_rate = float(sample_rate)
        self.lowpass_hz = float(lowpass_hz)
        self.lowpass_order = int(lowpass_order)
        self.include_lowpass = bool(include_lowpass)
        self.left_context = 0
        self.right_context = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected EMG shaped (B, C, T), got {x.shape}")
        feature_planes = [x]
        if self.include_lowpass:
            feature_planes.append(
                butterworth_lowpass(
                    x,
                    sample_rate=self.sample_rate,
                    cutoff_hz=self.lowpass_hz,
                    order=self.lowpass_order,
                )
            )
        x = torch.stack(feature_planes, dim=1)
        x = self.conv_blocks(x)
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))
        return x.squeeze(2)


class SensingDynamicsMLP(nn.Module):
    """Three-layer per-time-step MLP decoder with SMU activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 512,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            SMU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels),
            SMU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected features shaped (B, C, T), got {x.shape}")
        return self.network(x.transpose(1, 2)).transpose(1, 2)
