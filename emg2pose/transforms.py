# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import pywt
import torch
from scipy.interpolate import CubicSpline


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]
@dataclass
class ExtractField:
    """从 NumPy 结构化数组中提取指定字段（如 'emg'）。"""
    field: str = "emg"

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # data 原本是结构化数组，data[self.field] 返回纯数值数组
        return np.ascontiguousarray(data[self.field])

@dataclass
class ToFloatTensor:
    """简单的 NumPy 到 Tensor 转换。"""
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data).float()

@dataclass
class ExtractToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    field: str = "emg"

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data[self.field])


@dataclass
class ChannelDownsampling:
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :: self.downsampling]


@dataclass
class ChannelSelect:
    """Select a fixed subset of EMG channels by index."""

    indices: Sequence[int]

    def __post_init__(self):
        if len(self.indices) == 0:
            raise ValueError("indices must contain at least one channel index.")
        self._indices = list(self.indices)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[..., self._indices]


@dataclass
class RandomChannelMask:
    """Randomly mask entire channels by zeroing them."""

    mask_prob: float = 0.0
    min_masked: int = 0
    max_masked: int | None = None
    channel_dim: int = -1
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")
        if self.min_masked < 0:
            raise ValueError("min_masked must be >= 0.")
        if self.max_masked is not None and self.max_masked < self.min_masked:
            raise ValueError("max_masked must be >= min_masked.")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.mask_prob <= 0.0 and self.min_masked == 0:
            return data

        n_channels = data.shape[self.channel_dim]
        if n_channels <= 0:
            return data

        mask = np.random.rand(n_channels) < self.mask_prob
        mask_indices = np.flatnonzero(mask)

        if self.max_masked is not None and mask_indices.size > self.max_masked:
            mask_indices = np.random.choice(
                mask_indices, size=self.max_masked, replace=False
            )

        if mask_indices.size < self.min_masked:
            available = np.setdiff1d(np.arange(n_channels), mask_indices)
            if available.size > 0:
                need = min(self.min_masked - mask_indices.size, available.size)
                extra = np.random.choice(available, size=need, replace=False)
                mask_indices = np.concatenate([mask_indices, extra], axis=0)

        if mask_indices.size == 0:
            return data

        masked = data.clone()
        index = torch.as_tensor(mask_indices, dtype=torch.long, device=masked.device)
        return masked.index_fill(self.channel_dim, index, self.mask_value)


@dataclass
class RandomTimeMask:
    """Randomly mask contiguous time segments across all channels."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0
    mask_value: float = 0.0
    mask_prob: float = 1.0

    def __post_init__(self) -> None:
        if self.max_mask_size < 0:
            raise ValueError("max_mask_size must be >= 0.")
        if self.min_mask_size < 0:
            raise ValueError("min_mask_size must be >= 0.")
        if self.max_mask_size < self.min_mask_size:
            raise ValueError("max_mask_size must be >= min_mask_size.")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.max_mask_size == 0 or self.num_masks == 0 or self.mask_prob == 0.0:
            return data

        n_time = data.shape[self.time_dim]
        if n_time <= 0:
            return data

        masked = data.clone()
        for _ in range(self.num_masks):
            if np.random.rand() > self.mask_prob:
                continue
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_time)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_time - mask_size + 1)
            index = torch.arange(
                start,
                start + mask_size,
                device=masked.device,
                dtype=torch.long,
            )
            masked.index_fill_(self.time_dim, index, self.mask_value)

        return masked


@dataclass
class RandomFrequencyMask:
    """Randomly mask contiguous frequency bands via rFFT along time dimension."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0
    mask_prob: float = 1.0

    def __post_init__(self) -> None:
        if self.max_mask_size < 0:
            raise ValueError("max_mask_size must be >= 0.")
        if self.min_mask_size < 0:
            raise ValueError("min_mask_size must be >= 0.")
        if self.max_mask_size < self.min_mask_size:
            raise ValueError("max_mask_size must be >= min_mask_size.")
        if self.num_masks < 0:
            raise ValueError("num_masks must be >= 0.")
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1].")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.max_mask_size == 0 or self.num_masks == 0 or self.mask_prob == 0.0:
            return data

        n_time = data.shape[self.time_dim]
        if n_time <= 0:
            return data

        spec = torch.fft.rfft(data, dim=self.time_dim)
        n_freq = spec.shape[self.time_dim]

        for _ in range(self.num_masks):
            if np.random.rand() > self.mask_prob:
                continue
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_freq)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_freq - mask_size + 1)
            index = torch.arange(
                start,
                start + mask_size,
                device=spec.device,
                dtype=torch.long,
            )
            spec.index_fill_(self.time_dim, index, 0)

        return torch.fft.irfft(spec, n=n_time, dim=self.time_dim)

@dataclass
class RandomGaussianNoise:
    """基于信噪比(SNR)随机添加高斯噪声 (NumPy 版)"""
    min_snr_db: float = 25.0
    max_snr_db: float = 35.0
    mask_prob: float = 0.5
    time_dim: int = 0

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return data

        # 计算信号功率 P = mean(x^2)
        signal_power = np.mean(data**2)
        if signal_power == 0:
            return data

        snr = np.random.uniform(self.min_snr_db, self.max_snr_db)
        noise_power = signal_power / (10 ** (snr / 10))
        noise_std = np.sqrt(noise_power)

        # 生成噪声
        noise = np.random.normal(0, noise_std, size=data.shape)
        return data + noise


@dataclass
class RandomMagnitudeWarping:
    """通过三次样条曲线随机扭曲信号幅值 (NumPy 版)"""
    sigma: float = 0.15
    num_knots: int = 12
    mask_prob: float = 0.5
    time_dim: int = 0
    channel_independent: bool = True

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.mask_prob == 0.0 or self.sigma == 0 or np.random.rand() > self.mask_prob:
            return data

        # 统一转为 (Channel, Time) 进行内部处理
        if self.time_dim == 0:
            x = data.T # (C, T)
        else:
            x = data

        C, T = x.shape
        time_points = np.arange(T)
        knots_x = np.linspace(0, T - 1, self.num_knots)
        
        if self.channel_independent:
            knots_y = np.random.normal(loc=1.0, scale=self.sigma, size=(C, self.num_knots))
            warping_curves = np.zeros((C, T))
            for i in range(C):
                cs = CubicSpline(knots_x, knots_y[i])
                warping_curves[i] = cs(time_points)
        else:
            knots_y = np.random.normal(loc=1.0, scale=self.sigma, size=(self.num_knots,))
            cs = CubicSpline(knots_x, knots_y)
            warping_curves = cs(time_points)[np.newaxis, :]

        out = x * warping_curves
        return out.T if self.time_dim == 0 else out


@dataclass
class RandomWaveletDecomposition:
    """缩放小波细节系数 (NumPy 版)"""
    min_scale: float = 0.8
    max_scale: float = 1.2
    wavelet: str = "db7"
    level: int = 3
    mask_prob: float = 0.3
    time_dim: int = 0

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.mask_prob == 0.0 or np.random.rand() > self.mask_prob:
            return data

        if self.time_dim == 0:
            x = data.T
        else:
            x = data
            
        C, T = x.shape
        augmented_signal = np.zeros_like(x)
        
        for i in range(C):
            try:
                coeffs = pywt.wavedec(x[i], self.wavelet, level=self.level)
                coeffs_aug = [coeffs[0]]
                for cD in coeffs[1:]:
                    scale = np.random.uniform(self.min_scale, self.max_scale)
                    coeffs_aug.append(cD * scale)
                
                res = pywt.waverec(coeffs_aug, self.wavelet)
                
                if len(res) > T: res = res[:T]
                elif len(res) < T: res = np.pad(res, (0, T - len(res)))
                augmented_signal[i] = res
            except:
                augmented_signal[i] = x[i]

        return augmented_signal.T if self.time_dim == 0 else augmented_signal


@dataclass
class RotationAugmentation:
    """旋转通道 (NumPy 版)"""
    channel_dim: int = -1

    def __call__(self, data: np.ndarray) -> np.ndarray:
        rotation = np.random.choice([-1, 0, 1])
        if rotation == 0:
            return data
        return np.roll(data, rotation, axis=self.channel_dim)


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
