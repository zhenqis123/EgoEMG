# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


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
class RotationAugmentation:
    """Rotate EMG along the channel dimension by a random integer."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice([-1, 0, 1])
        return torch.roll(data, rotation, dims=-1)


@dataclass
class ChannelDownsampling:
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :: self.downsampling]


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
