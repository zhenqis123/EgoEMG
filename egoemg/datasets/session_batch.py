"""Batch reads into worker-shared buffers for session-v2 datasets.

The public single-sample readers stay usable with ordinary DataLoaders. The
training data module opts into this wrapper only for compatible session readers.
No reordering, RNG draws, target reduction or dtype conversion happen here.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, get_worker_info


class PrecollatedSessionBatch(dict[str, Any]):
    """Marker understood by the data module's collator."""


def batch_signature(dataset: Dataset) -> tuple[Any, ...] | None:
    if isinstance(dataset, ConcatDataset):
        signatures = [batch_signature(child) for child in dataset.datasets]
        if signatures and signatures[0] is not None:
            if all(item == signatures[0] for item in signatures):
                return signatures[0]
        return None
    if callable(getattr(dataset, "_read_into", None)):
        return getattr(dataset, "batch_read_signature", None)
    return None


def _shared_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if get_worker_info() is None:
        return torch.empty(shape, dtype=dtype, device="cpu")
    # Match PyTorch's default collator: allocate the final shared storage
    # directly, avoiding an empty private tensor followed by share_memory_ copy.
    template = torch.empty(0, dtype=dtype, device="cpu")
    storage = template._typed_storage()._new_shared(int(np.prod(shape)))
    return template.new(storage).resize_(shape)


class SessionBatchDataset(Dataset):
    """Keep ordinary item access, but fetch a full batch into its final storage."""

    def __init__(
        self,
        dataset: Dataset,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    ) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]

    def __getattr__(self, name: str) -> Any:
        # Pickle protocols belong to the wrapper, not its wrapped reader.
        if name.startswith("__"):
            raise AttributeError(name)
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _reader(self, index: int) -> tuple[Dataset, int]:
        dataset = self.dataset
        if index < 0 and isinstance(dataset, ConcatDataset):
            if -index > len(dataset):
                raise ValueError("absolute value of index exceeds dataset length")
            index += len(dataset)
        while isinstance(dataset, ConcatDataset):
            child = bisect_right(dataset.cumulative_sizes, index)
            if child:
                index -= dataset.cumulative_sizes[child - 1]
            dataset = dataset.datasets[child]
        return dataset, index

    def __getitems__(self, indices: Sequence[int]) -> PrecollatedSessionBatch:
        if not len(indices):
            return PrecollatedSessionBatch()
        first = self.dataset[indices[0]]
        fields = ("emg", "joint_angles", "label_valid_mask", "vision_img")
        tensors = {
            key: _shared_empty(
                (len(indices), *first[key].shape), torch.from_numpy(first[key]).dtype
            )
            for key in fields
            if key in first
        }
        arrays = {key: value.numpy() for key, value in tensors.items()}
        for key, array in arrays.items():
            array[0] = first.pop(key)
        metadata = [first]
        for position, index in enumerate(indices[1:], 1):
            reader, local_index = self._reader(int(index))
            sample = reader._read_into(
                local_index, {key: value[position] for key, value in arrays.items()}
            )
            metadata.append(
                {key: value for key, value in sample.items() if key not in tensors}
            )
        batch = self.collate_fn(metadata)
        batch.update(tensors)
        return PrecollatedSessionBatch(batch)
