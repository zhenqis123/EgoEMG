"""Optionally overlap pinned host-to-device copies with the previous GPU step."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection
from torch.utils.data import DataLoader


class _PrefetchIterator:
    def __init__(self, source: Iterator, device: torch.device, stream=None):
        self.source = source
        self.device = device
        self.stream = stream if stream is not None else torch.cuda.Stream(device=device)
        self.pending = None
        self._preload()

    def __iter__(self):
        return self

    def _preload(self):
        try:
            cpu_batch = next(self.source)
        except StopIteration:
            self.pending = None
            return
        with torch.cuda.stream(self.stream):
            self.pending = move_data_to_device(cpu_batch, self.device)

    def __next__(self):
        if self.pending is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.pending

        def record(tensor):
            if tensor.device.type == "cuda":
                tensor.record_stream(current)
            return tensor

        apply_to_collection(batch, torch.Tensor, record)
        # Queue the following copy after the wait above, so it can overlap
        # computation on the current batch. Neither tensors nor RNG are changed.
        self._preload()
        return batch


class CudaPrefetchDataLoader(DataLoader):
    """A normal DataLoader with an opt-in, one-batch CUDA prefetch iterator."""

    def __iter__(self) -> Iterator[Any]:
        source = super().__iter__()
        if not torch.cuda.is_available():
            return source
        if not self.pin_memory:
            raise ValueError("CUDA prefetch requires pin_memory=True")
        device = torch.device("cuda", torch.cuda.current_device())
        stream = getattr(self, "_cuda_prefetch_stream", None)
        if stream is None or stream.device != device:
            # Allocator caches belong to their allocation stream. A new stream
            # every epoch accumulates a separate cache for each stream in the
            # CUDA stream pool, even after all batch tensors have been freed.
            stream = self._cuda_prefetch_stream = torch.cuda.Stream(device=device)
        return _PrefetchIterator(source, device, stream=stream)
