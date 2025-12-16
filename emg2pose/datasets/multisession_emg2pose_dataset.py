from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from emg2pose import transforms
from emg2pose.transforms import Transform
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask


# -------- Worker-local HDF5 pool (LRU) --------

class H5Pool:
    """
    Per-process (per-worker) LRU cache of opened HDF5 files.
    Avoids "10k sessions -> 10k open file handles".
    """
    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: str) -> dict[str, Any]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        f = h5py.File(path, "r")
        g = f["emg2pose"]
        ts = g["timeseries"]
        item = {"file": f, "group": g, "timeseries": ts}
        self._cache[path] = item
        self._cache.move_to_end(path)

        if len(self._cache) > self.max_open:
            _, old = self._cache.popitem(last=False)
            try:
                old["file"].close()
            except Exception:
                pass

        return item

    def close_all(self):
        for _, v in list(self._cache.items()):
            try:
                v["file"].close()
            except Exception:
                pass
        self._cache.clear()

    def __del__(self):
        self.close_all()


# -------- Single dataset managing all sessions --------

@dataclass
class MultiSessionWindowedEmgDataset(Dataset):
    hdf5_paths: list[Path]
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=transforms.ExtractToTensor
    )
    skip_ik_failures: bool = False
    max_open_files: int = 16  # H5Pool capacity per worker

    def __post_init__(self):
        assert self.window_length > 0
        self.stride = self.stride or self.window_length
        assert self.stride > 0

        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        # Build a global index: blocks across all sessions
        # Store block metadata in compact arrays (avoid Python-object explosion).
        (
            self._block_session_idx,
            self._block_start,
            self._block_end,
            self._block_cumsum,
        ) = self._build_blocks_index()

        # pool is created lazily per worker/process
        self._pool: H5Pool | None = None

    def _get_pool(self) -> H5Pool:
        # Each worker has its own dataset copy -> its own pool
        if self._pool is None:
            self._pool = H5Pool(max_open=self.max_open_files)
        return self._pool

    def _session_len(self, path: Path) -> int:
        # Open/close quickly only for indexing.
        # This happens once at dataset construction (main process).
        with h5py.File(path, "r") as f:
            return len(f["emg2pose"]["timeseries"])

    def _compute_blocks_for_session(self, path: Path, session_len: int) -> list[tuple[int, int]]:
        """
        Returns list of (start_idx, end_idx_exclusive) blocks.
        If skip_ik_failures=False -> one block (0, session_len).
        If skip_ik_failures=True -> blocks of contiguous valid mask.
        NOTE: This can be expensive if you fully load mask; see comment below.
        """
        if not self.skip_ik_failures:
            return [(0, session_len)] if session_len >= self.window_length else []

        # --- If you want to keep focus on "head1", simplest is to compute blocks by loading mask.
        #     For very long sessions this can be heavy (head2). You can later switch to chunk-scan.
        with h5py.File(path, "r") as f:
            g = f["emg2pose"]
            if "ik_failure_mask" in g:
                # plugin stored as failure mask; we need no-failure mask
                no_fail = ~np.asarray(g["ik_failure_mask"][...], dtype=bool)
            else:
                joint_angles = g["timeseries"]["joint_angles"]
                no_fail = get_ik_failures_mask(joint_angles)

        blocks = get_contiguous_ones(no_fail)
        blocks = [(t0, t1) for (t0, t1) in blocks if (t1 - t0) >= self.window_length]
        return blocks

    def _build_blocks_index(self):
        block_session_idx = []
        block_start = []
        block_end = []
        block_lengths = []

        for si, p in enumerate(self.hdf5_paths):
            slen = self._session_len(p)
            blocks = self._compute_blocks_for_session(p, slen)

            for (s, e) in blocks:
                # number of windows in this block
                n = (e - s - self.window_length) // self.stride + 1
                if n <= 0:
                    continue
                block_session_idx.append(si)
                block_start.append(s)
                block_end.append(e)
                block_lengths.append(n)

        block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
        block_start = np.asarray(block_start, dtype=np.int64)
        block_end = np.asarray(block_end, dtype=np.int64)
        # prefix sum to map global idx -> block
        block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))
        return block_session_idx, block_start, block_end, block_cumsum

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # idx -> block
        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        si = int(self._block_session_idx[bi])
        start_idx = int(self._block_start[bi])
        end_idx = int(self._block_end[bi])
        rel = int(idx - self._block_cumsum[bi])

        offset = start_idx + rel * self.stride

        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")

        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding

        path = str(self.hdf5_paths[si])
        h5 = self._get_pool().get(path)
        ts = h5["timeseries"]

        window = ts[window_start:window_end]  # structured ndarray

        emg = self.transform(window)          # should return Tensor
        assert torch.is_tensor(emg)

        joint_angles = torch.as_tensor(window["joint_angles"])
        finite_mask = torch.isfinite(joint_angles).all(dim=1)  # T
        if not finite_mask.all():
            joint_angles = torch.nan_to_num(joint_angles, nan=0.0, posinf=0.0, neginf=0.0)

        # mask: use ik_failure_mask if exists; otherwise compute-on-the-fly
        g = h5["group"]
        if "ik_failure_mask" in g:
            no_fail = ~np.asarray(g["ik_failure_mask"][window_start:window_end], dtype=bool)
        else:
            # fallback; expensive if happens often; ideally always store ik_failure_mask
            ja_np = window["joint_angles"]
            no_fail = get_ik_failures_mask(ja_np)[...]
        mask = torch.as_tensor(no_fail, dtype=torch.bool)
        mask = mask & finite_mask

        return {
            "emg": emg.T,                       # CT
            "joint_angles": joint_angles.T,      # CT
            "label_valid_mask": mask,            # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "session_idx": si,
        }
