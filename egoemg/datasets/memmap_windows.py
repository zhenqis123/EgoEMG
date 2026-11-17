"""Common storage and window lookup for manifest-based pretraining memmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from egoemg.datasets.window_index import build_window_blocks


class MemmapWindowDataset(Dataset):
    """Corpus subclasses supply fields and modality-specific conversion."""

    root_dir: Path
    window_length: int
    stride: int | None
    padding: tuple[int, int]
    jitter: bool
    channel_alignment_mode: str

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing memmap directory at {self.root_dir}")

        self.stride = self.stride or self.window_length
        assert self.window_length > 0 and self.stride > 0
        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0
        if self.channel_alignment_mode not in {"interpolate", "sparse"}:
            raise ValueError(
                f"channel_alignment_mode must be 'interpolate' or 'sparse', got {self.channel_alignment_mode}"
            )

        # Load manifest
        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest at {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.total_rows = self.manifest["total_rows"]
        self.sessions = self.manifest["sessions"]

        indices, _, lengths, cumulative = build_window_blocks(
            [session["length"] for session in self.sessions],
            range(len(self.sessions)),
            self.window_length,
            self.stride,
        )
        self.file_cumsum = cumulative.tolist()
        self.total_windows = self.file_cumsum[-1]
        self.session_starts = [
            (self.sessions[int(index)]["start_idx"], int(length), int(count))
            for index, length, count in zip(indices, lengths, np.diff(cumulative))
        ]

        # Load memmaps
        self._memmaps: dict[str, np.memmap] = {}
        self._load_memmaps()

    def _load_memmaps(self) -> None:
        """Load memmap files from manifest."""
        for field_name, field_info in self.manifest["fields"].items():
            filepath = self.root_dir / field_info["filename"]
            dtype = np.dtype(field_info["dtype"])
            shape = tuple(field_info["shape"])
            self._memmaps[field_name] = np.memmap(
                str(filepath), dtype=dtype, mode="r", shape=shape
            )

    def __len__(self) -> int:
        return self.total_windows

    def _window_slice(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Find which session this window belongs to
        sess_idx = int(np.searchsorted(self.file_cumsum, idx, side="right")) - 1
        if sess_idx < 0 or sess_idx >= len(self.session_starts):
            raise IndexError(f"Session index {sess_idx} out of range")

        sess_start, sess_len, _ = self.session_starts[sess_idx]
        base_windows = self.file_cumsum[sess_idx]
        rel_idx = idx - base_windows

        # Calculate window position
        offset = rel_idx * self.stride
        window_end = offset + self.window_length

        # Handle jitter
        leftover = sess_len - window_end
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Read data
        start = max(offset - self.left_padding, 0)
        end = min(offset + self.window_length + self.right_padding, sess_len)

        # Adjust for global offset
        global_start = sess_start + start
        global_end = sess_start + end

        return global_start, global_end
