"""Lightweight dataset for merged incre EMG data (right-hand only).

Compatible with :class:`EgoEmgMemmapDataset` output format so it can be
mixed into the same DataLoader via Hydra config without code changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from egoemg.datasets.layout_utils import place_sparse_channels

log = logging.getLogger(__name__)


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


class EgoEmgIncreDataset(Dataset):
    """Right-hand-only incre EMG dataset with minimal field set.

    Emits the same dict keys as ``EgoEmgMemmapDataset`` for the core
    EMG-to-pose training path so the two can be concatenated transparently.

    Missing fields (wrist pitch/yaw, mano, mocap, video, left-hand)
    are either zero-padded or omitted, never raised as errors.
    """

    def __init__(
        self,
        memmap_dir: str | Path,
        window_length: int,
        stride: int | None = None,
        allowed_splits: Sequence[str] | None = None,
        target_hand: str = "right",
        emg_field_preference: str = "filtered",
        emg_layout: str = "emg2pose_interpolate16",
        emg2pose_channel_indices: Sequence[int] | None = None,
        channel_interpolate: bool = False,
        dataset_name: str = "egoemg_incre",
        norm_mode: str | None = None,
        norm_stats_path: str | None = None,
        transform: Any = None,
        jitter: bool = False,
        excluded_episode_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.memmap_dir = Path(memmap_dir)
        self.window_length = window_length
        self.stride = stride or window_length
        self.target_hand = target_hand
        self.emg_field_preference = emg_field_preference
        self.emg_layout = emg_layout
        self.channel_interpolate = channel_interpolate
        self.dataset_name = dataset_name
        self.norm_mode = norm_mode
        self.transform = transform
        self.jitter = jitter

        if target_hand not in {"left", "right"}:
            raise ValueError(f"target_hand must be left/right, got {target_hand}")
        if emg_field_preference not in {"raw", "filtered", "filtered_paper"}:
            raise ValueError(
                f"emg_field_preference must be raw/filtered/filtered_paper, "
                f"got {emg_field_preference}"
            )

        # Resolve channel indices
        if emg2pose_channel_indices is not None:
            indices = np.asarray(emg2pose_channel_indices, dtype=np.int64)
            if indices.ndim != 1:
                raise ValueError("emg2pose_channel_indices must be 1D")
            self._channel_indices = indices
        else:
            self._channel_indices = None

        # Load manifest
        with open(self.memmap_dir / "manifest.json") as f:
            self._manifest = json.load(f)

        # Load metadata
        md = np.load(self.memmap_dir / "metadata.npz", allow_pickle=False)
        self._episode_id = _decode_bytes(md["episode_id"])
        self._episode_subject = _decode_bytes(md["episode_subject"])
        self._episode_subject_id = md["episode_subject_id"]
        self._episode_start_idx = md["episode_start_idx"].astype(np.int64)
        self._episode_end_idx = md["episode_end_idx"].astype(np.int64)
        self._episode_length = md["episode_length"].astype(np.int64)
        self._splits = _decode_bytes(md["splits_split"])
        split_ids = md.get("episode_split_id")
        if split_ids is None:
            split_ids = np.zeros(len(self._episode_id), dtype=np.int32)
        self._episode_split_id = np.asarray(split_ids, dtype=np.int32)

        # Load normalization stats.
        # Default: per-channel + per-hand normalization, mirroring
        # EgoEmgMemmapDataset._load_norm_stats. Looks up field-aware key
        # "{dataset}__{field}_{hand}" with per_channel_mean/std (length-8
        # vectors). Falls back to scalar "{dataset}__{field}" or bare
        # "{dataset}" for back-compat.
        self._emg_mean = 0.0
        self._emg_std = 1.0
        if norm_mode == "per-dataset" and norm_stats_path:
            with open(norm_stats_path, encoding="utf-8") as f:
                stats = json.load(f)

            # Try per-channel + per-hand first (target_hand 8ch layout only).
            used_per_channel = False
            if emg_layout == "target_hand" and target_hand in {"left", "right"}:
                hand_key = f"{dataset_name}__{emg_field_preference}_{target_hand}"
                hand_item = stats.get(hand_key)
                pcm = hand_item.get("per_channel_mean") if hand_item else None
                pcs = hand_item.get("per_channel_std") if hand_item else None
                if pcm is not None and pcs is not None and len(pcm) == len(pcs) == 8:
                    self._emg_mean = np.asarray(pcm, dtype=np.float32)
                    self._emg_std = np.asarray(pcs, dtype=np.float32)
                    used_per_channel = True
                    log.info(
                        "incre norm stats: per-channel + per-hand (%s) for "
                        "dataset=%s field=%s. std=%s",
                        target_hand, dataset_name, emg_field_preference,
                        np.round(self._emg_std, 4).tolist(),
                    )

            if not used_per_channel:
                # Scalar fallback: field-aware key, then bare dataset_name.
                field_key = f"{dataset_name}__{emg_field_preference}"
                item = stats.get(field_key) or stats.get(dataset_name)
                if item is not None and "mean" in item and "std" in item:
                    self._emg_mean = float(item["mean"])
                    self._emg_std = float(item["std"])
                    log.info(
                        "incre norm stats: scalar for dataset=%s field=%s "
                        "(std=%.4f). Per-channel not available.",
                        dataset_name, emg_field_preference, self._emg_std,
                    )

        # Load per-frame split for episodes that mix train/val/test internally
        frame_split_available = "frame_split_id" in self._manifest.get("fields", {})
        if frame_split_available:
            self._frame_split_id = np.memmap(
                self.memmap_dir / "frame_split_id.dat",
                dtype=np.int8,
                mode="r",
                shape=tuple(self._manifest["fields"]["frame_split_id"]["shape"]),
            )
        else:
            self._frame_split_id = None

        # Filter episodes by split (episode-level + per-frame fallback)
        if allowed_splits:
            self._allowed_split_ids = self._resolve_allowed_split_ids(allowed_splits)
            allowed_list = list(self._allowed_split_ids)
            active = np.isin(self._episode_split_id, allowed_list)
            # Also include episodes that have at least one frame matching
            if self._frame_split_id is not None:
                for ep_idx in range(len(self._episode_id)):
                    if active[ep_idx]:
                        continue
                    ep_start = int(self._episode_start_idx[ep_idx])
                    ep_end = int(self._episode_end_idx[ep_idx])
                    if ep_start < ep_end:
                        ep_splits = self._frame_split_id[ep_start:ep_end]
                        if np.any(np.isin(ep_splits, allowed_list)):
                            active[ep_idx] = True
            self._active_episodes = np.nonzero(active)[0].astype(np.int64)
        else:
            self._allowed_split_ids = None
            self._active_episodes = np.arange(len(self._episode_id), dtype=np.int64)

        # Exclude specific episode IDs (applied on top of split filtering)
        if excluded_episode_ids:
            excluded = set(excluded_episode_ids)
            ep_ids = np.asarray(self._episode_id)
            keep = ~np.isin(ep_ids[self._active_episodes], list(excluded))
            self._active_episodes = self._active_episodes[np.nonzero(keep)[0]]

        # Open memmaps for fields we actually use
        emg_key = f"emg_{target_hand}_{emg_field_preference}"
        ja_key = f"generated_joint_angles_{target_hand}"
        self._emg = np.memmap(
            self.memmap_dir / f"{emg_key}.dat",
            dtype=np.float32,
            mode="r",
            shape=tuple(self._manifest["fields"][emg_key]["shape"]),
        )
        self._ja = np.memmap(
            self.memmap_dir / f"{ja_key}.dat",
            dtype=np.float32,
            mode="r",
            shape=tuple(self._manifest["fields"][ja_key]["shape"]),
        )
        lv_key = "generated_label_valid"
        if lv_key in self._manifest["fields"]:
            self._label_valid = np.memmap(
                self.memmap_dir / f"{lv_key}.dat",
                dtype=np.bool_,
                mode="r",
                shape=tuple(self._manifest["fields"][lv_key]["shape"]),
            )
        else:
            self._label_valid = None

        # Build window index
        self._build_window_index()

    def _resolve_allowed_split_ids(self, allowed: Sequence[str]) -> set[int]:
        value_to_id = {v: i for i, v in enumerate(self._splits)}
        # "test" is an alias for "val" (they share the same data)
        aliases = {"test": "val"}
        ids: set[int] = set()
        for a in allowed:
            a = aliases.get(a, a)
            if a in value_to_id:
                ids.add(value_to_id[a])
        return ids

    def _build_window_index(self) -> None:
        block_ep: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        counts: list[int] = []

        use_frame_split = self._frame_split_id is not None and self._allowed_split_ids is not None

        for ep_idx in self._active_episodes.tolist():
            ep_start = int(self._episode_start_idx[ep_idx])
            ep_end = int(self._episode_end_idx[ep_idx])
            n = (ep_end - ep_start - self.window_length) // self.stride + 1
            if n <= 0:
                continue
            ep_count = 0
            for i in range(n):
                s = ep_start + i * self.stride
                e = s + self.window_length
                if use_frame_split:
                    center = s + self.window_length // 2
                    if int(self._frame_split_id[center]) not in self._allowed_split_ids:
                        continue
                block_ep.append(ep_idx)
                block_start.append(s)
                block_end.append(e)
                ep_count += 1
            if ep_count > 0:
                counts.append(ep_count)

        self._block_ep = np.asarray(block_ep, dtype=np.int32)
        self._block_start = np.asarray(block_start, dtype=np.int64)
        self._block_end = np.asarray(block_end, dtype=np.int64)
        self._block_cumsum = np.cumsum(np.asarray([0] + counts, dtype=np.int64))
        self.name = f"{self.dataset_name}_{self.target_hand}"

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def _convert_emg_layout(self, emg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply emg2pose channel layout conversion."""
        if self._channel_indices is None or self.emg_layout == "target_hand":
            mask = np.ones((emg.shape[1],), dtype=bool)
            return emg.astype(np.float32), mask

        target_positions = self._channel_indices
        if self.channel_interpolate:
            raise NotImplementedError("channel_interpolate=True not supported for incre data")
        placed = place_sparse_channels(emg, 16, target_positions)
        mask = np.zeros((16,), dtype=bool)
        mask[target_positions] = True
        return placed, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx = int(self._block_ep[idx])
        start = int(self._block_start[idx])
        end = int(self._block_end[idx])

        # Jitter: random offset within the stride
        if self.jitter:
            ep_end = int(self._episode_end_idx[ep_idx])
            leftover = ep_end - end
            if leftover > 0:
                offset = np.random.randint(0, min(self.stride, leftover))
                start += offset
                end = start + self.window_length

        hand_idx = 0 if self.target_hand == "left" else 1

        # EMG
        emg = np.asarray(self._emg[start:end], dtype=np.float32)
        emg, emg_channel_mask = self._convert_emg_layout(emg)
        if self.norm_mode == "per-dataset":
            emg = (emg - self._emg_mean) / (self._emg_std + 1e-6)
        if self.transform is not None:
            emg = self.transform({"emg": emg})
            if isinstance(emg, dict):
                emg = emg.get("emg", emg)
            if hasattr(emg, "numpy"):
                emg = emg.numpy()
        emg = emg.T  # (C, T)

        # Joint angles  →  22-dim (zero-pad missing wrist pitch/yaw)
        ja = np.asarray(self._ja[start:end], dtype=np.float32)  # (T, 20)
        ja = np.concatenate(
            [ja, np.zeros((ja.shape[0], 2), dtype=np.float32)], axis=1
        )  # (T, 22)
        ja = ja.T  # (22, T)

        # Label valid
        if self._label_valid is not None:
            label_valid = np.asarray(self._label_valid[start:end, hand_idx], dtype=bool)
        else:
            label_valid = np.ones((end - start,), dtype=bool)

        return {
            "emg": emg,
            "emg_channel_mask": emg_channel_mask,
            "joint_angles": ja,
            "label_valid_mask": label_valid,
            "episode_id": self._episode_id[ep_idx],
            "episode_subject": self._episode_subject[ep_idx],
            "episode_subject_id": int(self._episode_subject_id[ep_idx]),
            "episode_source_parquet": "",
            "episode_zed_video_path": "",
            "episode_webcam_video_path": "",
            "window_start_idx": start,
            "window_end_idx": end,
            "window_length": self.window_length,
            "target_hand": self.target_hand,
            "dataset_name": self.dataset_name,
        }
