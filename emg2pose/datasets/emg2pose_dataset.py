# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Memmap-based dataset for EMG2Pose corpus.

This version reads from numpy memmap files (.dat) with metadata stored in
manifest.json and metadata.npz. This provides faster random access compared
to zarr, especially for large-scale training.

The original HDF5-based implementation is preserved in emg2pose_dataset_legacy.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from emg2pose import transforms
from emg2pose.transforms import Transform


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


def _as_tensor(value: Any) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _safe_np(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x)


def _resolve_str_filter(
    values: Sequence[str],
    allowed: Sequence[str] | None,
    label: str,
) -> set[int]:
    if not allowed:
        return set()
    value_to_id = {v: i for i, v in enumerate(values)}
    ids: set[int] = set()
    for a in allowed:
        if a not in value_to_id:
            raise ValueError(f"Unknown {label}: {a}")
        ids.add(int(value_to_id[a]))
    return ids


@dataclass
class Emg2PoseDataset(Dataset):
    """Windowed dataset for the global-concatenated EMG2Pose corpus using memmap.

    Reads from precomputed numpy memmap files for fast random access.
    Metadata (sessions, splits, users, stages) is loaded from metadata.npz.

    Args:
        memmap_dir: Path to memmap directory (required, or use root_dir)
        root_dir: Legacy alias for memmap_dir (backward compatibility)
    """

    memmap_dir: Path | None = None
    root_dir: Path | None = None  # Legacy alias for backward compatibility
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Transform[Any, Any] = field(
        default_factory=transforms.ExtractStructuredFields
    )
    transform_input: str = "dict"  # dict | structured
    skip_ik_failures: bool = True
    norm_mode: str | None = None  # null|batch|instance|per-dataset
    norm_eps: float = 1e-6
    norm_stats_path: str | None = None  # path to per-dataset stats JSON (for per-dataset mode)
    dataset_name: str | None = None  # name key for per-dataset normalization
    allowed_sessions: Sequence[str] | None = None
    allowed_session_indices: Sequence[int] | None = None
    allowed_users: Sequence[str] | None = None
    allowed_stages: Sequence[str] | None = None
    allowed_sides: Sequence[str] | None = None
    allowed_splits: Sequence[str] | None = None
    skip_fields: Sequence[str] | None = None  # fields to skip reading
    channel_indices: Sequence[int] | None = None  # select specific channels from 16ch EMG
    pad_joint_angles_to: int | None = None  # pad joint_angles to N dims (e.g. 20→22 for co-training with EgoEMG)

    def __post_init__(self) -> None:
        self._skip_fields: set[str] = set(self.skip_fields or [])

        # Handle root_dir vs memmap_dir
        if self.memmap_dir is None and self.root_dir is None:
            raise ValueError("Either memmap_dir or root_dir must be provided")
        if self.memmap_dir is None:
            # root_dir provided - check if it's memmap or legacy zarr
            self.memmap_dir = Path(self.root_dir)
            # If root_dir doesn't have manifest.json, try _memmap suffix
            if not (self.memmap_dir / "manifest.json").exists():
                root_str = str(self.root_dir).rstrip("/")
                memmap_alt = Path(root_str + "_memmap")
                if memmap_alt.exists():
                    self.memmap_dir = memmap_alt
                else:
                    raise FileNotFoundError(
                        f"No memmap found at {self.root_dir} or {memmap_alt}. "
                        f"Please run zarr_to_memmap conversion first."
                    )
        else:
            self.memmap_dir = Path(self.memmap_dir)

        if not self.memmap_dir.exists():
            raise FileNotFoundError(f"Missing memmap directory at {self.memmap_dir}")

        self.stride = self.stride or self.window_length
        assert self.window_length > 0 and self.stride > 0
        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        self._memmaps: dict[str, np.memmap] = {}
        self._manifest: dict | None = None
        self._metadata: dict | None = None

        self._load_memmaps()
        self._load_metadata()
        self._build_blocks_index()
        self._prepare_norm_stats()

        self._channel_indices = (
            np.asarray(self.channel_indices, dtype=np.int64)
            if self.channel_indices is not None
            else None
        )

    def _load_memmaps(self) -> None:
        """Load memmap files from manifest.json."""
        manifest_path = self.memmap_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path) as f:
            self._manifest = json.load(f)

        # Load memmap arrays
        for field_name, info in self._manifest["fields"].items():
            dat_path = self.memmap_dir / info["filename"]
            if not dat_path.exists():
                continue
            dtype = np.dtype(info["dtype"])
            shape = tuple(info["shape"])
            self._memmaps[field_name] = np.memmap(
                str(dat_path), dtype=dtype, mode="r", shape=shape,
            )

    def _load_metadata(self) -> None:
        """Load session metadata from metadata.npz."""
        metadata_npz = self.memmap_dir / "metadata.npz"
        if not metadata_npz.exists():
            raise FileNotFoundError(
                f"Missing metadata.npz. Run scripts/extract_metadata_to_memmap.py first."
            )

        data = np.load(metadata_npz)

        # Session arrays
        self._session_start = data["session_start_idx"]
        self._session_length = data["session_length"]
        self._session_end = data["session_end_idx"]
        self._session_user_id = data["session_user_id"]
        self._session_stage_id = data["session_stage_id"]
        self._session_side_id = data["session_side_id"]
        self._session_split_id = data["session_split_id"]
        self._session_held_out_user = data["session_held_out_user"]
        self._session_held_out_stage = data["session_held_out_stage"]

        # Session identifiers (decode bytes)
        self._session_id = _decode_bytes(data["session_session_id"])
        self._session_filename = _decode_bytes(data["session_filename"])

        # Lookup tables
        self._users = _decode_bytes(data["users_user"])
        self._stages = _decode_bytes(data["stages_stage"])
        self._sides = _decode_bytes(data["sides_side"])
        self._splits = _decode_bytes(data["splits_split"])

        # Blocks for skip_ik_failures
        self._blocks_session_idx = data["blocks_session_idx"]
        self._blocks_start = data["blocks_start"]
        self._blocks_end = data["blocks_end"]
        self._blocks_length = data["blocks_length"]

        # Stats not available in metadata.npz for now
        self._stats_sum = None
        self._stats_sumsq = None
        self._stats_count = None

    def _filter_session_indices(self) -> np.ndarray:
        n_sessions = len(self._session_id)
        mask = np.ones((n_sessions,), dtype=bool)

        if self.allowed_session_indices:
            allowed = np.asarray(self.allowed_session_indices, dtype=np.int64)
            mask &= np.isin(np.arange(n_sessions), allowed)

        if self.allowed_sessions:
            allowed_set = set(self.allowed_sessions)
            session_mask = np.array(
                [
                    (sid in allowed_set) or (fn in allowed_set)
                    for sid, fn in zip(self._session_id, self._session_filename)
                ],
                dtype=bool,
            )
            mask &= session_mask

        if self.allowed_users:
            allowed_user_ids = _resolve_str_filter(
                self._users, self.allowed_users, "user"
            )
            mask &= np.isin(self._session_user_id, list(allowed_user_ids))

        if self.allowed_stages:
            allowed_stage_ids = _resolve_str_filter(
                self._stages, self.allowed_stages, "stage"
            )
            mask &= np.isin(self._session_stage_id, list(allowed_stage_ids))

        if self.allowed_sides:
            allowed_side_ids = _resolve_str_filter(
                self._sides, self.allowed_sides, "side"
            )
            mask &= np.isin(self._session_side_id, list(allowed_side_ids))

        if self.allowed_splits:
            allowed_split_ids = _resolve_str_filter(
                self._splits, self.allowed_splits, "split"
            )
            mask &= np.isin(self._session_split_id, list(allowed_split_ids))

        return np.nonzero(mask)[0].astype(np.int64)

    def _build_blocks_index(self) -> None:
        allowed_sessions = self._filter_session_indices()

        if self.skip_ik_failures:
            # Use blocks from metadata.npz (precomputed from valid_mask)
            session_idx = self._blocks_session_idx
            start = self._blocks_start
            end = self._blocks_end
            mask = np.isin(session_idx, allowed_sessions)
            session_idx = session_idx[mask]
            start = start[mask]
            end = end[mask]
        else:
            session_idx = []
            start = []
            end = []
            for sidx in allowed_sessions.tolist():
                slen = int(self._session_length[sidx])
                if slen <= 0:
                    continue
                sstart = int(self._session_start[sidx])
                session_idx.append(sidx)
                start.append(sstart)
                end.append(sstart + slen)
            session_idx = np.asarray(session_idx, dtype=np.int32)
            start = np.asarray(start, dtype=np.int64)
            end = np.asarray(end, dtype=np.int64)

        block_lengths: list[int] = []
        keep: list[bool] = []
        for i in range(len(start)):
            n = (end[i] - start[i] - self.window_length) // self.stride + 1
            if n > 0:
                block_lengths.append(int(n))
                keep.append(True)
            else:
                keep.append(False)

        if keep:
            keep_mask = np.asarray(keep, dtype=bool)
            self._block_session_idx = session_idx[keep_mask]
            self._block_start = start[keep_mask]
            self._block_end = end[keep_mask]
            self._block_cumsum = np.cumsum(
                np.asarray([0] + block_lengths, dtype=np.int64)
            )
        else:
            self._block_session_idx = np.asarray([], dtype=np.int32)
            self._block_start = np.asarray([], dtype=np.int64)
            self._block_end = np.asarray([], dtype=np.int64)
            self._block_cumsum = np.asarray([0], dtype=np.int64)

    def _prepare_norm_stats(self) -> None:
        self._emg_mean = 0.0
        self._emg_std = 1.0
        self._norm_ready = False

        # batch normalization is applied in lightning.py _step(), not here.
        if not self.norm_mode or self.norm_mode in {"batch", "instance"}:
            return

        if self.norm_mode == "per-dataset" and self.norm_stats_path:
            import json
            with open(self.norm_stats_path) as f:
                all_stats = json.load(f)
            name = self.dataset_name or "emg2pose"
            stats = all_stats.get(name, {"mean": 0.0, "std": 1.0})
            self._emg_mean = float(stats["mean"])
            self._emg_std = float(stats["std"])
            self._norm_ready = True
        else:
            print(
                f"Warning: norm_mode='{self.norm_mode}' not supported in memmap version. "
                f"Use 'batch', 'instance', 'per-dataset', or null instead."
            )

    def _apply_norm(self, emg: torch.Tensor, *, user_id: int, side: str) -> torch.Tensor:
        if not self.norm_mode or self.norm_mode in {"batch", "null"}:
            return emg
        eps = float(self.norm_eps)
        if self.norm_mode == "instance":
            mean = emg.mean()
            std = emg.std()
            return (emg - mean) / (std + eps)
        if self.norm_mode == "per-dataset" and self._norm_ready and self._emg_std > 0:
            return (emg - self._emg_mean) / (self._emg_std + eps)
        if self.norm_mode == "batch" and self._norm_ready and self._emg_std > 0:
            if self._channel_std is not None:
                channel_mean = torch.as_tensor(self._channel_mean, dtype=emg.dtype, device=emg.device)
                channel_std = torch.as_tensor(self._channel_std, dtype=emg.dtype, device=emg.device)
                return (emg - channel_mean) / (channel_std + eps)
            return (emg - self._emg_mean) / (self._emg_std + eps)
        return emg

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def _get_window(self, start: int, end: int) -> dict[str, Any]:
        n = end - start
        mm = self._memmaps
        result: dict[str, Any] = {}

        # EMG: always needed
        result["emg"] = np.array(mm["emg"][start:end])
        if self._channel_indices is not None:
            result["emg"] = result["emg"][:, self._channel_indices]

        # joint_angles
        if "joint_angles" not in self._skip_fields:
            result["joint_angles"] = np.array(mm["joint_angles"][start:end])
        else:
            result["joint_angles"] = np.zeros((n, 20), dtype=np.float32)
        if self.pad_joint_angles_to is not None:
            cur_dim = result["joint_angles"].shape[1]
            target = self.pad_joint_angles_to
            if cur_dim < target:
                pad = np.zeros((n, target - cur_dim), dtype=result["joint_angles"].dtype)
                result["joint_angles"] = np.concatenate([result["joint_angles"], pad], axis=1)

        # time
        if "time" not in self._skip_fields:
            result["time"] = np.array(mm["time"][start:end])
        else:
            result["time"] = np.zeros(n, dtype=np.float64)

        # valid_mask
        if "valid_mask" not in self._skip_fields:
            vm = np.array(mm["valid_mask"][start:end]).astype(bool)
            # Fallback: if valid_mask is all-False, treat every sample as valid
            if not vm.any():
                vm = np.ones(n, dtype=bool)
            result["valid_mask"] = vm
        else:
            result["valid_mask"] = np.ones(n, dtype=bool)

        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

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

        session_start = int(self._session_start[si])
        session_end = int(self._session_end[si])

        window_start = max(offset - self.left_padding, session_start)
        window_end = min(
            offset + self.window_length + self.right_padding, session_end
        )
        window_start_local = window_start - session_start
        window_end_local = window_end - session_start

        window = self._get_window(window_start, window_end)

        if self.transform_input == "structured":
            emg = window["emg"]
            joint_angles = window["joint_angles"]
            time = window["time"]
            dtype = np.dtype(
                [
                    ("time", time.dtype),
                    ("joint_angles", joint_angles.dtype, (joint_angles.shape[1],)),
                    ("emg", emg.dtype, (emg.shape[1],)),
                ]
            )
            structured = np.empty((emg.shape[0],), dtype=dtype)
            structured["time"] = time
            structured["joint_angles"] = joint_angles
            structured["emg"] = emg
            transform_input = structured
        elif self.transform_input == "dict":
            transform_input = {
                "emg": window["emg"],
                "joint_angles": window["joint_angles"],
                "time": window["time"],
            }
        else:
            raise ValueError(
                "transform_input must be 'dict' or 'structured'. "
                f"Got: {self.transform_input}"
            )

        transformed = self.transform(transform_input) if self.transform else transform_input

        joint_angles = None
        if isinstance(transformed, tuple) and len(transformed) == 2:
            emg, joint_angles = transformed
        elif isinstance(transformed, dict):
            emg = transformed.get("emg")
            if emg is None:
                raise ValueError("Transform dict output must include 'emg'.")
            joint_angles = transformed.get("joint_angles")
        else:
            emg = transformed

        if joint_angles is None:
            joint_angles = window["joint_angles"]

        user_id = int(self._session_user_id[si])
        side = str(self._sides[int(self._session_side_id[si])])

        # Convert to tensor for norm application, then back to numpy
        emg_t = _as_tensor(emg)
        emg_t = self._apply_norm(emg_t, user_id=user_id, side=side)
        emg = emg_t.numpy()

        finite_mask = np.isfinite(joint_angles).all(axis=1)
        if not finite_mask.all():
            joint_angles = np.nan_to_num(
                joint_angles, nan=0.0, posinf=0.0, neginf=0.0
            )

        mask = window["valid_mask"] & finite_mask

        return {
            "emg": emg.T,
            "joint_angles": joint_angles.T,
            "label_valid_mask": mask,
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "session": self._session_id[si],
            "user": self._users[user_id],
            "side": side,
            "stage": self._stages[int(self._session_stage_id[si])],
            "held_out_user": bool(self._session_held_out_user[si]),
            "held_out_stage": bool(self._session_held_out_stage[si]),
        }


def _describe_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    return f"{type(value).__name__}"


def _print_sample(sample: dict[str, Any]) -> None:
    for key in sorted(sample.keys()):
        print(f"  {key}: {_describe_value(sample[key])}")


def _parse_padding(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("padding must be 'left,right'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick smoke test for Emg2PoseDataset (Memmap)."
    )
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        required=True,
        help="Path to the memmap dataset directory.",
    )
    parser.add_argument("--window-length", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--padding",
        type=_parse_padding,
        default=(0, 0),
        help="Left,right padding (e.g., 0,0).",
    )
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--skip-ik-failures", action="store_true")
    parser.add_argument(
        "--allowed-sessions",
        nargs="*",
        default=None,
        help="Session IDs or filenames to include.",
    )
    parser.add_argument("--allowed-users", nargs="*", default=None)
    parser.add_argument("--allowed-stages", nargs="*", default=None)
    parser.add_argument("--allowed-sides", nargs="*", default=None)
    parser.add_argument("--allowed-splits", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    dataset = Emg2PoseDataset(
        memmap_dir=args.memmap_dir,
        window_length=args.window_length,
        stride=args.stride,
        padding=args.padding,
        jitter=args.jitter,
        skip_ik_failures=args.skip_ik_failures,
        allowed_sessions=args.allowed_sessions,
        allowed_users=args.allowed_users,
        allowed_stages=args.allowed_stages,
        allowed_sides=args.allowed_sides,
        allowed_splits=args.allowed_splits,
    )

    allowed_sessions = dataset._filter_session_indices()
    print(f"Sessions: {len(allowed_sessions)}")
    print(f"Total windows: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty with current filters.")
        return

    n = min(args.num_samples, len(dataset))
    if args.sequential:
        indices = list(range(n))
    else:
        rng = np.random.default_rng(args.seed)
        indices = rng.integers(0, len(dataset), size=n).tolist()

    for i, idx in enumerate(indices):
        print(f"Sample {i} (idx={idx}):")
        sample = dataset[int(idx)]
        _print_sample(sample)


if __name__ == "__main__":
    main()