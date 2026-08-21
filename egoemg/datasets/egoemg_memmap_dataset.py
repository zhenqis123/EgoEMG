"""Memmap-backed EgoEMG dataset for fast multimodal training."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset, get_worker_info

from egoemg.datasets.egoemg_vision_dataset import (
    _build_intrinsics_and_frame_mapper,
    _convert_cvimg_to_tensor,
    _expand_to_aspect_ratio,
    _generate_image_patch_cv2,
    _get_bbox,
    _map_processed_points_to_raw,
    _project_world_points,
)
from egoemg.datasets.layout_utils import (
    circular_interpolate,
    place_sparse_channels,
    sparse_ring_interpolate,
)
from egoemg.video_io import (
    DEFAULT_ALLINTRA_SUFFIX,
    DecordVideoReaderCache,
    resolve_allintra_video_path,
)


log = logging.getLogger(__name__)


# py-lmdb permits only one Environment for a given path within a process.
# Fusion uses separate left/right dataset instances which can read the same
# episode crop database in one batch, so the environments must be shared.
# Values are ``[environment, number_of_dataset_caches]`` and are scoped by PID
# to remain safe with DataLoader workers created by fork.
_SHARED_READONLY_LMDB_ENVS: dict[tuple[int, str], list[Any]] = {}


def _acquire_readonly_lmdb(path: Path):
    """Return a process-shared read-only LMDB environment for ``path``."""
    import lmdb

    key = (os.getpid(), str(path.resolve()))
    entry = _SHARED_READONLY_LMDB_ENVS.get(key)
    if entry is None:
        entry = [lmdb.open(str(path), readonly=True, lock=False, readahead=False), 0]
        _SHARED_READONLY_LMDB_ENVS[key] = entry
    entry[1] += 1
    return entry[0]


def _release_readonly_lmdb(path: Path, env: Any) -> None:
    """Release one dataset-cache reference, closing an unused environment."""
    key = (os.getpid(), str(path.resolve()))
    entry = _SHARED_READONLY_LMDB_ENVS.get(key)
    if entry is None or entry[0] is not env:
        return
    entry[1] -= 1
    if entry[1] <= 0:
        env.close()
        del _SHARED_READONLY_LMDB_ENVS[key]



def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


def _resolve_str_filter(values: Sequence[str], allowed: Sequence[str] | None, label: str) -> set[int]:
    if not allowed:
        return set()
    value_to_id = {v: i for i, v in enumerate(values)}
    ids: set[int] = set()
    for a in allowed:
        if a not in value_to_id:
            raise ValueError(f"Unknown {label}: {a}")
        ids.add(int(value_to_id[a]))
    return ids


MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "emg": (
        "emg_left_raw",
        "emg_right_raw",
        "emg_left_filtered_paper",
        "emg_right_filtered_paper",
    ),
    "imu": (
        "imu_band_left",
        "imu_band_right",
        "imu_cam_head",
        "imu_cam_wrist_left",
        "imu_cam_wrist_right",
    ),
    "mocap_hands": (
        "mocap_left_keypoints",
        "mocap_right_keypoints",
        "mocap_left_valid",
        "mocap_right_valid",
    ),
    "wrist": (
        "mocap_left_wrist_position",
        "mocap_left_wrist_orientation",
        "mocap_left_wrist_pitch",
        "mocap_left_wrist_yaw",
        "mocap_left_wrist_angles_valid",
        "mocap_left_wrist_rigid_markers",
        "mocap_right_wrist_position",
        "mocap_right_wrist_orientation",
        "mocap_right_wrist_pitch",
        "mocap_right_wrist_yaw",
        "mocap_right_wrist_angles_valid",
        "mocap_right_wrist_rigid_markers",
    ),
    "head_mocap": (
        "mocap_head_position",
        "mocap_head_orientation",
        "mocap_head_valid",
        "mocap_head_rigid_markers",
    ),
    "video_index": (
        "image_zed_frame_index",
        "image_head_frame_index",
        "image_wrist_left_frame_index",
        "image_wrist_right_frame_index",
        "image_zed_stale",
        "image_zed_delta_ms",
        "image_head_stale",
        "image_head_delta_ms",
        "image_wrist_left_stale",
        "image_wrist_left_delta_ms",
        "image_wrist_right_stale",
        "image_wrist_right_delta_ms",
    ),
    "labels": (
        "label_gesture_class",
        "label_gesture_active",
        "generated_label_valid",
    ),
    "split": ("frame_split_id",),
    "mano": (
        "generated_mano_left_pose",
        "generated_mano_right_pose",
        "generated_mano_left_beta",
        "generated_mano_right_beta",
    ),
    "joint_angles": (
        "generated_joint_angles_left",
        "generated_joint_angles_right",
    ),
    "timing": (
        "timestamp_us",
        "episode_index",
        "frame_index",
        "subject_id",
        "is_first",
        "is_last",
    ),
}

VIDEO_STREAM_PATH_KEYS = {
    "head": "episode_head_video_path",
    "wrist_left": "episode_wrist_left_video_path",
    "wrist_right": "episode_wrist_right_video_path",
    "zed": "episode_zed_video_path",
}


@dataclass
class EgoEmgMemmapDataset(Dataset):
    memmap_dir: Path
    window_length: int
    video_root: Path | None = None
    allintra_root: Path | None = None
    allintra_suffix: str = DEFAULT_ALLINTRA_SUFFIX
    stride: int | None = None
    eval_center_stride: int | None = None  # fixed center-frame grid for val/test (independent of WL)
    jitter: bool = False
    allowed_subjects: Sequence[str] | None = None
    allowed_episode_ids: Sequence[str] | None = None
    excluded_episode_ids: Sequence[str] | None = None
    allowed_splits: Sequence[str] | None = None
    skip_fields: Sequence[str] | None = None
    modalities: Sequence[str] | None = None
    fields: Sequence[str] | None = None
    video_streams: Sequence[str] | None = None
    decode_video: bool = False
    target_hand: str | None = None
    emg_field_preference: str = "filtered_paper"
    emg_layout: str = "target_hand"
    emg2pose_channel_indices: Sequence[int] | None = None
    channel_interpolate: bool = True  # True: sparse_ring_interpolate to 16ch; False: place_sparse_channels (zero-fill)
    norm_mode: str | None = None
    norm_stats_path: str | None = None
    dataset_name: str = "egoemg"
    transform: Any | None = None
    mano_npy_dir: Path | None = None
    vision_patch_size: int = 256
    vision_num_frames: int = 0
    vision_frame_stride: int = 1
    vision_frame_selection: str = "center"
    vision_return_mode: str = "patch"
    vision_mean: Sequence[float] | None = None
    vision_std: Sequence[float] | None = None
    per_episode_crops_dir: str | None = None
    cached_vit_features_dir: str | None = None
    preload_features: bool = False  # preload all ViT features into RAM before training
    skip_emg_loading: bool = False  # vision_only mode: skip all EMG I/O & transforms
    center_target_only: bool = False  # center_supervised: full EMG, point-read all else

    def __post_init__(self) -> None:
        self.memmap_dir = Path(self.memmap_dir)
        self.video_root = Path(self.video_root) if self.video_root is not None else self.memmap_dir.parent
        self.allintra_root = (
            Path(self.allintra_root)
            if self.allintra_root is not None
            else self.video_root.parent / "EgoEMG_allintra"
        )
        self.stride = self.stride or self.window_length
        if self.target_hand not in {None, "left", "right"}:
            raise ValueError(f"target_hand must be one of None/left/right, got {self.target_hand}")
        if self.emg_field_preference not in {
            "raw",
            "filtered",
            "filtered_paper",
        }:
            raise ValueError(
                "emg_field_preference must be one of "
                "raw/filtered/filtered_paper, "
                f"got {self.emg_field_preference}"
            )
        if self.emg_layout not in {
            "target_hand",
            "emg2pose_sparse16",
            "emg2pose_interpolate16",
        }:
            raise ValueError(
                "emg_layout must be one of "
                "target_hand/emg2pose_sparse16/emg2pose_interpolate16, "
                f"got {self.emg_layout}"
            )
        if self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"} and self.target_hand is None:
            raise ValueError(f"{self.emg_layout} requires target_hand to be set")
        self._skip_fields = set(self.skip_fields or [])
        self._manifest: dict[str, Any] | None = None
        self._frame_memmaps: dict[str, np.memmap] = {}
        self._episode_memmaps: dict[str, np.memmap] = {}
        self._frame_split_available = False
        self._emg_mean = 0.0
        self._emg_std = 1.0
        self._emg2pose_channel_indices = self._resolve_emg2pose_channel_indices()
        self._video_reader_cache = DecordVideoReaderCache()
        self._intrinsics_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        self._runtime_pid = os.getpid()
        self._vision_enabled = self.vision_num_frames > 0
        self._cached_vit_enabled = self.cached_vit_features_dir is not None
        self._center_target_only = self.center_target_only and not self.skip_emg_loading
        if self._cached_vit_enabled:
            # Need target_hand to select left/right feature files
            if self.target_hand is None:
                raise ValueError("cached_vit_features_dir requires target_hand to be set")
            self._cached_vit_features_dir = Path(self.cached_vit_features_dir)
            self._init_cached_vit_features()
        if self.vision_frame_selection not in {"center"}:
            raise ValueError(
                f"vision_frame_selection must be 'center', got {self.vision_frame_selection}"
            )
        if self.vision_return_mode not in {"patch"}:
            raise ValueError(
                f"vision_return_mode must be 'patch', got {self.vision_return_mode}"
            )
        self.vision_mean = (
            np.asarray(self.vision_mean, dtype=np.float32)
            if self.vision_mean is not None
            else 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
        )
        self.vision_std = (
            np.asarray(self.vision_std, dtype=np.float32)
            if self.vision_std is not None
            else 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)
        )
        self._load_manifest()
        self._frame_fields_to_load, self._episode_fields_to_load = self._resolve_requested_fields()
        # Always include wrist pitch/yaw/valid fields when joint_angles are requested
        # so that 22-dim output (20 joint angles + 2 wrist) can be assembled.
        if self.target_hand and {"emg", "joint_angles", "labels"} & set(self.modalities or []):
            hand = self.target_hand
            extra_fields = {
                f"mocap_{hand}_wrist_pitch",
                f"mocap_{hand}_wrist_yaw",
                f"mocap_{hand}_wrist_angles_valid",
            }
            manifest_fields = set(self._manifest["fields"].keys())
            self._frame_fields_to_load |= (extra_fields & manifest_fields)
        # Always load the per-frame source-id field when present (needed by
        # _resolve_dataset_name for physically-merged memmaps).
        if "dataset_source_id" in self._manifest.get("fields", {}):
            self._frame_fields_to_load.add("dataset_source_id")
        # Prune unused hand-specific fields to reduce memmap I/O.
        # __getitem__ only accesses the target hand's EMG/JA/wrist/MANO fields,
        # but MODALITY_GROUPS is coarse-grained (e.g. "emg" pulls 4 variants,
        # "joint_angles" pulls both hands). Remove fields for the non-target hand
        # and the non-preferred EMG variant before opening memmaps.
        if self.target_hand is not None:
            pruned = self._prune_hand_fields(
                self._frame_fields_to_load,
                target_hand=self.target_hand,
                emg_field_preference=self.emg_field_preference,
                emg_layout=self.emg_layout,
            )
            # Fail at construction with a config-pointing message when the
            # preferred EMG variant does not exist for the target hand (the
            # v3 schema ships raw + filtered_paper for both hands only).
            required_emg = f"emg_{self.target_hand}_{self.emg_field_preference}"
            if (
                self.emg_field_preference != "raw"
                and "emg" in (self.modalities or ("emg",))
                and required_emg not in self._manifest["fields"]
            ):
                raise ValueError(
                    f"emg_field_preference={self.emg_field_preference!r} with "
                    f"target_hand={self.target_hand!r} requires the memmap field "
                    f"{required_emg!r}, which does not exist. Available EMG "
                    f"variants: "
                    f"{sorted(f for f in self._manifest['fields'] if f.startswith('emg_'))}. "
                    "Use emg_field_preference=raw or filtered_paper."
                )
            self._frame_fields_to_load = pruned
        if self.skip_emg_loading:
            emg_fields = {f for f in self._frame_fields_to_load if f.startswith("emg_")}
            self._frame_fields_to_load -= emg_fields
            self._skip_fields |= emg_fields
        if self._vision_enabled:
            if self.target_hand is None:
                raise ValueError("vision_num_frames>0 requires target_hand to be set")
            self._frame_fields_to_load |= {"image_head_frame_index"}
            # Only add mocap fields when not using pre-crops (needed for bbox computation)
            if self.per_episode_crops_dir is None:
                self._frame_fields_to_load |= {
                    "mocap_head_transform",
                    "mocap_head_valid",
                    "generated_label_valid",
                    f"mocap_{self.target_hand}_keypoints",
                    f"mocap_{self.target_hand}_valid",
                }
        if self._cached_vit_enabled:
            self._frame_fields_to_load |= {"image_head_frame_index"}
        if self.allowed_splits and self._frame_split_available:
            self._skip_fields.discard("frame_split_id")
        self._skip_fields |= {
            name
            for name in self._manifest["fields"].keys()
            if name not in self._frame_fields_to_load
        }
        if self.allowed_splits and self._frame_split_available:
            self._skip_fields.discard("frame_split_id")
        self._load_metadata()
        if not self.skip_emg_loading:
            # EMG is never read in vision_only mode, so skip the norm-stats
            # load (and its "raw-EMG statistics" sanity warning) entirely.
            self._load_norm_stats()
        self._open_memmaps()
        self._load_mano_npy()
        self._load_vision_calibration()
        self._build_window_index()
        self.name = (
            f"{self.dataset_name}_{self.target_hand}"
            if self.target_hand is not None
            else self.dataset_name
        )

        # Preload ViT features into RAM as a flat array indexed by dataset position.
        # A plain np.ndarray avoids COW explosions in forked DataLoader workers
        # because the data buffer stays read-only — no Python-object refcount
        # modifications inside the buffer, unlike a dict-of-tuples.
        self._preloaded_features: np.ndarray | None = None
        self._preloaded_features_valid: np.ndarray | None = None
        self._preloaded_joint_angles: np.ndarray | None = None  # (N, 22)
        self._preloaded_label_valid: np.ndarray | None = None    # (N,)
        if self.preload_features and self._cached_vit_enabled:
            self._preload_all_features()

        # Pre-crop LMDB cache (fork-safe)
        self._pec_lmdb_cache: dict[str, tuple[Path, Any, Any]] = {}
        self._pec_lmdb_max_open = 8
        self._pec_pid = os.getpid()
        if self.per_episode_crops_dir is not None:
            self._init_per_episode_crops_cache()

    def _resolve_emg2pose_channel_indices(self) -> np.ndarray | None:
        """Resolve the 0-based 16-position emg2pose ring indices for the 8 source channels.

        `emg2pose_channel_indices` is 0-based (range [0, 16)).
        """
        if self.emg2pose_channel_indices is None:
            return None
        indices = np.asarray(self.emg2pose_channel_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("emg2pose_channel_indices must be a 1D sequence")
        if np.any(indices < 0) or np.any(indices >= 16):
            raise ValueError(f"emg2pose_channel_indices out of range: {self.emg2pose_channel_indices}")
        return indices

    def _load_manifest(self) -> None:
        manifest_path = self.memmap_dir / "manifest.json"
        with open(manifest_path) as f:
            self._manifest = json.load(f)
        self._frame_split_available = "frame_split_id" in self._manifest["fields"]

    def _load_metadata(self) -> None:
        md = np.load(self.memmap_dir / "metadata.npz", allow_pickle=False)
        self._episode_id = _decode_bytes(md["episode_id"])
        self._episode_subject = _decode_bytes(md["episode_subject"])
        self._episode_subject_id = md["episode_subject_id"]
        self._episode_source_parquet = _decode_bytes(md["episode_source_parquet"])
        self._episode_zed_video_path = _decode_bytes(md["episode_zed_video_path"])
        self._episode_head_video_path = _decode_bytes(md["episode_head_video_path"])
        self._episode_wrist_left_video_path = _decode_bytes(
            md["episode_wrist_left_video_path"]
        ) if "episode_wrist_left_video_path" in md else [""] * len(self._episode_id)
        self._episode_wrist_right_video_path = _decode_bytes(
            md["episode_wrist_right_video_path"]
        ) if "episode_wrist_right_video_path" in md else [""] * len(self._episode_id)
        self._episode_start_idx = md["episode_start_idx"]
        self._episode_end_idx = md["episode_end_idx"]
        self._episode_length = md["episode_length"]
        self._episode_beta_idx = md["episode_beta_idx"]
        self._subjects = _decode_bytes(md["subjects_subject"])
        self._splits = _decode_bytes(md["splits_split"])
        split_ids = md.get("episode_split_id")
        if split_ids is None and not self._frame_split_available:
            if len(self._splits) == 1 and self._splits[0] == "all":
                # Deterministic initial split for EgoEMG v2 memmap when builder
                # has not yet materialized dedicated split ids.
                n = len(self._episode_id)
                split_ids = np.zeros((n,), dtype=np.int32)
                if n >= 9:
                    split_ids[max(0, n - 9) : max(0, n - 5)] = 1  # val
                    split_ids[max(0, n - 5) :] = 2  # test
                self._splits = ["train", "val", "test"]
            else:
                split_ids = np.zeros((len(self._episode_id),), dtype=np.int32)
        if split_ids is None:
            split_ids = np.zeros((len(self._episode_id),), dtype=np.int32)
        self._episode_split_id = np.asarray(split_ids, dtype=np.int32)

        # Per-frame source provenance, used to restore a per-sample dataset_name
        # when multiple datasets are physically merged into one memmap (see
        # scripts/data/merge_datasets_to_unified_memmap.py).  When absent the
        # dataset falls back to the instance-level ``dataset_name``.
        ds_sources = self._manifest.get("dataset_sources") if self._manifest else None
        if ds_sources:
            self._source_id_to_name = {int(k): str(v) for k, v in ds_sources.items()}
        else:
            self._source_id_to_name = None

    def _resolve_allowed_split_ids(self) -> set[int]:
        if not self.allowed_splits:
            return set()
        aliases: dict[str, tuple[str, ...]] = {
            "val": ("user", "gesture", "both"),
            "test": ("user", "gesture", "both"),
            "holdout": ("user", "gesture", "both"),
        }
        available = set(self._splits)
        expanded: list[str] = []
        for name in self.allowed_splits:
            if name == "all":
                expanded.extend(self._splits)
                continue
            # If the literal name is in the available splits, use it directly
            # (e.g. simple train/val/test datasets). Otherwise expand aliases
            # for EgoEMG-style cross-user/cross-gesture splits.
            if name in available:
                expanded.append(name)
            else:
                expanded.extend(aliases.get(name, (name,)))
        return _resolve_str_filter(self._splits, expanded, "split")

    def _load_norm_stats(self) -> None:
        if self.norm_mode != "per-dataset" or not self.norm_stats_path:
            return
        with open(self.norm_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Field-aware lookup: prefer <dataset>__<emg_field> so that stats are
        # matched to the actual EMG variant being loaded. Falls back to the
        # legacy bare <dataset> key for back-compat with older stats files.
        field_key = f"{self.dataset_name}__{self.emg_field_preference}"
        item = stats.get(field_key)
        if item is None or "mean" not in item or "std" not in item:
            # Back-compat: legacy single-key form (often a raw-EMG key being
            # applied to filtered EMG, which silently produces wrong results).
            item = stats.get(self.dataset_name)
            if item is None or "mean" not in item or "std" not in item:
                raise KeyError(
                    f"Dataset normalization stats for '{self.dataset_name}' "
                    f"(emg_field={self.emg_field_preference!r}) not found in "
                    f"{self.norm_stats_path}. Tried keys '{field_key}' and "
                    f"'{self.dataset_name}'."
                )
            log.warning(
                "norm stats: using legacy key '%s' for emg_field=%s. "
                "Prefer adding '%s' to the stats file so the stats match "
                "the EMG variant being loaded.",
                self.dataset_name, self.emg_field_preference, field_key,
            )

        # Default: per-channel + per-hand normalization. Prefer the
        # `<dataset>__<field>_<hand>` entry's per_channel_mean/std vectors
        # (length-8), so each channel is scaled to a comparable range and
        # the left/right hands — which have very different std profiles — are
        # normalized independently. This is the default for the target_hand
        # 8-channel layout. The 16-channel interpolate/sparse layouts fall
        # back to the scalar stats because the per-channel vectors are 8-dim
        # and would not broadcast against the (T, 16) EMG tensor.
        self._emg_mean = float(item["mean"])
        self._emg_std = float(item["std"])
        used_per_channel = False
        if (
            self.emg_layout == "target_hand"
            and self.target_hand in {"left", "right"}
        ):
            hand_key = f"{self.dataset_name}__{self.emg_field_preference}_{self.target_hand}"
            hand_item = stats.get(hand_key)
            pcm = hand_item.get("per_channel_mean") if hand_item else None
            pcs = hand_item.get("per_channel_std") if hand_item else None
            if (
                pcm is not None
                and pcs is not None
                and len(pcm) == len(pcs) == 8
            ):
                self._emg_mean = np.asarray(pcm, dtype=np.float32)
                self._emg_std = np.asarray(pcs, dtype=np.float32)
                used_per_channel = True
                log.info(
                    "norm stats: per-channel + per-hand (%s) for dataset=%s "
                    "emg_field=%s. mean=%s std=%s",
                    self.target_hand, self.dataset_name,
                    self.emg_field_preference,
                    np.round(self._emg_mean, 4).tolist(),
                    np.round(self._emg_std, 4).tolist(),
                )
            else:
                log.warning(
                    "norm stats: per-hand key '%s' missing or malformed "
                    "(need per_channel_mean/std of length 8). Falling back to "
                    "scalar normalization for dataset=%s emg_field=%s.",
                    hand_key, self.dataset_name, self.emg_field_preference,
                )
        elif self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"}:
            log.info(
                "norm stats: using scalar (mean, std) for layout=%s (per-channel "
                "vectors are 8-dim and do not broadcast against 16-channel EMG).",
                self.emg_layout,
            )

        # Sanity check: stats for raw EMG typically have std >= 20 even after
        # the 8->16 channel layout (zero-fill halves the mean-of-squares but
        # not the std of the live channels). Filtered EMG (notch+bandpass) is
        # usually < 20. Warn loudly so future mis-matches surface immediately.
        # For per-channel vectors, flag if the mean std across channels is high.
        std_for_check = (
            float(np.mean(self._emg_std)) if used_per_channel else self._emg_std
        )
        if std_for_check >= 20.0 and self.emg_field_preference not in {"raw"}:
            log.warning(
                "norm stats for dataset=%s emg_field=%s have std=%.2f, which "
                "looks like raw-EMG statistics. If you intended to normalize "
                "filtered EMG with raw-EMG stats, predictions will be "
                "compressed into noise (root cause of the WL14638 "
                "reproduction issue, see docs/egoemg_filtered_repro_issue.md). "
                "Add a field-aware key 'egoemg__%s' to the stats file.",
                self.dataset_name, self.emg_field_preference, std_for_check,
                self.emg_field_preference,
            )

    @staticmethod
    def _prune_hand_fields(
        fields: set[str],
        *,
        target_hand: str,
        emg_field_preference: str,
        emg_layout: str,
    ) -> set[str]:
        """Remove fields for the non-target hand and unused EMG variants."""
        # Fields that __getitem__ will actually access for the target hand
        needed = {
            f"emg_{target_hand}_{emg_field_preference}",
            f"generated_joint_angles_{target_hand}",
            f"generated_mano_{target_hand}_pose",
            f"generated_mano_{target_hand}_beta",
            f"mocap_{target_hand}_keypoints",
            f"mocap_{target_hand}_valid",
            f"mocap_{target_hand}_wrist_pitch",
            f"mocap_{target_hand}_wrist_yaw",
            f"mocap_{target_hand}_wrist_angles_valid",
        }
        # Stream frame indices are per-view, not per-target-hand data:
        # keep both wrist views even when only one hand is supervised.
        needed |= {
            "image_wrist_left_frame_index", "image_wrist_left_stale",
            "image_wrist_left_delta_ms", "image_wrist_right_frame_index",
            "image_wrist_right_stale", "image_wrist_right_delta_ms",
        }

        # Remove hand-tagged fields not in the needed set.
        # Hand-tagged = field name contains _left or _right (may be
        # mid-field, e.g. emg_left_filtered, generated_joint_angles_right).
        pruned = set(fields)
        for f in fields:
            if ("_left" in f or "_right" in f) and f not in needed:
                pruned.discard(f)
        return pruned

    def _resolve_requested_fields(self) -> tuple[set[str], set[str]]:
        assert self._manifest is not None
        frame_fields = set(self._manifest["fields"].keys())
        episode_fields = set(self._manifest["episode_fields"].keys())

        if self.modalities is None and self.fields is None:
            return frame_fields, episode_fields

        wanted: set[str] = set()
        explicit_fields: set[str] = set()
        for name in self.modalities or []:
            if name not in MODALITY_GROUPS:
                raise ValueError(f"Unknown modality group: {name}")
            wanted.update(MODALITY_GROUPS[name])
        for f in self.fields or []:
            wanted.add(f)
            explicit_fields.add(f)

        available = frame_fields | episode_fields
        missing = wanted - available
        if missing:
            # Fields from MODALITY_GROUPS are best-effort: silently skip any
            # that the manifest does not contain (e.g. single-hand datasets
            # omit the other hand's EMG/joint_angles, some datasets lack
            # gesture labels).  Only fields explicitly requested via
            # ``self.fields`` are strictly required.
            missing_explicit = missing & explicit_fields
            if missing_explicit:
                raise ValueError(f"Unknown memmap field(s): {sorted(missing_explicit)}")
            wanted -= missing

        requested_frame = wanted & frame_fields
        requested_episode = wanted & episode_fields
        return requested_frame, requested_episode

    def _load_vision_calibration(self) -> None:
        if not self._vision_enabled:
            self._K_calib = None
            self._dist_calib = None
            self._calib_w = None
            self._calib_h = None
            return
        if self.per_episode_crops_dir is not None:
            # Pre-crops don't need calibration
            self._K_calib = None
            self._dist_calib = None
            self._calib_w = None
            self._calib_h = None
            return
        calib_path = (
            self.video_root / "reprojection_assets" / "GX010023_standard_calibration.json"
        )
        try:
            with calib_path.open("r", encoding="utf-8") as f:
                calib = json.load(f)
        except (OSError, ValueError) as exc:
            # Missing/malformed camera calibration: fall back to identity
            # intrinsics so headless/center-frame evaluation can run without
            # the reprojection_assets calibration files.
            self._K_calib = np.eye(3, dtype=np.float64)
            self._dist_calib = np.zeros((1, 5), dtype=np.float64)
            self._calib_w = 640
            self._calib_h = 480
            print(f"Warning: calibration unavailable ({exc}); using identity intrinsics.", flush=True)
            return
        self._K_calib = np.asarray(calib["camera_matrix"], dtype=np.float64)
        self._dist_calib = np.asarray(
            calib["distortion_coefficients"], dtype=np.float64
        ).reshape(-1, 1)
        self._calib_w = int(calib["image_width"])
        self._calib_h = int(calib["image_height"])

    def _init_per_episode_crops_cache(self) -> None:
        """Validate that pre-crop LMDB files exist."""
        crops_dir = Path(self.per_episode_crops_dir)
        if not crops_dir.exists():
            raise FileNotFoundError(f"Pre-crop directory not found: {crops_dir}")
        # Verify at least one LMDB exists
        sample_lmdb = crops_dir / "episode_000000.lmdb"
        if not sample_lmdb.exists():
            raise FileNotFoundError(f"Sample LMDB not found: {sample_lmdb}")

    def _get_pec_txn(self, episode_id: str):
        """Get or open an LMDB read transaction for a per-episode crops database."""
        if os.getpid() != self._pec_pid:
            self._pec_lmdb_cache = {}
            self._pec_pid = os.getpid()

        cached = self._pec_lmdb_cache.get(episode_id)
        if cached is not None:
            return cached[2]

        if len(self._pec_lmdb_cache) >= self._pec_lmdb_max_open:
            oldest_key = next(iter(self._pec_lmdb_cache))
            old_path, old_env, _ = self._pec_lmdb_cache.pop(oldest_key)
            _release_readonly_lmdb(old_path, old_env)

        lmdb_path = Path(self.per_episode_crops_dir) / f"{episode_id}.lmdb"
        if not lmdb_path.exists():
            return None
        env = _acquire_readonly_lmdb(lmdb_path)
        txn = env.begin()
        self._pec_lmdb_cache[episode_id] = (lmdb_path, env, txn)
        return txn

    def _read_episode_crops(
        self,
        episode_id: str,
        frame_indices: np.ndarray,
        hand_code: str,
    ) -> np.ndarray | None:
        """Read pre-cropped JPEG images from per-episode LMDB.

        Args:
            episode_id: e.g. "episode_000000"
            frame_indices: array of frame indices within the episode
            hand_code: "L" or "R"

        Returns:
            (N, H, W, 3) RGB numpy array, or None if crops unavailable.
        """
        txn = self._get_pec_txn(episode_id)
        if txn is None:
            return None

        import io
        from PIL import Image

        frames = []
        for fi in frame_indices:
            key = f"{fi:08d}_{hand_code}".encode()
            jpeg_bytes = txn.get(key)
            if jpeg_bytes is None:
                return None
            img = Image.open(io.BytesIO(jpeg_bytes))
            frames.append(np.asarray(img))  # (H, W, 3) RGB
        return np.stack(frames, axis=0)

    def _open_memmaps(self) -> None:
        assert self._manifest is not None
        for name, info in self._manifest["fields"].items():
            if name in self._skip_fields:
                continue
            self._frame_memmaps[name] = np.memmap(
                self.memmap_dir / info["filename"],
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=tuple(info["shape"]),
            )
        for name, info in self._manifest["episode_fields"].items():
            self._episode_memmaps[name] = np.memmap(
                self.memmap_dir / info["filename"],
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=tuple(info["shape"]),
            )

    def _load_mano_npy(self) -> None:
        """Load per-episode MANO .npy files as memmaps when mano_npy_dir is set."""
        self._mano_npy_pose: dict[tuple[int, str], np.memmap] = {}
        self._mano_npy_beta: dict[tuple[int, str], np.ndarray] = {}
        self._mano_npy_trans: dict[tuple[int, str], np.ndarray] = {}
        self._mano_npy_world_R: dict[tuple[int, str], np.memmap] = {}
        self._mano_npy_world_t: dict[tuple[int, str], np.memmap] = {}
        if self.mano_npy_dir is None:
            return
        npy_dir = Path(self.mano_npy_dir)
        if not npy_dir.is_dir():
            raise FileNotFoundError(f"mano_npy_dir not found: {npy_dir}")
        for ep_idx, ep_id in enumerate(self._episode_id):
            for hand in ("left", "right"):
                pose_path = npy_dir / f"{ep_id}_{hand}_pose.npy"
                beta_path = npy_dir / f"{ep_id}_{hand}_beta.npy"
                trans_path = npy_dir / f"{ep_id}_{hand}_trans.npy"
                if not pose_path.exists():
                    continue
                key = (ep_idx, hand)
                self._mano_npy_pose[key] = np.load(pose_path, mmap_mode="r")
                self._mano_npy_beta[key] = np.load(beta_path)
                if trans_path.exists():
                    self._mano_npy_trans[key] = np.load(trans_path)
                world_R_path = npy_dir / f"{ep_id}_{hand}_world_R.npy"
                world_t_path = npy_dir / f"{ep_id}_{hand}_world_t.npy"
                if world_R_path.exists():
                    self._mano_npy_world_R[key] = np.load(world_R_path, mmap_mode="r")
                if world_t_path.exists():
                    self._mano_npy_world_t[key] = np.load(world_t_path, mmap_mode="r")

    def _filter_episode_indices(self) -> np.ndarray:
        n = len(self._episode_id)
        mask = np.ones((n,), dtype=bool)

        if self.allowed_episode_ids:
            allowed = set(self.allowed_episode_ids)
            mask &= np.asarray([eid in allowed for eid in self._episode_id], dtype=bool)

        if self.allowed_subjects:
            allowed_subject_ids = _resolve_str_filter(self._subjects, self.allowed_subjects, "subject")
            mask &= np.isin(self._episode_subject_id, list(allowed_subject_ids))

        if self.allowed_splits and not self._frame_split_available:
            allowed_split_ids = self._resolve_allowed_split_ids()
            mask &= np.isin(self._episode_split_id, list(allowed_split_ids))

        if self.excluded_episode_ids:
            mask &= ~np.isin(self._episode_id, list(self.excluded_episode_ids))

        return np.nonzero(mask)[0].astype(np.int64)

    def _build_window_index(self) -> None:
        allowed = self._filter_episode_indices()
        episode_idx: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        num_per_episode: list[int] = []
        window_split_ids: list[np.ndarray] = []
        allowed_split_ids = self._resolve_allowed_split_ids()
        split_mm = self._frame_memmaps.get("frame_split_id")

        for ep_idx in allowed.tolist():
            start = int(self._episode_start_idx[ep_idx])
            end = int(self._episode_end_idx[ep_idx])

            # If eval_center_stride is set, build the center-frame grid from it
            # (independent of window_length), so models with different WLs are
            # evaluated at identical center positions.
            if self.eval_center_stride is not None:
                ecs = self.eval_center_stride
                half_wl = self.window_length // 2
                # center grid: start + ecs//2, start + ecs//2 + ecs, ...
                # window must fit: center - half_wl >= start AND center + half_wl <= end
                # Keep the center positions on the shared ``ecs`` grid while
                # advancing past any centers whose model-specific window
                # would cross the episode boundary.
                grid_offset = ecs // 2
                min_offset = half_wl
                first_grid_step = max(
                    0,
                    (min_offset - grid_offset + ecs - 1) // ecs,
                )
                first_center = start + grid_offset + first_grid_step * ecs
                last_center = end - half_wl
                if last_center < first_center:
                    continue
                center_positions = np.arange(
                    first_center, last_center + 1, ecs, dtype=np.int64
                )
                window_starts = center_positions - half_wl
                n = len(center_positions)
                if split_mm is not None and self.allowed_splits:
                    center_split_ids = np.asarray(
                        split_mm[center_positions], dtype=np.int32
                    )
                    keep = np.isin(center_split_ids, list(allowed_split_ids))
                    if not np.any(keep):
                        continue
                    window_starts = window_starts[keep]
                    episode_idx.extend([ep_idx] * int(window_starts.shape[0]))
                    starts.extend(window_starts.tolist())
                    ends.extend((window_starts + self.window_length).tolist())
                    num_per_episode.append(int(window_starts.shape[0]))
                    window_split_ids.append(center_split_ids[keep])
                    continue
                episode_idx.extend([ep_idx] * n)
                starts.extend(window_starts.tolist())
                ends.extend((window_starts + self.window_length).tolist())
                num_per_episode.append(n)
                continue

            n = (end - start - self.window_length) // self.stride + 1
            if n <= 0:
                continue
            if split_mm is not None and self.allowed_splits:
                start_positions = start + np.arange(n, dtype=np.int64) * self.stride
                center_positions = start_positions + self.window_length // 2
                center_split_ids = np.asarray(split_mm[center_positions], dtype=np.int32)
                keep = np.isin(center_split_ids, list(allowed_split_ids))
                if not np.any(keep):
                    continue
                window_starts = start_positions[keep]
                episode_idx.extend([ep_idx] * int(window_starts.shape[0]))
                starts.extend(window_starts.tolist())
                ends.extend((window_starts + self.window_length).tolist())
                num_per_episode.append(int(window_starts.shape[0]))
                window_split_ids.append(center_split_ids[keep])
                continue
            episode_idx.append(ep_idx)
            starts.append(start)
            ends.append(end)
            num_per_episode.append(int(n))

        self._block_episode_idx = np.asarray(episode_idx, dtype=np.int32)
        if split_mm is not None and self.allowed_splits:
            self._block_start = np.asarray(starts, dtype=np.int64)
            self._block_end = np.asarray(ends, dtype=np.int64)
            self._block_cumsum = np.arange(len(self._block_start) + 1, dtype=np.int64)
            self._window_split_id = (
                np.concatenate(window_split_ids).astype(np.int32)
                if window_split_ids
                else np.zeros((0,), dtype=np.int32)
            )
        else:
            self._block_start = np.asarray(starts, dtype=np.int64)
            self._block_end = np.asarray(ends, dtype=np.int64)
            self._block_cumsum = np.cumsum(np.asarray([0] + num_per_episode, dtype=np.int64))
            self._window_split_id = None

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def _resolve_video_path(self, raw_video_path: str) -> str:
        return str(
            resolve_allintra_video_path(
                raw_video_path=raw_video_path,
                data_root=self.video_root,
                allintra_root=self.allintra_root,
                suffix=self.allintra_suffix,
            )
        )

    def _decode_video_frames(
        self,
        video_path: str,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        return self._video_reader_cache.read_many_rgb(video_path, frame_indices)

    def reset_runtime_caches(self) -> None:
        self._video_reader_cache = DecordVideoReaderCache()
        self._intrinsics_cache.clear()
        self._runtime_pid = os.getpid()

    def _ensure_runtime_owner(self) -> None:
        if self._runtime_pid != os.getpid():
            self.reset_runtime_caches()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_frame_memmaps"] = {}
        state["_episode_memmaps"] = {}
        state["_video_reader_cache"] = None
        state["_intrinsics_cache"] = {}
        state.pop("_cvf_lmdb_cache", None)
        # Keep preloaded feature arrays in the pickle — numpy C-contiguous
        # buffers serialize efficiently.  Fork-based workers (Linux) inherit
        # them via COW anyway; spawn-based workers (Windows) receive them
        # without needing to re-open LMDB in the child, which avoids SIGABRT.
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_memmaps()
        self._video_reader_cache = DecordVideoReaderCache()
        self._intrinsics_cache = {}
        self._runtime_pid = os.getpid()
        if self._cached_vit_enabled:
            self._cvf_lmdb_cache = {}
            self._cvf_pid = os.getpid()

    def _convert_to_emg2pose_layout(
        self,
        emg: np.ndarray,
        *,
        interpolate_missing: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._emg2pose_channel_indices is None:
            raise ValueError(
                "emg2pose alignment requires emg2pose_channel_indices to be provided"
            )
        aligned = emg
        target_positions = self._emg2pose_channel_indices
        if aligned.shape[1] != len(target_positions):
            aligned = circular_interpolate(aligned, len(target_positions))
        if interpolate_missing:
            placed = sparse_ring_interpolate(aligned, 16, target_positions)
        else:
            placed = place_sparse_channels(aligned, 16, target_positions)
        mask = np.zeros((16,), dtype=np.bool_)
        mask[target_positions] = True
        return placed, mask

    def _vision_frame_local_indices(self, length: int) -> np.ndarray:
        center = length // 2
        if self.vision_num_frames <= 1:
            return np.array([center], dtype=np.int64)
        offsets = np.arange(self.vision_num_frames, dtype=np.int64) - (
            self.vision_num_frames // 2
        )
        indices = center + offsets * int(self.vision_frame_stride)
        return np.clip(indices, 0, length - 1).astype(np.int64)

    def _vision_intrinsics(
        self,
        first_frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        video_h, video_w = first_frame_bgr.shape[:2]
        key = (video_h, video_w)
        cached = self._intrinsics_cache.get(key)
        if cached is not None:
            return cached
        intrinsics = _build_intrinsics_and_frame_mapper(
            self._K_calib,
            self._dist_calib,
            self._calib_w,
            self._calib_h,
            video_w,
            video_h,
            first_frame_bgr,
        )
        self._intrinsics_cache[key] = intrinsics
        return intrinsics

    def _build_vision_sample(
        self,
        result: dict[str, Any],
        hand: str,
        hand_idx: int,
        dataset_idx: int,
    ) -> dict[str, Any]:
        # Cached ViT features take priority over raw image loading
        if self._cached_vit_enabled:
            if self._preloaded_features is not None:
                feature = self._preloaded_features[dataset_idx]
                valid = bool(self._preloaded_features_valid[dataset_idx])
            else:
                ep_id = result["episode_id"]
                # Fast path: center value already read directly from memmap
                if "_image_head_frame_index_center" in result:
                    center_frame_idx = result["_image_head_frame_index_center"]
                else:
                    center_frame_idx = int(result["image_head_frame_index"][result["window_length"] // 2])
                hand_code = "L" if hand == "left" else "R"
                feature, valid = self._get_cached_vit_lmdb(ep_id, center_frame_idx, hand_code)
            return {
                "vision_features": feature,
                "vision_valid_mask": np.asarray([valid], dtype=bool),
            }

        if not self._vision_enabled:
            return {}

        local_indices = self._vision_frame_local_indices(result["window_length"])
        video_frame_indices = np.asarray(
            result["image_head_frame_index"], dtype=np.int64
        )[local_indices]
        episode_id = result["episode_id"]
        hand_code = "L" if hand == "left" else "R"

        # Read from pre-crops if available, otherwise decode from video
        if self.per_episode_crops_dir is not None:
            frames_rgb = self._read_episode_crops(episode_id, video_frame_indices, hand_code)
            crops_valid = frames_rgb is not None
            if frames_rgb is None:
                # Crops not available - return placeholder so batch can collate,
                # but mark every frame INVALID so losses/eval exclude it (the
                # same contract as the vision-only and center-supervised fast
                # paths).
                frames_rgb = np.zeros(
                    (len(video_frame_indices), self.vision_patch_size, self.vision_patch_size, 3),
                    dtype=np.uint8,
                )
            # Pre-crops are already 256x256 RGB patches. Just normalize and return.
            patches: list[np.ndarray] = []
            for frame_rgb in frames_rgb:
                img_patch = frame_rgb.transpose(2, 0, 1).astype(np.float32)
                for channel_idx in range(3):
                    img_patch[channel_idx] = (
                        img_patch[channel_idx] - self.vision_mean[channel_idx]
                    ) / self.vision_std[channel_idx]
                patches.append(img_patch)
            vision_img = np.stack(patches, axis=0)
            if vision_img.shape[0] == 1:
                vision_img = vision_img[0]
            return {
                "vision_img": vision_img.astype(np.float32),
                "vision_valid_mask": np.asarray([crops_valid] * len(frames_rgb), dtype=bool),
                "vision_frame_indices": np.asarray(video_frame_indices, dtype=np.int64),
            }

        video_path = self._resolve_video_path(result["episode_head_video_path"])
        frames_rgb = self._decode_video_frames(video_path, video_frame_indices)
        frames_bgr = frames_rgb[:, :, :, ::-1].copy()
        K_use, dist_use, intrinsics_info = self._vision_intrinsics(frames_bgr[0])

        patches: list[np.ndarray] = []
        valid_mask: list[bool] = []
        box_centers: list[np.ndarray] = []
        box_sizes: list[np.float32] = []
        is_left = hand == "left"

        marker_world_seq = np.asarray(result[f"mocap_{hand}_keypoints"], dtype=np.float64)
        marker_valid_seq = np.asarray(result[f"mocap_{hand}_valid"], dtype=bool)
        cam_transform_seq = np.asarray(result["mocap_head_transform"], dtype=np.float64)
        tracked_seq = np.asarray(result["mocap_head_valid"], dtype=bool)
        label_valid_seq = np.asarray(result["generated_label_valid"], dtype=bool)[:, hand_idx]

        for frame_bgr, local_idx, video_frame_idx in zip(
            frames_bgr, local_indices, video_frame_indices
        ):
            video_h, video_w = frame_bgr.shape[:2]
            T_W_C = np.eye(4, dtype=np.float64)
            t12_cam = cam_transform_seq[local_idx]
            T_W_C[:3, :3] = t12_cam[:9].reshape(3, 3)
            T_W_C[:3, 3] = t12_cam[9:12]

            marker_world = marker_world_seq[local_idx]
            marker_valid = marker_valid_seq[local_idx]
            label_valid = bool(label_valid_seq[local_idx]) and bool(tracked_seq[local_idx])
            marker_proc, marker_depth_valid = _project_world_points(
                marker_world,
                T_W_C,
                K_use,
                dist_use,
            )
            marker_raw = _map_processed_points_to_raw(marker_proc, intrinsics_info)
            marker_in_image = (
                (marker_raw[:, 0] >= 0)
                & (marker_raw[:, 0] < video_w)
                & (marker_raw[:, 1] >= 0)
                & (marker_raw[:, 1] < video_h)
            )
            markers_2d_xy = marker_raw.astype(np.float32)
            if is_left:
                frame_bgr = np.ascontiguousarray(frame_bgr[:, ::-1])
                markers_2d_xy[:, 0] = (video_w - 1) - markers_2d_xy[:, 0]
            markers_conf = (
                marker_valid & marker_depth_valid & marker_in_image
            ).astype(np.float32)
            if not label_valid:
                markers_conf[:] = 0.0
            keypoints_2d = np.concatenate([markers_2d_xy, markers_conf[:, None]], axis=-1)
            valid_for_bbox = keypoints_2d[:, 2] > 0
            if valid_for_bbox.sum() < 2:
                center = np.array([video_w / 2.0, video_h / 2.0], dtype=np.float32)
                scale = np.array([video_w / 3.0, video_h / 3.0], dtype=np.float32)
                valid = False
            else:
                center, scale = _get_bbox(keypoints_2d, rescale=1.2)
                valid = True
            scale = _expand_to_aspect_ratio(scale, (192, 256))
            bbox_size = float(max(scale[0], scale[1]))
            img_patch_bgr, _ = _generate_image_patch_cv2(
                frame_bgr,
                float(center[0]),
                float(center[1]),
                bbox_size,
                bbox_size,
                self.vision_patch_size,
                self.vision_patch_size,
                False,
                1.0,
                0.0,
            )
            img_patch_rgb = img_patch_bgr[:, :, ::-1]
            img_patch = _convert_cvimg_to_tensor(img_patch_rgb)
            for channel_idx in range(3):
                img_patch[channel_idx] = (
                    img_patch[channel_idx] - self.vision_mean[channel_idx]
                ) / self.vision_std[channel_idx]
            patches.append(img_patch.astype(np.float32))
            valid_mask.append(valid)
            box_centers.append(center.astype(np.float32))
            box_sizes.append(np.float32(bbox_size))

        vision_img = np.stack(patches, axis=0)
        if vision_img.shape[0] == 1:
            vision_img = vision_img[0]
        return {
            "vision_img": vision_img.astype(np.float32),
            "vision_valid_mask": np.asarray(valid_mask, dtype=bool),
            "vision_frame_indices": np.asarray(video_frame_indices, dtype=np.int64),
            "vision_box_center": np.stack(box_centers, axis=0).astype(np.float32),
            "vision_box_size": np.asarray(box_sizes, dtype=np.float32),
            "vision_raw_right": np.float32(0.0 if is_left else 1.0),
            "vision_video_path": video_path,
        }

    # ── Cached ViT features ───────────────────────────────────────────────

    def _init_cached_vit_features(self) -> None:
        """Open per-episode ViT feature LMDBs on demand.

        Feature LMDBs use the same key scheme as crops LMDBs:
        {frame_idx:08d}_{L/R} -> 1280 float32 bytes.
        No manifest, no split directories, no index arrays needed.
        """
        self._cvf_dir = Path(self.cached_vit_features_dir)
        self._cvf_lmdb_cache: dict[str, tuple] = {}
        self._cvf_lmdb_max_open = 8
        self._cvf_pid = os.getpid()
        self._cached_vit_feat_dim = 1280

    def _get_cvf_txn(self, episode_id: str):
        """Get or open an LMDB read transaction for a per-episode feature database."""
        import lmdb as _lmdb

        if os.getpid() != self._cvf_pid:
            self._cvf_lmdb_cache = {}
            self._cvf_pid = os.getpid()

        cached = self._cvf_lmdb_cache.get(episode_id)
        if cached is not None:
            return cached[1]

        if len(self._cvf_lmdb_cache) >= self._cvf_lmdb_max_open:
            oldest_key = next(iter(self._cvf_lmdb_cache))
            old_env, _ = self._cvf_lmdb_cache.pop(oldest_key)
            old_env.close()

        lmdb_path = self._cvf_dir / f"{episode_id}.lmdb"
        if not lmdb_path.exists():
            return None
        env = _lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
        txn = env.begin()
        self._cvf_lmdb_cache[episode_id] = (env, txn)
        return txn

    @staticmethod
    def _read_center_ja(
        center: int,
        ja_mm: np.memmap | None,
        pitch_mm: np.memmap | None,
        yaw_mm: np.memmap | None,
    ) -> np.ndarray:
        """Read center-frame 22-dim joint angles: 20 MANO joints + pitch + yaw."""
        ja = ja_mm[center] if ja_mm is not None else np.zeros(20, dtype=np.float32)
        pitch = np.deg2rad(float(pitch_mm[center])) if pitch_mm is not None else 0.0
        yaw = np.deg2rad(float(yaw_mm[center])) if yaw_mm is not None else 0.0
        out = np.empty(22, dtype=np.float32)
        out[:20] = ja
        out[20] = pitch
        out[21] = yaw
        return out

    @staticmethod
    def _read_center_lv(
        center: int, lv_mm: np.memmap | None, hand_idx: int
    ) -> bool:
        """Read center-frame label_valid for the target hand."""
        if lv_mm is None:
            return True
        return bool(lv_mm[center][hand_idx])

    def _preload_all_features(self) -> None:
        """Iterate through all windows and load every ViT feature + center-frame
        joint angles into RAM.

        Stores features ``(N, feat_dim)``, joint_angles ``(N, 22)``, and two
        ``(N,)`` bool arrays (feature_valid, label_valid), all indexed by
        dataset position.  Plain numpy arrays avoid COW page-copying from
        Python-object refcounts inside dicts/tuples.
        """
        from tqdm import tqdm

        wc_fi_mm = self._frame_memmaps["image_head_frame_index"]
        hand = self.target_hand
        hand_code = "L" if hand == "left" else "R"
        hand_idx = 0 if hand == "left" else 1
        n = len(self)
        if n == 0:
            return

        total_windows = sum(
            1 if self._window_split_id is not None
            else self._block_cumsum[bi + 1] - self._block_cumsum[bi]
            for bi in range(len(self._block_episode_idx))
        )

        feat_dim = self._cached_vit_feat_dim
        features = np.empty((total_windows, feat_dim), dtype=np.float32)
        valid_mask = np.empty(total_windows, dtype=bool)
        joint_angles_all = np.empty((total_windows, 22), dtype=np.float32)
        label_valid_all = np.empty(total_windows, dtype=bool)
        pos = 0

        # Memmap handles for joint-angle fields
        ja_mm = self._frame_memmaps.get(f"generated_joint_angles_{hand}")
        pitch_mm = self._frame_memmaps.get(f"mocap_{hand}_wrist_pitch")
        yaw_mm = self._frame_memmaps.get(f"mocap_{hand}_wrist_yaw")
        lv_mm = self._frame_memmaps.get("generated_label_valid")

        pbar = tqdm(total=total_windows, desc="Preloading ViT features + joint angles", unit="win")

        for bi in range(len(self._block_episode_idx)):
            ep_idx = int(self._block_episode_idx[bi])
            ep_id = self._episode_id[ep_idx]
            block_start = int(self._block_start[bi])

            if self._window_split_id is not None:
                center = block_start + self.window_length // 2
                frame_idx = int(wc_fi_mm[center])
                feat, valid = self._get_cached_vit_lmdb(ep_id, frame_idx, hand_code)
                features[pos] = feat
                valid_mask[pos] = valid
                joint_angles_all[pos] = self._read_center_ja(
                    center, ja_mm, pitch_mm, yaw_mm
                )
                label_valid_all[pos] = self._read_center_lv(center, lv_mm, hand_idx)
                pos += 1
                pbar.update(1)
            else:
                n_windows = self._block_cumsum[bi + 1] - self._block_cumsum[bi]
                for rel in range(n_windows):
                    start = block_start + rel * self.stride
                    center = start + self.window_length // 2
                    frame_idx = int(wc_fi_mm[center])
                    feat, valid = self._get_cached_vit_lmdb(ep_id, frame_idx, hand_code)
                    features[pos] = feat
                    valid_mask[pos] = valid
                    joint_angles_all[pos] = self._read_center_ja(
                        center, ja_mm, pitch_mm, yaw_mm
                    )
                    label_valid_all[pos] = self._read_center_lv(center, lv_mm, hand_idx)
                    pos += 1
                    pbar.update(1)

        pbar.close()
        self._preloaded_features = features
        self._preloaded_features_valid = valid_mask
        self._preloaded_joint_angles = joint_angles_all
        self._preloaded_label_valid = label_valid_all

        try:
            for (env, _txn) in list(self._cvf_lmdb_cache.values()):
                try:
                    env.close()
                except Exception:
                    pass
        finally:
            self._cvf_lmdb_cache.clear()

    def _get_cached_vit_lmdb(self, ep_id: str, frame_idx: int, hand_code: str) -> tuple[np.ndarray, bool]:
        """Direct LMDB lookup, bypassing the in-memory cache. Used during preloading."""
        txn = self._get_cvf_txn(ep_id)
        if txn is None:
            return np.zeros((self._cached_vit_feat_dim,), dtype=np.float32), False
        lmdb_key = f"{frame_idx:08d}_{hand_code}".encode()
        feature_bytes = txn.get(lmdb_key)
        if feature_bytes is None:
            return np.zeros((self._cached_vit_feat_dim,), dtype=np.float32), False
        return np.frombuffer(feature_bytes, dtype=np.float32).copy(), True

    def _getitem_vision_only_preloaded(
        self, idx: int, center_idx: int
    ) -> dict[str, Any]:
        """Fast path: vision_only + preloaded features.

        Reads exclusively from preloaded numpy arrays — zero file I/O.
        ``center_idx`` is used only for ``_resolve_dataset_name`` (dataset
        provenance); joint angles were preloaded at the same center frame
        during ``_preload_all_features``.
        """
        hand = self.target_hand
        hand_idx = 0 if hand == "left" else 1

        # All data from preloaded arrays — no memmap access, no .copy()
        feature = self._preloaded_features[idx]  # view, no copy
        feature_valid = bool(self._preloaded_features_valid[idx])

        # joint_angles preloaded at shape (N, 22) → reshape to (22, 1)
        ja_full = self._preloaded_joint_angles[idx]  # (22,) view
        label_valid = bool(self._preloaded_label_valid[idx])

        n_channels = 16 if self.emg_layout in {
            "emg2pose_sparse16", "emg2pose_interpolate16"
        } else 8

        dataset_name = self._resolve_dataset_name(center_idx)
        return {
            "vision_features": feature,
            "vision_valid_mask": np.array([feature_valid], dtype=bool),
            "joint_angles": ja_full[:, None],  # (22, 1) – center frame only
            "label_valid_mask": np.array([label_valid], dtype=bool),
            "target_hand": hand,
            "target_hand_index": hand_idx,
            "dataset_name": dataset_name,
            "emg": np.zeros((n_channels, 0), dtype=np.float32),
            "emg_channel_mask": np.ones((n_channels,), dtype=bool),
        }

    def _getitem_vision_only_crops(
        self, ep_idx: int, center_idx: int
    ) -> dict[str, Any]:
        """Fast path: vision_only + per-episode pre-crops.

        Does point reads on frame memmaps (no full-window slicing) and skips
        MANO .npy, episode metadata strings, and unused label fields.
        """
        hand = self.target_hand
        hand_idx = 0 if hand == "left" else 1
        hand_code = "L" if hand == "left" else "R"

        # ── Point-read center frame index from memmap (8 bytes) ─────────
        fi_mm = self._frame_memmaps["image_head_frame_index"]
        video_frame_idx = int(fi_mm[center_idx])

        # ── Point-read joint angles at center frame (20 float32 = 80 bytes) ─
        ja_key = f"generated_joint_angles_{hand}"
        ja = np.asarray(self._frame_memmaps[ja_key][center_idx], dtype=np.float32)

        # ── Point-read wrist pitch/yaw, deg → rad ───────────────────────
        pitch_key = f"mocap_{hand}_wrist_pitch"
        if pitch_key in self._frame_memmaps:
            pitch = np.deg2rad(float(self._frame_memmaps[pitch_key][center_idx]))
            yaw = np.deg2rad(
                float(self._frame_memmaps[f"mocap_{hand}_wrist_yaw"][center_idx])
            )
            ja_full = np.empty(22, dtype=np.float32)
            ja_full[:20] = ja
            ja_full[20] = pitch
            ja_full[21] = yaw
        else:
            ja_full = ja  # keep as 20-dim if no wrist fields

        # ── Point-read label_valid ──────────────────────────────────────
        lv = self._frame_memmaps["generated_label_valid"][center_idx]
        label_valid = bool(lv[hand_idx])

        # ── Read pre-crop from per-episode LMDB ─────────────────────────
        ep_id = self._episode_id[ep_idx]
        frames_rgb = self._read_episode_crops(
            ep_id, np.array([video_frame_idx], dtype=np.int64), hand_code
        )
        if frames_rgb is None:
            # Missing pre-crop: hand back a zero patch for shape consistency
            # but mark the vision sample INVALID so downstream losses/eval
            # can exclude it instead of silently training on black images.
            frames_rgb = np.zeros(
                (1, self.vision_patch_size, self.vision_patch_size, 3),
                dtype=np.uint8,
            )
            vision_ok = False
        else:
            vision_ok = True

        # ── Normalize single image ──────────────────────────────────────
        frame_rgb = frames_rgb[0]
        img_patch = frame_rgb.transpose(2, 0, 1).astype(np.float32)
        for c in range(3):
            img_patch[c] = (img_patch[c] - self.vision_mean[c]) / self.vision_std[c]

        n_channels = 16 if self.emg_layout in {
            "emg2pose_sparse16", "emg2pose_interpolate16"
        } else 8

        dataset_name = self._resolve_dataset_name(center_idx)
        return {
            "vision_img": img_patch,  # (3, 256, 256)
            "vision_valid_mask": np.array([vision_ok], dtype=bool),
            "vision_frame_indices": np.array([video_frame_idx], dtype=np.int64),
            "joint_angles": ja_full[:, None],  # (22, 1)
            "label_valid_mask": np.array([label_valid], dtype=bool),
            "target_hand": hand,
            "target_hand_index": hand_idx,
            "dataset_name": dataset_name,
            "emg": np.zeros((n_channels, 0), dtype=np.float32),
            "emg_channel_mask": np.ones((n_channels,), dtype=bool),
        }

    def _getitem_center_supervised(
        self, ep_idx: int, start: int, end: int, center_idx: int
    ) -> dict[str, Any]:
        """Fast path: center_supervised fusion mode.

        Full-window EMG read (needed for temporal featurizer) + point-reads
        for joint_angles, label_valid, wrist angles at center frame.
        Skips MANO .npy, episode metadata strings, and unused mocap fields.
        """
        hand = self.target_hand
        hand_idx = 0 if hand == "left" else 1
        hand_code = "L" if hand == "left" else "R"

        # ── Full-window EMG read ───────────────────────────────────────
        emg_key = f"emg_{hand}_{self.emg_field_preference}"
        emg_raw = np.asarray(
            self._frame_memmaps[emg_key][start:end], dtype=np.float32
        )  # (T, 8)
        if self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"}:
            emg, emg_channel_mask = self._convert_to_emg2pose_layout(
                emg_raw, interpolate_missing=self.channel_interpolate,
            )  # (T, 16)
        else:
            # target_hand layout: keep 8ch as-is
            emg = emg_raw
            emg_channel_mask = np.ones((emg.shape[1],), dtype=np.bool_)

        if self.norm_mode == "per-dataset":
            emg = (emg - self._emg_mean) / (self._emg_std + 1e-6)
        if self.transform is not None:
            emg = self.transform({"emg": emg})
            if isinstance(emg, dict):
                emg = emg.get("emg", emg)
            if hasattr(emg, "numpy"):
                emg = emg.numpy()
        emg = emg.T  # (C, T) where C=8 for target_hand or 16 for emg2pose layouts

        # ── Point-read center frame index (8 bytes) ─────────────────────
        fi_mm = self._frame_memmaps["image_head_frame_index"]
        video_frame_idx = int(fi_mm[center_idx])

        # ── Point-read joint angles at center (20 float32 = 80 bytes) ───
        ja_key = f"generated_joint_angles_{hand}"
        ja = np.asarray(self._frame_memmaps[ja_key][center_idx], dtype=np.float32)

        # ── Point-read wrist pitch/yaw, deg → rad ───────────────────────
        pitch_key = f"mocap_{hand}_wrist_pitch"
        if pitch_key in self._frame_memmaps:
            pitch = np.deg2rad(float(self._frame_memmaps[pitch_key][center_idx]))
            yaw = np.deg2rad(
                float(self._frame_memmaps[f"mocap_{hand}_wrist_yaw"][center_idx])
            )
            ja_full = np.empty(22, dtype=np.float32)
            ja_full[:20] = ja
            ja_full[20] = pitch
            ja_full[21] = yaw
        else:
            ja_full = ja

        # ── Point-read label_valid ──────────────────────────────────────
        lv = self._frame_memmaps["generated_label_valid"][center_idx]
        label_valid = bool(lv[hand_idx])

        # ── Read pre-crop from per-episode LMDB ─────────────────────────
        ep_id = self._episode_id[ep_idx]
        frames_rgb = self._read_episode_crops(
            ep_id, np.array([video_frame_idx], dtype=np.int64), hand_code
        )
        if frames_rgb is None:
            # Missing pre-crop: hand back a zero patch for shape consistency
            # but mark the vision sample INVALID so downstream losses/eval
            # can exclude it instead of silently training on black images.
            frames_rgb = np.zeros(
                (1, self.vision_patch_size, self.vision_patch_size, 3),
                dtype=np.uint8,
            )
            vision_ok = False
        else:
            vision_ok = True

        # ── Normalize single image ──────────────────────────────────────
        frame_rgb = frames_rgb[0]
        img_patch = frame_rgb.transpose(2, 0, 1).astype(np.float32)
        for c in range(3):
            img_patch[c] = (img_patch[c] - self.vision_mean[c]) / self.vision_std[c]

        dataset_name = self._resolve_dataset_name(center_idx)
        return {
            "vision_img": img_patch,  # (3, 256, 256)
            "vision_valid_mask": np.array([vision_ok], dtype=bool),
            "vision_frame_indices": np.array([video_frame_idx], dtype=np.int64),
            "joint_angles": ja_full[:, None],  # (22, 1)
            "label_valid_mask": np.array([label_valid], dtype=bool),
            "target_hand": hand,
            "target_hand_index": hand_idx,
            "dataset_name": dataset_name,
            "emg": emg.astype(np.float32),  # (16, T)
            "emg_channel_mask": emg_channel_mask,
        }

    def _getitem_center_supervised_emg(
        self, ep_idx: int, start: int, end: int
    ) -> dict[str, Any]:
        """Fast path: center_target_only EMG-only mode.

        Full-window EMG read (needed for temporal featurizer) + point-reads
        for joint_angles, label_valid, wrist angles at center frame only.
        No vision I/O at all.
        """
        hand = self.target_hand
        hand_idx = 0 if hand == "left" else 1

        # ── Full-window EMG read ───────────────────────────────────────
        emg_key = f"emg_{hand}_{self.emg_field_preference}"
        emg_raw = np.asarray(
            self._frame_memmaps[emg_key][start:end], dtype=np.float32
        )  # (T, 8)
        if self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"}:
            emg, emg_channel_mask = self._convert_to_emg2pose_layout(
                emg_raw, interpolate_missing=self.channel_interpolate,
            )
        else:
            emg = emg_raw
            emg_channel_mask = np.ones((emg.shape[1],), dtype=np.bool_)

        if self.norm_mode == "per-dataset":
            emg = (emg - self._emg_mean) / (self._emg_std + 1e-6)
        if self.transform is not None:
            emg = self.transform({"emg": emg})
            if isinstance(emg, dict):
                emg = emg.get("emg", emg)
            if hasattr(emg, "numpy"):
                emg = emg.numpy()
        emg = emg.T  # (C, T) where C=8 for target_hand or 16 for emg2pose layouts

        # ── Point-read center frame ────────────────────────────────────
        center = start + self.window_length // 2

        # joint_angles at center (20 float32 = 80 bytes)
        ja_key = f"generated_joint_angles_{hand}"
        ja = np.asarray(self._frame_memmaps[ja_key][center], dtype=np.float32)  # (20,)

        # wrist pitch/yaw at center, deg → rad
        pitch_key = f"mocap_{hand}_wrist_pitch"
        yaw_key = f"mocap_{hand}_wrist_yaw"
        if pitch_key in self._frame_memmaps and yaw_key in self._frame_memmaps:
            pitch = np.deg2rad(float(self._frame_memmaps[pitch_key][center]))
            yaw = np.deg2rad(float(self._frame_memmaps[yaw_key][center]))
            ja_full = np.empty(22, dtype=np.float32)
            ja_full[:20] = ja
            ja_full[20] = pitch
            ja_full[21] = yaw
        else:
            ja_full = ja

        # label_valid at center
        lv = self._frame_memmaps["generated_label_valid"][center]
        label_valid = bool(lv[hand_idx])

        dataset_name = self._resolve_dataset_name(center)
        return {
            "emg": emg.astype(np.float32),
            "emg_channel_mask": emg_channel_mask,
            "joint_angles": ja_full[:, None],  # (22, 1)
            "label_valid_mask": np.array([label_valid], dtype=bool),
            "target_hand": hand,
            "target_hand_index": hand_idx,
            "dataset_name": dataset_name,
        }

    def _resolve_index_to_center(self, idx: int) -> tuple[int, int]:
        """Return (episode_index, center_frame_global_position) for a sample index."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        ep_idx = int(self._block_episode_idx[bi])
        block_start = int(self._block_start[bi])
        if self._window_split_id is None:
            rel = int(idx - self._block_cumsum[bi])
            start = block_start + rel * self.stride
        else:
            start = block_start
        center = start + self.window_length // 2
        return ep_idx, center

    def get_subject(self, idx: int) -> str:
        """Return the subject name for a given sample index."""
        ep_idx, _ = self._resolve_index_to_center(idx)
        return self._episode_subject[ep_idx]

    def get_gesture_class(self, idx: int) -> int:
        """Return the gesture class label at the center frame for a sample index."""
        _, center = self._resolve_index_to_center(idx)
        gc_mm = self._frame_memmaps.get("label_gesture_class")
        if gc_mm is not None:
            return int(gc_mm[center])
        return -1

    def _resolve_dataset_name(self, center_idx: int) -> str:
        """Return the per-sample dataset name.

        For a physically-merged memmap with a ``dataset_source_id`` field, this
        restores the origin dataset name (e.g. egoemg/showee/egoemg_incre) so
        downstream loss masking (lightning.py wrist-mask keyed on dataset_name)
        behaves as in the mixed-dataset regime.  Falls back to the instance
        ``dataset_name`` when the field is absent (non-merged datasets).
        """
        if self._source_id_to_name is None:
            return self.dataset_name
        sid_mm = self._frame_memmaps.get("dataset_source_id")
        if sid_mm is None:
            return self.dataset_name
        sid = int(sid_mm[center_idx])
        return self._source_id_to_name.get(sid, self.dataset_name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._ensure_runtime_owner()

        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        ep_idx = int(self._block_episode_idx[bi])
        ep_end = int(self._episode_end_idx[ep_idx])
        block_start = int(self._block_start[bi])
        if self._window_split_id is None:
            rel = int(idx - self._block_cumsum[bi])
            start = block_start + rel * self.stride
            end = start + self.window_length
            window_split_id = None
        else:
            start = block_start
            end = int(self._block_end[bi])
            window_split_id = int(self._window_split_id[bi])

        # Fast path: vision_only + preloaded ViT features.
        # Zero file I/O — all data was preloaded into numpy arrays.
        # Jitter is not applied here because features were preloaded at fixed
        # positions matching the dataset index.
        if self.skip_emg_loading and self._preloaded_features is not None:
            center_idx = start + self.window_length // 2
            return self._getitem_vision_only_preloaded(idx, center_idx)

        if self.jitter:
            leftover = ep_end - end
            if leftover > 0:
                start += np.random.randint(0, min(self.stride, leftover))
                end = start + self.window_length

        # Fast path: vision_only + per-episode pre-crops.
        # Point-reads only the center frame from memmaps (skips full-window
        # slicing) and bypasses MANO .npy, episode metadata strings, and
        # unused label/mocap fields.  Placed after jitter so that the random
        # window offset still varies the center frame.
        if (
            self.skip_emg_loading
            and self._vision_enabled
            and self.per_episode_crops_dir is not None
        ):
            center_idx = start + self.window_length // 2
            return self._getitem_vision_only_crops(ep_idx, center_idx)

        # Fast path: center_supervised fusion mode.
        # Full EMG window read (needed for temporal featurizer) + point-reads
        # for joint_angles, label_valid, wrist angles at center frame only.
        if (
            self._center_target_only
            and self._vision_enabled
            and self.per_episode_crops_dir is not None
        ):
            center_idx = start + self.window_length // 2
            return self._getitem_center_supervised(ep_idx, start, end, center_idx)

        # Fast path: center_supervised EMG-only mode.
        # Full-window EMG read + point-reads for joint_angles at center frame.
        # No vision, no full-window joint_angle / mocap / label slicing.
        if self._center_target_only and not self._vision_enabled:
            return self._getitem_center_supervised_emg(ep_idx, start, end)

        # Center frame index for the main (full-window) path. Needed by
        # _resolve_dataset_name below; the fast paths above compute their own.
        center_idx = start + self.window_length // 2

        result: dict[str, Any] = {}
        for name, mm in self._frame_memmaps.items():
            if name == "image_head_frame_index" and self._cached_vit_enabled:
                # Only the center value is needed for ViT feature lookup.
                # Reading the full 7790-element array is wasteful.
                result["_image_head_frame_index_center"] = int(
                    mm[start + self.window_length // 2]
                )
                continue
            result[name] = np.asarray(mm[start:end])

        beta_idx = int(self._episode_beta_idx[ep_idx])
        for name, mm in self._episode_memmaps.items():
            result[name] = np.asarray(mm[beta_idx])

        result["episode_id"] = self._episode_id[ep_idx]
        result["episode_subject"] = self._episode_subject[ep_idx]
        result["episode_subject_id"] = int(self._episode_subject_id[ep_idx])
        result["episode_source_parquet"] = self._episode_source_parquet[ep_idx]
        result["episode_zed_video_path"] = self._episode_zed_video_path[ep_idx]
        result["episode_head_video_path"] = self._episode_head_video_path[ep_idx]
        result["episode_wrist_left_video_path"] = \
            self._episode_wrist_left_video_path[ep_idx]
        result["episode_wrist_right_video_path"] = \
            self._episode_wrist_right_video_path[ep_idx]
        result["window_start_idx"] = start
        result["window_end_idx"] = end
        result["window_length"] = self.window_length
        if window_split_id is not None:
            result["window_split_id"] = window_split_id
            result["window_split"] = self._splits[window_split_id]
            split_name = self._splits[window_split_id]
            result["held_out_user"] = split_name in ("user", "both")
            result["held_out_stage"] = split_name in ("gesture", "both")

        for stream in self.video_streams or []:
            if stream not in VIDEO_STREAM_PATH_KEYS:
                raise ValueError(f"Unknown video stream: {stream}")
            path_key = VIDEO_STREAM_PATH_KEYS[stream]
            frame_key = f"image_{stream}_frame_index"
            frames = np.asarray(result[frame_key], dtype=np.int64)
            result[f"{stream}_video_path"] = self._resolve_video_path(result[path_key])
            if self.decode_video:
                result[f"{stream}_video_frames"] = self._decode_video_frames(
                    result[f"{stream}_video_path"],
                    frames,
                )

        if self.target_hand is not None:
            hand = self.target_hand
            hand_idx = 0 if hand == "left" else 1

            if self.skip_emg_loading:
                n_channels = 16 if self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"} else 8
                # Minimal placeholder — the model either pre-computes T or
                # runs the featurizer on a single-frame dummy.  We no longer
                # materialise (n_channels × window_length) zeros per sample.
                result["emg"] = np.zeros((n_channels, 0), dtype=np.float32)
                result["emg_channel_mask"] = np.ones((n_channels,), dtype=np.bool_)
            else:
                if self.emg_layout in {"emg2pose_sparse16", "emg2pose_interpolate16"}:
                    emg_key = f"emg_{hand}_{self.emg_field_preference}"
                    emg_raw = np.asarray(result[emg_key], dtype=np.float32)
                    emg, emg_channel_mask = self._convert_to_emg2pose_layout(
                        emg_raw,
                        interpolate_missing=self.channel_interpolate,
                    )
                else:
                    emg_key = f"emg_{hand}_{self.emg_field_preference}"
                    emg = np.asarray(result[emg_key], dtype=np.float32)
                    emg_channel_mask = np.ones((emg.shape[1],), dtype=np.bool_)
                if emg is not None:
                    if self.norm_mode == "per-dataset":
                        emg = (emg - self._emg_mean) / (self._emg_std + 1e-6)
                    if self.transform is not None:
                        emg = self.transform({"emg": emg})
                        if isinstance(emg, dict):
                            emg = emg.get("emg", emg)
                        if hasattr(emg, "numpy"):
                            emg = emg.numpy()
                    result["emg"] = emg.T
                    result["emg_channel_mask"] = emg_channel_mask

                # Clean up raw emg fields to avoid collation mismatch
                # between left/right hand samples in the same batch
                for emg_suffix in ("_raw", "_filtered"):
                    for h in ("left", "right"):
                        result.pop(f"emg_{h}{emg_suffix}", None)

            ja_key = f"generated_joint_angles_{hand}"
            if ja_key in result:
                ja = np.asarray(result[ja_key]).T  # (C, T) where C=20

                # Append wrist pitch/yaw if available → 22-dim output
                pitch_key = f"mocap_{hand}_wrist_pitch"
                yaw_key = f"mocap_{hand}_wrist_yaw"
                wrist_valid_key = f"mocap_{hand}_wrist_angles_valid"
                if pitch_key in result and yaw_key in result:
                    # Wrist angles are stored in degrees; convert to radians
                    # to match joint angles unit
                    pitch = np.deg2rad(np.asarray(result[pitch_key], dtype=np.float32))
                    yaw = np.deg2rad(np.asarray(result[yaw_key], dtype=np.float32))
                    wrist = np.stack([pitch, yaw], axis=0)  # (2, T)
                    ja = np.concatenate([ja, wrist], axis=0)  # (22, T)

                    # Clean up raw wrist fields to avoid collation mismatch
                    # between left/right hand samples in the same batch
                    result.pop(pitch_key, None)
                    result.pop(yaw_key, None)
                    if wrist_valid_key in result:
                        # Loaded for completeness; currently not surfaced to the
                        # loss (wrist-masking is driven by dataset_name instead).
                        result.pop(wrist_valid_key, None)

                result["joint_angles"] = ja

            # Vision-only: keep only the center frame joint angles.
            # One image → one pose.  No broadcast needed.
            if self.skip_emg_loading and self._vision_enabled:
                center = result["joint_angles"].shape[-1] // 2
                result["joint_angles"] = result["joint_angles"][:, center:center + 1]

            kp_key = f"mocap_{hand}_keypoints"
            if kp_key in result:
                result["mocap_keypoints"] = np.asarray(result[kp_key])

            valid_key = f"mocap_{hand}_valid"
            if valid_key in result:
                result["mocap_valid"] = np.asarray(result[valid_key])

            if "generated_label_valid" in result:
                result["label_valid_mask"] = np.asarray(result["generated_label_valid"])[:, hand_idx]

            # Vision-only: keep only center frame validity.
            if self.skip_emg_loading and self._vision_enabled and "label_valid_mask" in result:
                center = result["label_valid_mask"].shape[-1] // 2
                result["label_valid_mask"] = result["label_valid_mask"][center:center + 1]

            beta_key = f"generated_mano_{hand}_beta"
            pose_key = f"generated_mano_{hand}_pose"
            npy_key = (ep_idx, hand)
            if npy_key in self._mano_npy_pose:
                pose_mm = self._mano_npy_pose[npy_key]
                ep_start = int(self._episode_start_idx[ep_idx])
                local_start = start - ep_start
                local_end = end - ep_start
                result["mano_pose"] = np.asarray(pose_mm[local_start:local_end], dtype=np.float32)
                result["mano_beta"] = np.asarray(self._mano_npy_beta[npy_key], dtype=np.float32)
                if npy_key in self._mano_npy_trans:
                    result["mano_trans"] = np.asarray(self._mano_npy_trans[npy_key], dtype=np.float32)
                if npy_key in self._mano_npy_world_R:
                    result["mano_world_R"] = np.asarray(
                        self._mano_npy_world_R[npy_key][local_start:local_end], dtype=np.float32)
                if npy_key in self._mano_npy_world_t:
                    result["mano_world_t"] = np.asarray(
                        self._mano_npy_world_t[npy_key][local_start:local_end], dtype=np.float32)
            else:
                if beta_key in result:
                    result["mano_beta"] = np.asarray(result[beta_key])
                if pose_key in result:
                    result["mano_pose"] = np.asarray(result[pose_key])

            result["target_hand"] = hand
            result["target_hand_index"] = hand_idx
            result["dataset_name"] = self._resolve_dataset_name(center_idx)
            result.update(self._build_vision_sample(result, hand, hand_idx, idx))
        else:
            result["dataset_name"] = self._resolve_dataset_name(center_idx)

        # Remove raw hand-specific memmap fields to keep collation keys
        # consistent when left-hand and right-hand datasets are concatenated.
        # Match precise per-hand field families instead of "_left"/"_right"
        # substrings: substring matching silently swallowed the imu wrist
        # fields (imu_band_left/right, imu_cam_wrist_left/right), which are
        # positional, not per-hand.
        per_hand_prefixes = (
            "emg_left_", "emg_right_",
            "generated_joint_angles_left", "generated_joint_angles_right",
            "generated_mano_left", "generated_mano_right",
            "mocap_left_", "mocap_right_",
        )
        for key in list(result.keys()):
            if key.startswith(per_hand_prefixes):
                result.pop(key, None)

        # Fill default values for optional fields that were requested via modalities
        # but are missing from this dataset. This ensures consistent keys across
        # datasets when using ConcatDataset.
        T = self.window_length
        if "label_gesture_class" not in result and "labels" in (self.modalities or []):
            result["label_gesture_class"] = np.full(T, -1, dtype=np.int32)
        if "label_gesture_active" not in result and "labels" in (self.modalities or []):
            result["label_gesture_active"] = np.zeros(T, dtype=bool)

        if get_worker_info() is None:
            self.reset_runtime_caches()
        return result
