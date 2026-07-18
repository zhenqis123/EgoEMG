from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2pose import transforms
from torch.utils.data import Dataset

_IGNORE_INDEX = -100


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def _normalize_list(cfgs: Any) -> list[Any]:
    if cfgs is None:
        return []
    if OmegaConf.is_dict(cfgs):
        return [cfgs]
    if OmegaConf.is_list(cfgs):
        return list(cfgs)
    if isinstance(cfgs, Sequence) and not isinstance(cfgs, (str, bytes)):
        return list(cfgs)
    return [cfgs]


def _to_int_mapping(raw: Mapping[Any, Any] | None) -> dict[int, int] | None:
    if raw is None:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def _resolve_transform(transform: Any) -> Callable[[Any], Any] | None:
    if transform is None:
        return None
    if callable(transform):
        return transform

    if isinstance(transform, (DictConfig, ListConfig, dict)):
        configs = _normalize_list(transform)
    elif isinstance(transform, Sequence) and not isinstance(transform, (str, bytes)):
        configs = list(transform)
    else:
        return transform

    built: list[Callable[[Any], Any]] = []
    for cfg in configs:
        if cfg is None:
            continue
        if callable(cfg):
            built.append(cfg)
        elif OmegaConf.is_config(cfg) or isinstance(cfg, dict):
            built.append(instantiate(cfg))
        else:
            raise TypeError(f"Unsupported transform config type: {type(cfg)!r}")

    if not built:
        return None
    if len(built) == 1:
        return built[0]
    return transforms.Compose(built)


@dataclass
class PretrainWrapperDataset(Dataset):
    """Wrap a base dataset to standardize EMG channels and supervision targets."""

    base: Any
    name: str
    gesture_spaces: Sequence[Mapping[str, Any]] | None = None
    angle_dim: int = 22
    transform: Any | None = None
    norm_mode: str | None = None
    norm_stats_path: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.base, Dataset):
            self._base = self.base
        else:
            base_cfg = self.base
            if not OmegaConf.is_config(base_cfg) and not isinstance(base_cfg, dict):
                raise ValueError("base must be a Dataset or a config mapping.")
            self._base = instantiate(base_cfg, transform=None, _convert_="all")

        self._spaces = []
        for space in _normalize_list(self.gesture_spaces):
            if space is None:
                continue
            space_dict = dict(space)
            space_dict["num_classes"] = int(space_dict["num_classes"])
            space_dict["datasets"] = (
                list(space_dict["datasets"])
                if space_dict.get("datasets") is not None
                else None
            )
            space_dict["value_map"] = _to_int_mapping(space_dict.get("value_map"))
            self._spaces.append(space_dict)

        self._transform = _resolve_transform(self.transform)

        # Pre-load per-dataset normalization stats
        self._emg_mean = 0.0
        self._emg_std = 1.0
        if self.norm_mode == "per-dataset" and self.norm_stats_path:
            import json
            with open(self.norm_stats_path) as f:
                all_stats = json.load(f)
            stats = all_stats.get(self.name, {"mean": 0.0, "std": 1.0})
            self._emg_mean = float(stats["mean"])
            self._emg_std = float(stats["std"])

    def __len__(self) -> int:
        return len(self._base)

    def __getstate__(self):
        # Ensure _transform is properly set before pickling
        state = self.__dict__.copy()
        if '_transform' not in state:
            state['_transform'] = _resolve_transform(state.get('transform'))
        # Remove the original transform to avoid confusion
        state.pop('transform', None)
        return state

    def __setstate__(self, state):
        # Restore state and ensure _transform exists
        self.__dict__.update(state)
        if '_transform' not in self.__dict__:
            # This shouldn't happen if __getstate__ worked correctly
            self._transform = _resolve_transform(self.__dict__.get('transform'))
        # Ensure transform field exists for dataclass compatibility
        if 'transform' not in self.__dict__:
            self.transform = None

    def _normalize_joint_angles(
        self,
        joint_angles: torch.Tensor | None,
        valid_mask: torch.Tensor,
        time_len: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angle_target = torch.zeros(self.angle_dim, time_len, dtype=dtype)
        angle_mask = torch.zeros(self.angle_dim, time_len, dtype=torch.bool)
        if joint_angles is None:
            return angle_target, angle_mask

        ja = _as_tensor(joint_angles, dtype=dtype)
        if ja.ndim == 1:
            ja = ja.unsqueeze(0)
        if ja.shape[-1] != time_len and ja.shape[0] == time_len:
            ja = ja.T

        if ja.shape[0] == self.angle_dim:
            used = self.angle_dim
            angle_target[:used] = ja[:used]
            finite = torch.isfinite(ja[:used])
            angle_mask[:used] = finite & valid_mask[None, :]
            # Ninapro DB9-derived angles do not provide MIDDLE_MCP_AA (index 8).
            # Keep target value but exclude it from supervision.
            if self.name == "ninapro" and self.angle_dim > 8:
                angle_mask[8] = False
        elif ja.shape[0] == self.angle_dim - 2:
            used = self.angle_dim - 2
            angle_target[:used] = ja[:used]
            finite = torch.isfinite(ja[:used])
            angle_mask[:used] = finite & valid_mask[None, :]

        if not torch.isfinite(angle_target).all():
            angle_target = torch.nan_to_num(
                angle_target, nan=0.0, posinf=0.0, neginf=0.0
            )
        return angle_target, angle_mask

    def _map_labels(
        self, raw: torch.Tensor, num_classes: int, value_map: dict[int, int] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full(raw.shape, _IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(raw.shape, dtype=torch.bool)
        if value_map:
            for src, dst in value_map.items():
                match = raw == src
                if match.any():
                    labels[match] = int(dst)
                    mask[match] = True
            return labels, mask

        labels = raw.to(torch.long)
        mask = (labels >= 0) & (labels < num_classes)
        labels = torch.where(mask, labels, torch.full_like(labels, _IGNORE_INDEX))
        return labels, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._base[idx]
        if not isinstance(sample, dict):
            raise ValueError("Base dataset must return a dict sample.")

        emg = _as_tensor(sample.get("emg"), dtype=torch.float32)
        if emg.ndim != 2:
            raise ValueError(f"Expected emg shape (C, T), got {tuple(emg.shape)}")

        # Per-dataset normalization (before transforms/augmentations)
        if self.norm_mode == "per-dataset" and self._emg_std > 0:
            emg = (emg - self._emg_mean) / (self._emg_std + 1e-6)

        time_len = int(emg.shape[-1])

        valid_mask = sample.get("label_valid_mask")
        if valid_mask is None:
            valid_mask_t = torch.ones(time_len, dtype=torch.bool)
        else:
            valid_mask_t = _as_tensor(valid_mask, dtype=torch.bool)
            if valid_mask_t.ndim > 1:
                valid_mask_t = valid_mask_t.reshape(-1)
            if valid_mask_t.numel() != time_len:
                valid_mask_t = valid_mask_t[:time_len]

        angle_target, angle_mask = self._normalize_joint_angles(
            sample.get("joint_angles"), valid_mask_t, time_len, emg.dtype
        )

        if self._transform is not None:
            # Transpose to (T, C) so that transforms expecting
            # channel_dim=-1 / time_dim=0 operate on the correct axes.
            emg_t = emg.T
            transformed = self._transform({"emg": emg_t})
            if isinstance(transformed, dict):
                emg_t = transformed.get("emg", emg_t)
            else:
                emg_t = transformed
            emg = emg_t.T

        gesture_labels = torch.full(
            (len(self._spaces), time_len), _IGNORE_INDEX, dtype=torch.long
        )
        gesture_masks = torch.zeros((len(self._spaces), time_len), dtype=torch.bool)
        for i, space in enumerate(self._spaces):
            datasets = space.get("datasets")
            if datasets is not None and self.name not in datasets:
                continue
            field = space.get("field")
            raw = sample.get(field)
            if raw is None:
                continue
            raw_t = _as_tensor(raw)
            if raw_t.ndim > 1:
                raw_t = raw_t.reshape(-1)
            if raw_t.numel() != time_len:
                raw_t = raw_t[:time_len]
            mapped, mask = self._map_labels(
                raw_t, int(space["num_classes"]), space.get("value_map")
            )
            gesture_labels[i] = mapped
            gesture_masks[i] = mask

        # Handle keystroke labels for emg2qwerty dataset
        keystroke_labels = sample.get("target_keystrokes")
        if keystroke_labels is not None:
            # import ipdb; ipdb.set_trace()
            keystroke_labels = _as_tensor(keystroke_labels, dtype=torch.long)
        else:
            keystroke_labels = torch.tensor([], dtype=torch.long)

        output = {
            "emg": emg,
            "label_valid_mask": valid_mask_t,
            "angle_target": angle_target,
            "angle_mask": angle_mask,
            "gesture_labels": gesture_labels,
            "gesture_masks": gesture_masks,
            "keystroke_labels": keystroke_labels,
            "dataset_name": self.name,
        }
        if "emg_channel_mask" in sample:
            output["emg_channel_mask"] = _as_tensor(sample["emg_channel_mask"], dtype=torch.bool)
        else:
            output["emg_channel_mask"] = torch.ones(emg.shape[0], dtype=torch.bool)
        return output
