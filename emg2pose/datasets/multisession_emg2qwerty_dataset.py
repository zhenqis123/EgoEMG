# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from collections import OrderedDict
import string

try:
    import unidecode
except ImportError:  # pragma: no cover
    unidecode = None


def _unidecode(text: str) -> str:
    if unidecode is None:
        return text
    return unidecode.unidecode(text)


class H5Pool:
    """Per-process LRU cache of opened HDF5 files."""

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: str) -> dict[str, Any]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        f = h5py.File(path, "r")
        g = f["emg2qwerty"]
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


class CharacterSet:
    """Minimal charset utility for keystroke-to-label conversion."""

    CHAR_TO_UNICODE = [
        (c, ord(c)) for c in string.ascii_letters + string.digits + string.punctuation
    ]
    MODIFIER_TO_UNICODE = [
        ("Key.backspace", 9003),  # ⌫
        ("Key.enter", 9166),  # ⏎
        ("Key.space", 32),
        ("Key.shift", 8679),  # ⇧
    ]
    KEY_TO_UNICODE = OrderedDict([*CHAR_TO_UNICODE, *MODIFIER_TO_UNICODE])
    UNICHAR_TO_KEY = {
        " ": "Key.space",
        "\r": "Key.enter",
        "\u21E5": "Key.tab",
        "\u21E7": "Key.shift",
        "\u2303": "Key.ctrl",
        "\u2318": "Key.cmd",
        "\u232B": "Key.backspace",
        "\u23CE": "Key.enter",
        "\u2191": "Key.shift_l",
        "\u21E1": "Key.shift_r",
    }
    CHAR_SUBSTITUTIONS = {
        "\n": "⏎",
        "\r": "⏎",
        "\b": "⌫",
        "’": "'",
        "“": '"',
        "”": '"',
        "—": "-",
    }

    def __init__(self):
        self._key_to_unicode = self.KEY_TO_UNICODE
        self._unicode_to_key = {v: k for k, v in self._key_to_unicode.items()}

    def __contains__(self, item: str | int) -> bool:
        if isinstance(item, str):
            return item in self._key_to_unicode
        if isinstance(item, int):
            return item in self._unicode_to_key
        return False

    @property
    def allowed_keys(self) -> tuple[str, ...]:
        return tuple(self._key_to_unicode.keys())

    @property
    def allowed_unicodes(self) -> tuple[int, ...]:
        return tuple(self._key_to_unicode.values())

    def key_to_unicode(self, key: str) -> int:
        return self._key_to_unicode[key]

    def key_to_label(self, key: str) -> int:
        return self.allowed_keys.index(key)

    def label_to_key(self, label: int) -> str:
        return self.allowed_keys[label]

    def keys_to_str(self, keys: list[str]) -> str:
        return "".join(chr(self.key_to_unicode(key)) for key in keys)

    def str_to_keys(self, unicode_str: str) -> list[str]:
        keys = list(self._normalize_str(unicode_str))
        return self.clean_keys(keys)

    def str_to_labels(self, unicode_str: str) -> list[int]:
        keys = self.str_to_keys(unicode_str)
        return [self.key_to_label(key) for key in keys]

    def labels_to_str(self, labels: list[int]) -> str:
        keys = [self.label_to_key(label) for label in labels]
        return self.keys_to_str(keys)

    def clean_keys(self, keys: list[str]) -> list[str]:
        keys = self._normalize_keys(keys)
        return [key for key in keys if key in self]

    def clean_str(self, unicode_str: str) -> str:
        keys = list(self._normalize_str(unicode_str))
        keys = self.clean_keys(keys)
        return self.keys_to_str(keys)

    def _normalize_keys(self, keys: list[str]) -> list[str]:
        def _normalize_key(key: str) -> str:
            if key in self:
                return key
            if len(key) == 1:
                key = self._normalize_str(key)
                key = self.UNICHAR_TO_KEY.get(key, key)
            return key

        return [_normalize_key(key) for key in keys]

    def _normalize_str(self, unicode_str: str) -> str:
        normalized_str = unicode_str
        for k, v in self.CHAR_SUBSTITUTIONS.items():
            normalized_str = normalized_str.replace(k, v)

        def _spurious_char(c: str) -> bool:
            return c not in self and c not in self.UNICHAR_TO_KEY

        unidecode_map = {}
        for c in normalized_str:
            if not _spurious_char(c):
                continue
            c_ = _unidecode(c)
            if c_ != c and len(c_) == 1 and not _spurious_char(c_):
                unidecode_map[c] = c_

        for k, v in unidecode_map.items():
            normalized_str = normalized_str.replace(k, v)

        return normalized_str


_CHARSET = CharacterSet()


class LabelData:
    def __init__(self, text: str, timestamps: list[float] | None = None):
        self.text = text
        self.timestamps = None if timestamps is None else np.array(timestamps)

    @classmethod
    def from_keystrokes(cls, keystrokes: list[dict[str, Any]], start_t: float, end_t: float) -> "LabelData":
        label_data = cls(text="", timestamps=[])
        for key in keystrokes:
            if key["start"] > end_t:
                break
            if key["start"] >= start_t:
                label_data += cls.from_key(key)
        return label_data

    @classmethod
    def from_key(cls, key: str | dict[str, Any]) -> "LabelData":
        if isinstance(key, str):
            _key = key
            timestamp = None
        else:
            _key = key["key"]
            timestamp = key["start"]

        _key = _CHARSET._normalize_keys([_key])[0]
        if _key not in _CHARSET:
            return cls(text="", timestamps=[])

        text = _CHARSET.keys_to_str([_key])
        timestamps = [timestamp] if timestamp is not None else None
        return cls(text, timestamps)

    @classmethod
    def from_prompts(cls, prompts: list[dict[str, Any]], start_t: float, end_t: float) -> "LabelData":
        label_data = cls(text="")
        for prompt in prompts:
            if prompt["start"] > end_t:
                break
            if prompt["start"] >= start_t:
                label_data += cls.from_prompt(prompt)
        return label_data

    @classmethod
    def from_prompt(cls, prompt: str | dict[str, Any], enforce_newline: bool = True) -> "LabelData":
        if isinstance(prompt, str):
            text = prompt
        else:
            payload = prompt.get("payload")
            text = payload.get("text") if payload is not None else None

        if text is None:
            return cls(text="")

        text = _CHARSET.clean_str(text)
        if enforce_newline and (len(text) == 0 or text[-1] != "⏎"):
            text += "⏎"
        return cls(text)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(_CHARSET.str_to_labels(self.text), dtype=np.int32)

    def __add__(self, other: "LabelData") -> "LabelData":
        text = self.text + other.text
        if self.timestamps is not None and other.timestamps is not None:
            timestamps = np.append(self.timestamps, other.timestamps).tolist()
        else:
            timestamps = None
        return LabelData(text, timestamps)


@dataclass
class MultiSessionWindowedEmg2QwertyDataset(Dataset):
    hdf5_paths: list[Path]
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Any | None = None
    max_open_files: int = 32

    def __post_init__(self):
        assert self.window_length > 0
        self.stride = self.stride or self.window_length
        assert self.stride > 0

        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        (
            self._block_session_idx,
            self._block_start,
            self._block_end,
            self._block_cumsum,
        ) = self._build_blocks_index()

        self._pool: H5Pool | None = None
        self._meta_cache: dict[str, dict[str, Any]] = {}

    def _get_pool(self) -> H5Pool:
        if self._pool is None:
            self._pool = H5Pool(max_open=self.max_open_files)
        return self._pool

    def _session_len(self, path: Path) -> int:
        with h5py.File(path, "r") as f:
            return len(f["emg2qwerty"]["timeseries"])

    def _build_blocks_index(self):
        block_session_idx = []
        block_start = []
        block_end = []
        block_lengths = []

        for si, p in enumerate(self.hdf5_paths):
            slen = self._session_len(p)
            if slen < self.window_length:
                continue
            block_session_idx.append(si)
            block_start.append(0)
            block_end.append(slen)
            n = (slen - self.window_length) // self.stride + 1
            block_lengths.append(n)

        block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
        block_start = np.asarray(block_start, dtype=np.int64)
        block_end = np.asarray(block_end, dtype=np.int64)
        block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))
        return block_session_idx, block_start, block_end, block_cumsum

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def _get_meta(self, path: str, group: h5py.Group) -> dict[str, Any]:
        if path in self._meta_cache:
            return self._meta_cache[path]
        meta: dict[str, Any] = {}
        for key, val in group.attrs.items():
            if key in {"keystrokes", "prompts"}:
                meta[key] = json.loads(val)
            else:
                meta[key] = val
        self._meta_cache[path] = meta
        return meta

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

        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding

        path = str(self.hdf5_paths[si])
        h5 = self._get_pool().get(path)
        g = h5["group"]
        ts = h5["timeseries"]
        window = ts[window_start:window_end]

        # EMG: stack left/right into 32 channels
        if self.transform is None:
            emg_left = torch.as_tensor(window["emg_left"])
            emg_right = torch.as_tensor(window["emg_right"])
            emg = torch.cat([emg_left, emg_right], dim=-1)  # (T, 32)
        else:
            emg = self.transform(window)
            if emg.ndim == 3 and emg.shape[1] == 2:
                emg = emg.reshape(emg.shape[0], -1)
        assert torch.is_tensor(emg)

        # Labels: keystrokes within [start_t, end_t]
        timestamps = window["time"]
        start_t = timestamps[offset - window_start]
        end_t = timestamps[(offset + self.window_length - 1) - window_start]

        meta = self._get_meta(path, g)
        condition = str(meta.get("condition", "on_keyboard"))
        keystrokes = meta.get("keystrokes", [])
        prompts = meta.get("prompts", [])

        if condition == "on_keyboard":
            label_data = LabelData.from_keystrokes(keystrokes, start_t=start_t, end_t=end_t)
        else:
            label_data = LabelData.from_prompts(prompts, start_t=start_t, end_t=end_t)
        target_keystrokes = torch.as_tensor(label_data.labels, dtype=torch.int32)

        return {
            "emg": emg.T,  # CT
            "target_keystrokes": target_keystrokes,
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "session_idx": si,
            "user": str(meta.get("user", "unknown")),
            "condition": condition,
        }
