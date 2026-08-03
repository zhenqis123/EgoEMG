# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import zarr
import string

try:
    import unidecode
except ImportError:  # pragma: no cover
    unidecode = None


def _unidecode(text: str) -> str:
    if unidecode is None:
        return text
    return unidecode.unidecode(text)


def _decode_bytes(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            decoded.append(v.decode("utf-8", errors="replace").rstrip("\x00"))
        else:
            decoded.append(str(v))
    return decoded


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


class _EventCache:
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._cache: OrderedDict[int, tuple[np.ndarray, list[str]]] = OrderedDict()

    def get(self, key: int) -> tuple[np.ndarray, list[str]] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: tuple[np.ndarray, list[str]]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


def _select_random_side(emg: torch.Tensor) -> torch.Tensor:
    # emg expected shape (T, C) or (C, T), choose one side of 16 channels
    if emg.ndim != 2:
        return emg
    if emg.shape[1] >= 32:
        side = int(torch.randint(0, 2, ()).item())
        start = side * 16
        return emg[:, start : start + 16]
    if emg.shape[0] >= 32 and emg.shape[1] < 32:
        side = int(torch.randint(0, 2, ()).item())
        start = side * 16
        return emg[start : start + 16, :]
    return emg


@dataclass
class Emg2QwertyDataset(Dataset):
    root_dir: Path
    window_length: int = 10_000
    stride: int | None = None
    padding: tuple[int, int] = (0, 0)
    jitter: bool = False
    transform: Any | None = None
    allowed_sessions: Sequence[str] | None = None
    allowed_users: Sequence[str] | None = None
    allowed_conditions: Sequence[str] | None = None
    memmap_dir: Path | None = None  # directory with memmap .dat files (from zarr_to_memmap.py)

    def __post_init__(self) -> None:
        self.stride = self.stride or self.window_length
        assert self.window_length > 0 and self.stride > 0

        self.left_padding, self.right_padding = self.padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        self.root_dir = Path(self.root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing Zarr store at {self.root_dir}")

        self._root: zarr.Group | None = None
        self._memmaps: dict[str, np.memmap] = {}
        self._metadata_npz: dict | None = None

        # Try loading from metadata.npz first, then fallback to zarr
        self._load_catalog()
        self._build_blocks_index()
        self._init_memmaps()

        self._keystroke_cache = _EventCache(max_size=64)
        self._prompt_cache = _EventCache(max_size=64)

    def _get_root(self) -> zarr.Group:
        if self._root is None:
            self._root = zarr.open_group(str(self.root_dir), mode="r")
        return self._root

    def _init_memmaps(self) -> None:
        # Determine memmap directory - check multiple possible locations
        mdir = None

        # 1. Explicit memmap_dir
        if self.memmap_dir is not None:
            mdir = Path(self.memmap_dir)

        # 2. root_dir might be the memmap directory itself
        if mdir is None and (self.root_dir / "manifest.json").exists():
            mdir = self.root_dir

        # 3. root_dir_memmap suffix
        if mdir is None:
            candidate = Path(str(self.root_dir).rstrip("/") + "_memmap")
            if candidate.exists():
                mdir = candidate

        if mdir is None or not mdir.exists():
            return

        manifest_path = mdir / "manifest.json"
        if not manifest_path.exists():
            return

        with open(manifest_path) as f:
            manifest = json.load(f)

        for field_name, info in manifest.get("fields", {}).items():
            dat_path = mdir / info["filename"]
            if not dat_path.exists():
                continue
            dtype = np.dtype(info["dtype"])
            shape = tuple(info["shape"])
            self._memmaps[field_name] = np.memmap(
                str(dat_path), dtype=dtype, mode="r", shape=shape,
            )

    def _load_catalog(self) -> None:
        # Try loading from metadata.npz first
        # Check multiple possible locations
        metadata_paths = []

        # 1. Explicit memmap_dir
        if self.memmap_dir is not None:
            metadata_paths.append(Path(self.memmap_dir) / "metadata.npz")

        # 2. root_dir might be the memmap directory itself
        metadata_paths.append(self.root_dir / "metadata.npz")

        # 3. root_dir_memmap suffix
        metadata_paths.append(Path(str(self.root_dir).rstrip("/") + "_memmap") / "metadata.npz")

        for metadata_path in metadata_paths:
            if metadata_path.exists():
                self._load_catalog_from_npz(metadata_path)
                return

        # Fallback to zarr
        root = self._get_root()
        sessions = root["sessions"]

        self._session_id = _decode_bytes(np.asarray(sessions["session_id"]))
        self._session_start = np.asarray(sessions["start_idx"], dtype=np.int64)
        self._session_length = np.asarray(sessions["length"], dtype=np.int64)
        self._session_end = np.asarray(sessions["end_idx"], dtype=np.int64)
        self._session_user_id = np.asarray(sessions["user_id"], dtype=np.int32)
        self._session_condition_id = np.asarray(
            sessions["condition_id"], dtype=np.int32
        )

        self._users = _decode_bytes(np.asarray(root["users"]["user"]))
        self._conditions = _decode_bytes(np.asarray(root["conditions"]["condition"]))

        keystrokes = root["keystrokes"]
        self._keystroke_start = keystrokes["start"]
        self._keystroke_key = keystrokes["key"]
        self._keystroke_offset = np.asarray(
            keystrokes["session_offset"], dtype=np.int64
        )
        self._keystroke_length = np.asarray(
            keystrokes["session_length"], dtype=np.int64
        )

        prompts = root["prompts"]
        self._prompt_start = prompts["start"]
        self._prompt_text = prompts["text"]
        self._prompt_offset = np.asarray(prompts["session_offset"], dtype=np.int64)
        self._prompt_length = np.asarray(prompts["session_length"], dtype=np.int64)

    def _load_catalog_from_npz(self, npz_path: Path) -> None:
        """Load catalog from metadata.npz."""
        data = np.load(npz_path)

        # Sessions
        self._session_id = _decode_bytes(data["session_session_id"])
        self._session_start = data["session_start_idx"]
        self._session_length = data["session_length"]
        self._session_end = data["session_end_idx"]
        self._session_user_id = data["session_user_id"]
        self._session_condition_id = data["session_condition_id"]

        # Users and conditions
        self._users = _decode_bytes(data["users_user"])
        self._conditions = _decode_bytes(data["conditions_condition"])

        # Keystrokes
        self._keystroke_start = data["keystrokes_start"]
        self._keystroke_key = data["keystrokes_key"]
        self._keystroke_offset = data["keystrokes_session_offset"]
        self._keystroke_length = data["keystrokes_session_length"]

        # Prompts
        self._prompt_start = data["prompts_start"]
        self._prompt_text = data["prompts_text"]
        self._prompt_offset = data["prompts_session_offset"]
        self._prompt_length = data["prompts_session_length"]

    def _filter_session_indices(self) -> np.ndarray:
        n_sessions = len(self._session_id)
        mask = np.ones((n_sessions,), dtype=bool)

        if self.allowed_sessions:
            allowed = set(self.allowed_sessions)
            session_mask = np.array(
                [sid in allowed for sid in self._session_id], dtype=bool
            )
            mask &= session_mask

        if self.allowed_users:
            user_to_id = {u: i for i, u in enumerate(self._users)}
            ids = {user_to_id[u] for u in self.allowed_users if u in user_to_id}
            mask &= np.isin(self._session_user_id, list(ids))

        if self.allowed_conditions:
            cond_to_id = {c: i for i, c in enumerate(self._conditions)}
            ids = {cond_to_id[c] for c in self.allowed_conditions if c in cond_to_id}
            mask &= np.isin(self._session_condition_id, list(ids))

        return np.nonzero(mask)[0].astype(np.int64)

    def _build_blocks_index(self) -> None:
        allowed_sessions = self._filter_session_indices()

        block_session_idx: list[int] = []
        block_start: list[int] = []
        block_end: list[int] = []
        block_lengths: list[int] = []

        for sidx in allowed_sessions.tolist():
            slen = int(self._session_length[sidx])
            if slen < self.window_length:
                continue
            block_session_idx.append(sidx)
            block_start.append(0)
            block_end.append(slen)
            n = (slen - self.window_length) // self.stride + 1
            block_lengths.append(int(n))

        self._block_session_idx = np.asarray(block_session_idx, dtype=np.int32)
        self._block_start = np.asarray(block_start, dtype=np.int64)
        self._block_end = np.asarray(block_end, dtype=np.int64)
        self._block_cumsum = np.cumsum(np.asarray([0] + block_lengths, dtype=np.int64))

    def __len__(self) -> int:
        return int(self._block_cumsum[-1])

    def _get_keystrokes(self, session_idx: int) -> tuple[np.ndarray, list[str]]:
        cached = self._keystroke_cache.get(session_idx)
        if cached is not None:
            return cached

        offset = int(self._keystroke_offset[session_idx])
        length = int(self._keystroke_length[session_idx])
        if length <= 0:
            starts = np.asarray([], dtype=np.float64)
            keys: list[str] = []
        else:
            starts = np.asarray(
                self._keystroke_start[offset : offset + length], dtype=np.float64
            )
            keys = _decode_bytes(
                np.asarray(self._keystroke_key[offset : offset + length])
            )
        self._keystroke_cache.put(session_idx, (starts, keys))
        return starts, keys

    def _get_prompts(self, session_idx: int) -> tuple[np.ndarray, list[str]]:
        cached = self._prompt_cache.get(session_idx)
        if cached is not None:
            return cached

        offset = int(self._prompt_offset[session_idx])
        length = int(self._prompt_length[session_idx])
        if length <= 0:
            starts = np.asarray([], dtype=np.float64)
            texts: list[str] = []
        else:
            starts = np.asarray(
                self._prompt_start[offset : offset + length], dtype=np.float64
            )
            texts = _decode_bytes(
                np.asarray(self._prompt_text[offset : offset + length])
            )
        self._prompt_cache.put(session_idx, (starts, texts))
        return starts, texts

    def _labels_from_keystrokes(
        self, starts: np.ndarray, keys: list[str], start_t: float, end_t: float
    ) -> np.ndarray:
        if starts.size == 0:
            return np.asarray([], dtype=np.int32)
        i0 = int(np.searchsorted(starts, start_t, side="left"))
        i1 = int(np.searchsorted(starts, end_t, side="right"))
        if i1 <= i0:
            return np.asarray([], dtype=np.int32)
        subset = keys[i0:i1]
        normalized = _CHARSET._normalize_keys(list(subset))
        labels = [
            _CHARSET.key_to_label(k) for k in normalized if k in _CHARSET
        ]
        return np.asarray(labels, dtype=np.int32)

    def _labels_from_prompts(
        self, starts: np.ndarray, texts: list[str], start_t: float, end_t: float
    ) -> np.ndarray:
        if starts.size == 0:
            return np.asarray([], dtype=np.int32)
        i0 = int(np.searchsorted(starts, start_t, side="left"))
        i1 = int(np.searchsorted(starts, end_t, side="right"))
        if i1 <= i0:
            return np.asarray([], dtype=np.int32)
        prompt_texts = texts[i0:i1]
        out = ""
        for text in prompt_texts:
            cleaned = _CHARSET.clean_str(text)
            if len(cleaned) == 0 or cleaned[-1] != "⏎":
                cleaned += "⏎"
            out += cleaned
        return np.asarray(_CHARSET.str_to_labels(out), dtype=np.int32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        bi = int(np.searchsorted(self._block_cumsum, idx, side="right") - 1)
        si = int(self._block_session_idx[bi])
        start_idx = int(self._block_start[bi])
        end_idx = int(self._block_end[bi])
        rel = int(idx - self._block_cumsum[bi])

        offset_local = start_idx + rel * self.stride
        leftover = end_idx - (offset_local + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset_local += np.random.randint(0, min(self.stride, leftover))

        session_start = int(self._session_start[si])
        session_end = int(self._session_end[si])

        window_start_global = max(
            session_start + offset_local - self.left_padding, session_start
        )
        window_end_global = min(
            session_start + offset_local + self.window_length + self.right_padding,
            session_end,
        )
        window_start_local = window_start_global - session_start
        window_end_local = window_end_global - session_start

        mm = self._memmaps

        # Load emg_left
        if "emg_left" in mm:
            emg_left = np.array(mm["emg_left"][window_start_global:window_end_global])
        else:
            emg_left = np.asarray(
                self._get_root()["emg_left"][window_start_global:window_end_global],
                dtype=np.float32,
            )

        # Load emg_right
        if "emg_right" in mm:
            emg_right = np.array(mm["emg_right"][window_start_global:window_end_global])
        else:
            emg_right = np.asarray(
                self._get_root()["emg_right"][window_start_global:window_end_global],
                dtype=np.float32,
            )

        # Load time
        if "time" in mm:
            time = np.array(mm["time"][window_start_global:window_end_global])
        else:
            time = np.asarray(
                self._get_root()["time"][window_start_global:window_end_global],
                dtype=np.float64,
            )

        if self.transform is None:
            emg = torch.as_tensor(np.concatenate([emg_left, emg_right], axis=-1))
        else:
            dtype = np.dtype(
                [
                    ("emg_right", emg_right.dtype, (emg_right.shape[1],)),
                    ("time", time.dtype),
                    ("emg_left", emg_left.dtype, (emg_left.shape[1],)),
                ]
            )
            structured = np.empty((emg_left.shape[0],), dtype=dtype)
            structured["emg_right"] = emg_right
            structured["time"] = time
            structured["emg_left"] = emg_left
            emg = self.transform(structured)
            if torch.is_tensor(emg) and emg.ndim == 3 and emg.shape[1] == 2:
                emg = emg.reshape(emg.shape[0], -1)
            if not torch.is_tensor(emg):
                emg = torch.as_tensor(emg)

        emg = _select_random_side(emg)

        start_t = time[offset_local - window_start_local]
        end_t = time[(offset_local + self.window_length - 1) - window_start_local]

        condition = self._conditions[int(self._session_condition_id[si])]
        if condition == "on_keyboard":
            ks_start, ks_keys = self._get_keystrokes(si)
            labels = self._labels_from_keystrokes(ks_start, ks_keys, start_t, end_t)
        else:
            pr_start, pr_text = self._get_prompts(si)
            labels = self._labels_from_prompts(pr_start, pr_text, start_t, end_t)

        # Fields:
        # emg: EMG window (32, T). sEMG-RD, 16 electrodes per wrist, 2 kHz,
        #   concatenated left(16)+right(16).
        # target_keystrokes: variable-length keystroke labels (N,), derived from
        #   key-down events (key-down/up timestamps are recorded); labels index
        #   CharacterSet.allowed_keys.
        # window_start_idx/window_end_idx: window indices within session.
        # session_idx: integer index into sessions table.
        # user: subject identifier string.
        # condition: session condition string (controls label source).
        return {
            "emg": emg.T,
            "target_keystrokes": torch.as_tensor(labels, dtype=torch.int32),
            "window_start_idx": int(window_start_local),
            "window_end_idx": int(window_end_local),
            "session_idx": si,
            "user": self._users[int(self._session_user_id[si])],
            "condition": condition,
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
        description="Quick smoke test for Emg2QwertyDataset (Zarr)."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Path to the Zarr dataset root.",
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
    parser.add_argument(
        "--allowed-sessions",
        nargs="*",
        default=None,
        help="Session IDs to include.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    dataset = Emg2QwertyDataset(
        root_dir=args.root_dir,
        window_length=args.window_length,
        stride=args.stride,
        padding=args.padding,
        jitter=args.jitter,
        allowed_sessions=args.allowed_sessions,
    )

    print(f"Sessions: {len(dataset._session_id)}")
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
