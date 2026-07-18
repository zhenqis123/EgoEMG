"""Model loading and real-time inference pipeline.

Loads EMGFormer from a Lightning checkpoint, auto-detects the architecture
(16ch EgoEMG vs 8ch Manus), and provides a unified predict() method that
runs the full pipeline: filter → channel map → normalize → model forward.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from emg2pose.datasets.layout_utils import place_sparse_channels
from emg2pose.lightning import EmgPredictionModule

from .filter import RealtimeFFTFilter

# Hardware channel → 16-channel ring position (0-based).
# From emg2pose_channel_indices: [10, 12, 0, 1, 2, 4, 5, 6]
_HW_TO_16CH = np.array([10, 12, 0, 1, 2, 4, 5, 6])


def _build_ring_interp_matrix(target_positions: np.ndarray, target_channels: int) -> np.ndarray:
    """Build sparse ring interpolation weight matrix.
    Identical to emg2pose.datasets.layout_utils.get_sparse_ring_interp_matrix.
    Returns weights of shape (K, target_channels) where K = len(target_positions).
    """
    pos = np.asarray(target_positions, dtype=np.int64)
    order = np.argsort(pos)
    sorted_pos = pos[order]
    k = len(sorted_pos)
    weights_sorted = np.zeros((k, target_channels), dtype=np.float32)

    for out_idx in range(target_channels):
        if out_idx in sorted_pos:
            anchor_idx = int(np.where(sorted_pos == out_idx)[0][0])
            weights_sorted[anchor_idx, out_idx] = 1.0
            continue
        insert = int(np.searchsorted(sorted_pos, out_idx, side="right"))
        right_i = insert % k
        left_i = (insert - 1) % k
        left_pos = int(sorted_pos[left_i])
        right_pos = int(sorted_pos[right_i])
        gap = (right_pos - left_pos) % target_channels
        if gap == 0:
            weights_sorted[left_i, out_idx] = 1.0
            continue
        d_left = (out_idx - left_pos) % target_channels
        d_right = (right_pos - out_idx) % target_channels
        weights_sorted[left_i, out_idx] = np.float32(d_right / gap)
        weights_sorted[right_i, out_idx] = np.float32(d_left / gap)

    inv_order = np.argsort(order)
    return weights_sorted[inv_order]


# Pre-compute the interpolation matrix once: (8, 16)
_RING_INTERP_MATRIX = _build_ring_interp_matrix(_HW_TO_16CH, 16)


@dataclass
class ModelConfig:
    variant: str  # "egoemg_16ch" or "manus_8ch"
    in_channels: int
    out_channels: int
    norm_mean: float
    norm_std: float
    window_length: int
    sample_rate: float
    channel_positions: np.ndarray | None  # 0-based ring positions for 8ch→16ch
    channel_interpolate: bool = True  # True: ring interp; False: zero-pad (place_sparse)


class ModelLoader:
    """Load EMGFormer from a Lightning checkpoint, auto-detect architecture."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        channel_interpolate: bool | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._channel_interpolate_override = channel_interpolate

    def load(self) -> tuple[nn.Module, ModelConfig]:
        """Load model and return (model, config)."""
        ckpt = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        hparams = ckpt["hyper_parameters"]

        module = EmgPredictionModule.load_from_checkpoint(
            self.checkpoint_path,
            module_conf=hparams["module_conf"],
            optimizer_conf=hparams["optimizer_conf"],
            lr_scheduler_conf=hparams["lr_scheduler_conf"],
            loss_weights=hparams["loss_weights"],
            datamodule=hparams.get("datamodule"),
            map_location="cpu",
        )
        module.eval()
        module.to(self.device)

        # Detect architecture variant
        featurizer = module.model.featurizer
        in_ch = featurizer.layers[0].conv[0].in_channels
        variant = "egoemg_16ch" if in_ch >= 16 else "manus_8ch"
        out_ch = hparams["module_conf"].get("out_channels", 22)

        # Resolve normalization stats
        norm_mean, norm_std = self._resolve_norm_stats(hparams, variant)

        # Resolve window length from training config
        dm = hparams.get("datamodule") or {}
        window_length = dm.get("window_length", 2000)
        sample_rate = 2000.0

        # Detect channel_interpolate setting from training config
        if self._channel_interpolate_override is not None:
            channel_interpolate = self._channel_interpolate_override
        else:
            channel_interpolate = self._detect_channel_interpolate(hparams)

        config = ModelConfig(
            variant=variant,
            in_channels=in_ch,
            out_channels=out_ch,
            norm_mean=norm_mean,
            norm_std=norm_std,
            window_length=window_length,
            sample_rate=sample_rate,
            channel_positions=_HW_TO_16CH if variant == "egoemg_16ch" else None,
            channel_interpolate=channel_interpolate,
        )

        return module.model, config

    def _detect_channel_interpolate(self, hparams: dict) -> bool:
        """Detect whether training used channel_interpolate=True or False.

        Checks dataset configs in hparams. Default True (ring interpolation).
        If any dataset split has channel_interpolate=false, returns False.

        Heuristic fallback: Manus fine-tuned models (from EgoEMG pretrain)
        use channel_interpolate=false with 16ch input and 22ch output.
        """
        dm = hparams.get("datamodule") or {}

        # Direct datamodule setting
        if "channel_interpolate" in dm:
            return bool(dm["channel_interpolate"])

        # Check inline dataset configs (used by regression_manus_from_pretrain)
        for key in ("train_datasets", "val_datasets", "test_datasets",
                     "dataset_train", "dataset_val", "dataset_test"):
            datasets = dm.get(key)
            if datasets is None:
                continue
            if isinstance(datasets, list):
                for ds_conf in datasets:
                    if isinstance(ds_conf, dict) and "channel_interpolate" in ds_conf:
                        return bool(ds_conf["channel_interpolate"])
            elif isinstance(datasets, dict) and "channel_interpolate" in datasets:
                return bool(datasets["channel_interpolate"])

        # Check nested dataset config (Hydra may store it differently)
        dataset_conf = hparams.get("dataset")
        if isinstance(dataset_conf, dict):
            for split_key in ("train", "val", "test"):
                split_conf = dataset_conf.get(split_key)
                if isinstance(split_conf, list):
                    for ds_conf in split_conf:
                        if isinstance(ds_conf, dict) and "channel_interpolate" in ds_conf:
                            return bool(ds_conf["channel_interpolate"])

        # Heuristic: Manus fine-tuned models from EgoEMG pretrain use
        # channel_interpolate=false. They have pretrained_checkpoint set,
        # 16ch featurizer input (emg2pose_interpolate16 layout),
        # and ignore_head_tail_dims=2 (wrist masking for 20-dim hand angles).
        pretrained = hparams.get("pretrained_checkpoint")
        ignore_dims = hparams.get("ignore_head_tail_dims", 0)
        if pretrained and ignore_dims == 2:
            return False

        return True  # default: ring interpolation

    def _resolve_norm_stats(
        self, hparams: dict, variant: str
    ) -> tuple[float, float]:
        """Resolve normalization mean/std from config or stats file."""
        dm = hparams.get("datamodule") or {}
        stats_path = dm.get("per_dataset_norm_stats_path")
        norm_mode = dm.get("norm_mode")

        # Default stats
        if variant == "egoemg_16ch":
            default_mean, default_std = 0.00007, 3.173
        else:
            default_mean, default_std = 0.0, 2.245

        if norm_mode != "per-dataset" or not stats_path:
            return default_mean, default_std

        stats_file = Path(stats_path)
        if not stats_file.is_absolute():
            stats_file = _PROJECT_ROOT / stats_file

        if not stats_file.exists():
            return default_mean, default_std

        with open(stats_file) as f:
            stats = json.load(f)

        # Try dataset name from hparams, fallback to variant default
        dataset_name = "egoemg" if variant == "egoemg_16ch" else "manus"
        entry = stats.get(dataset_name)
        if entry is None:
            return default_mean, default_std

        return float(entry["mean"]), float(entry["std"])


class InferenceEngine:
    """Full real-time inference pipeline: filter → map → normalize → model."""

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: torch.device,
        compute_landmarks: bool = False,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.compute_landmarks = compute_landmarks

        # FFT filter (matches training pipeline)
        self.fft_filter = RealtimeFFTFilter(
            window_length=config.window_length,
            fs=config.sample_rate,
        )

        # Optional: UmeTrack FK for landmark computation
        self._hand_model = None
        if compute_landmarks:
            self._init_fk()

        # Overlap-save filter padding: accumulate extra samples on the left
        # to absorb FFT circular convolution boundary effects.
        # Larger padding gives better match to full-session FFT filtering.
        self._filter_pad = min(config.window_length, 8000)  # up to 4s of padding
        self._filter_buf = np.zeros(
            (0, 8), dtype=np.float32
        )  # history buffer for padding
        self._fft_filter_padded = RealtimeFFTFilter(
            window_length=config.window_length + self._filter_pad,
            fs=config.sample_rate,
        )

        # Debug counter: print pipeline stats for first N predictions
        self._debug_count = 0
        self._debug_max = 5

    def _init_fk(self) -> None:
        """Initialize UmeTrack hand model for FK landmark computation."""
        try:
            from emg2pose.kinematics import (
                apply_to_hand_model,
                load_default_hand_model,
            )

            self._hand_model = load_default_hand_model()
            self._hand_model = apply_to_hand_model(
                self._hand_model, lambda t: t.float().to(self.device)
            )
        except Exception as e:
            print(f"Warning: Could not load UmeTrack hand model: {e}")
            print("Landmark computation will be disabled.")
            self.compute_landmarks = False

    @torch.no_grad()
    def predict(
        self, raw_window: np.ndarray, raw_is_mv: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the full prediction pipeline on a raw EMG window.

        Args:
            raw_window: (window_length, 8) float32 raw EMG samples.
            raw_is_mv: If True, input is already in mV (skip µV→mV conversion).
                Use this when feeding pre-converted data (e.g. from memmap).

        Returns:
            (angles, landmarks) where:
              - angles: (out_channels,) float array of joint angles in radians
              - landmarks: (21, 3) float array or None if FK is disabled
        """
        debug = self._debug_count < self._debug_max

        # 1. Convert µV → mV (matches build_manus_memmap.py line 115)
        if raw_is_mv:
            raw_window_mv = raw_window.astype(np.float32, copy=False)
        else:
            raw_window_mv = raw_window / 1000.0
        if debug:
            print(
                f"[debug pred #{self._debug_count}] "
                f"raw_input: mean={raw_window.mean():.1f}, std={raw_window.std():.1f}, "
                f"min={raw_window.min():.1f}, max={raw_window.max():.1f}",
                flush=True,
            )
            print(
                f"  after µV→mV: mean={raw_window_mv.mean():.4f}, "
                f"std={raw_window_mv.std():.4f}, "
                f"min={raw_window_mv.min():.4f}, max={raw_window_mv.max():.4f}",
                flush=True,
            )

        # 2. FFT filter with overlap-save padding to reduce boundary effects.
        #    Prepend filter_pad historical samples, filter the padded window,
        #    then extract the last window_length samples (clean portion).
        self._filter_buf = np.concatenate(
            [self._filter_buf, raw_window_mv], axis=0
        )
        # Keep only what we need: pad + window_length
        needed = self._filter_pad + self.config.window_length
        if self._filter_buf.shape[0] > needed:
            self._filter_buf = self._filter_buf[-needed:]

        if self._filter_buf.shape[0] >= needed:
            padded = self._filter_buf  # (pad + window_length, 8)
            filtered_padded = self._fft_filter_padded.filter(padded)
            filtered = filtered_padded[-self.config.window_length:]
        else:
            # Not enough history yet, fall back to per-window filter
            filtered = self.fft_filter.filter(raw_window_mv)

        if debug:
            print(
                f"  after filter: mean={filtered.mean():.4f}, "
                f"std={filtered.std():.4f}, "
                f"min={filtered.min():.4f}, max={filtered.max():.4f}"
                + (f"  [overlap-save, pad={self._filter_pad}]"
                   if self._filter_buf.shape[0] >= needed else "  [fallback]"),
                flush=True,
            )

        # 3. Channel mapping
        emg_input = self._map_channels(filtered)  # (T, in_channels)
        if debug:
            print(
                f"  after ch_map: shape={emg_input.shape}, "
                f"mean={emg_input.mean():.4f}, std={emg_input.std():.4f}",
                flush=True,
            )

        # 4. Transpose to model input shape: (1, C, T)
        tensor_input = torch.from_numpy(emg_input.T).unsqueeze(0).to(self.device)

        # 5. Normalize (per-dataset stats on mV + filtered data)
        tensor_input = (tensor_input - self.config.norm_mean) / (
            self.config.norm_std + 1e-6
        )
        if debug:
            print(
                f"  after norm: mean={tensor_input.mean():.4f}, "
                f"std={tensor_input.std():.4f}",
                flush=True,
            )

        # 6. Model forward
        batch = {"emg": tensor_input}
        output = self.model(batch)

        # 7. Extract predictions
        if isinstance(output, tuple):
            output = output[0]
        # output shape: (1, out_channels, T_pred)
        # Use the last time step — it corresponds to the most recent EMG
        # data in the window, minimizing effective latency.
        angles = output[0, :, -1].cpu().numpy()
        if debug:
            print(
                f"  output: shape={output.shape}, "
                f"angles[0:5]={angles[:5].tolist()}",
                flush=True,
            )
            print(
                f"  angles[rad]: mean={angles.mean():.4f}, "
                f"std={angles.std():.4f}, "
                f"min={angles.min():.4f}, max={angles.max():.4f}",
                flush=True,
            )
            print(
                f"  angles[deg]: mean={np.degrees(angles).mean():.1f}, "
                f"std={np.degrees(angles).std():.1f}",
                flush=True,
            )
            self._debug_count += 1

        # 8. Optional FK landmarks
        landmarks = None
        if self.compute_landmarks and self._hand_model is not None:
            landmarks = self._compute_landmarks(angles)

        return angles, landmarks

    def _map_channels(self, filtered: np.ndarray) -> np.ndarray:
        """Map 8 hardware channels to the model's expected input layout.

        When channel_interpolate=True: sparse ring interpolation (default).
        When channel_interpolate=False: zero-fill placement (matches training
        that used channel_interpolate=false, e.g. Manus fine-tuning).
        """
        if self.config.variant == "egoemg_16ch":
            if self.config.channel_interpolate:
                # (T, 8) @ (8, 16) → (T, 16)
                return filtered.astype(np.float32, copy=False) @ _RING_INTERP_MATRIX
            else:
                # Zero-fill: place 8 channels at their ring positions
                return place_sparse_channels(
                    filtered, 16, _HW_TO_16CH
                )
        else:
            # Direct 8-channel input
            return filtered.astype(np.float32, copy=False)

    def _compute_landmarks(self, angles: np.ndarray) -> np.ndarray:
        """Compute 21 hand keypoints from joint angles via UmeTrack FK."""
        from emg2pose.UmeTrack.lib.common.hand_skinning import skin_landmarks
        from emg2pose.kinematics import apply_to_hand_model, broadcast_hand_model_to

        hm = broadcast_hand_model_to(self._hand_model, (1,))
        hm = apply_to_hand_model(hm, lambda t: t.float())

        wrist_tf = torch.eye(4, device=self.device).unsqueeze(0)

        # Pad to 22D if needed (model may output 20 or 22)
        if len(angles) < 22:
            angles_22 = np.concatenate([angles, np.zeros(22 - len(angles))])
        else:
            angles_22 = angles

        a = torch.from_numpy(angles_22).float().to(self.device).reshape(1, -1)
        landmarks = skin_landmarks(hm, a[:, :20], wrist_tf)
        return landmarks[0].cpu().numpy()  # (21, 3)
