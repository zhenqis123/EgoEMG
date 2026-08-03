from __future__ import annotations

import time
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from egoemg.realtime_local.buffer import SlidingWindowBuffer
from egoemg.realtime_local.preprocess import SmallPreprocessor, load_noise_floor
from egoemg.realtime_local.small_model import (
    SMALL_WINDOW_LENGTH,
    load_small_emgformer,
    prepare_small_input,
)


@dataclass(frozen=True)
class Prediction:
    angles: np.ndarray
    timestamp: float
    sample_index: int
    inference_ms: float

    def to_jsonable(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "sample_index": self.sample_index,
            "inference_ms": self.inference_ms,
            "angles": self.angles.astype(float).tolist(),
        }


class LocalSmallStreamer:
    """In-process streaming inference engine for the Incre small model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        stride_samples: int = 200,
        device: str = "cuda",
        noise_floor_path: str | Path | None = None,
        input_scale: float = 1.0,
        remove_sample_mean: bool = False,
        output_delay_s: float = 0.5,
        compile_model: bool = False,
        callback: Callable[[Prediction], None] | None = None,
        window_callback: Callable[[np.ndarray, Prediction], None] | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = load_small_emgformer(checkpoint_path, self.device)
        self.model_lock = threading.RLock()
        self.input_channels = int(self.model.featurizer.layers[0].conv[0].weight.shape[1])
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.buffer = SlidingWindowBuffer(
            window_length=SMALL_WINDOW_LENGTH,
            stride=int(stride_samples),
            n_channels=8,
        )
        self.preprocess = SmallPreprocessor(
            noise_floor=load_noise_floor(noise_floor_path),
            input_scale=input_scale,
            remove_sample_mean=remove_sample_mean,
        )
        self.callback = callback
        self.window_callback = window_callback
        self.sample_rate_hz = 2000.0
        self.left_context = int(getattr(self.model, "left_context", 510))
        self.output_delay_s = float(output_delay_s)

        # One synthetic warmup keeps CUDA kernel setup out of the first live emit.
        with torch.inference_mode():
            dummy = torch.zeros(
                (1, self.input_channels, SMALL_WINDOW_LENGTH),
                device=self.device,
            )
            with self.model_lock:
                _ = self.model({"emg": dummy})
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def push_samples(
        self,
        samples: np.ndarray,
        timestamp: float | None = None,
    ) -> list[Prediction]:
        filtered = self.preprocess(samples)
        self.buffer.push(filtered)
        predictions: list[Prediction] = []
        if not self.buffer.has_window():
            return predictions

        # If acquisition outruns inference, keep the latest window only.
        if self.buffer.ready_count() > 1:
            self.buffer.keep_latest_ready()

        raw_window = self.buffer.get_window()
        pred = self.predict_window(
            raw_window,
            timestamp=time.time() if timestamp is None else timestamp,
            sample_index=self.buffer.total_samples,
        )
        predictions.append(pred)
        if self.window_callback is not None:
            self.window_callback(raw_window.copy(), pred)
        if self.callback is not None:
            self.callback(pred)
        return predictions

    @torch.inference_mode()
    def predict_window(
        self,
        filtered_window_8ch: np.ndarray,
        timestamp: float,
        sample_index: int,
    ) -> Prediction:
        emg_in = prepare_small_input(filtered_window_8ch, self.model)
        tensor = torch.from_numpy(emg_in.T).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with self.model_lock:
            self.model.eval()
            output = self.model({"emg": tensor})
        if isinstance(output, tuple):
            output = output[0]
        target_len = filtered_window_8ch.shape[0] - self.left_context
        if target_len <= 0:
            raise ValueError(
                f"Window is shorter than model left_context: {filtered_window_8ch.shape[0]}"
            )
        output = F.interpolate(output, size=target_len, mode="linear")
        delay_samples = int(round(self.output_delay_s * self.sample_rate_hz))
        pred_idx = max(0, target_len - 1 - delay_samples)
        angles = output[0, :, pred_idx].detach().cpu().numpy().astype(np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(
            angles=angles,
            timestamp=timestamp - self.output_delay_s,
            sample_index=int(sample_index),
            inference_ms=float(inference_ms),
        )
