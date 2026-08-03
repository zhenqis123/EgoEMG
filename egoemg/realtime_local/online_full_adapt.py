from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np
import torch
import torch.nn.functional as F

from egoemg.realtime_local.pipeline import LocalSmallStreamer, Prediction
from egoemg.realtime_local.small_model import prepare_small_input


@dataclass(frozen=True)
class OnlineAdaptStatus:
    teacher_count: int
    sample_count: int
    step_count: int
    last_loss: float | None
    last_teacher_age_s: float | None


@dataclass(frozen=True)
class _TeacherFrame:
    timestamp: float
    angles20: np.ndarray


@dataclass(frozen=True)
class _TrainSample:
    window8: np.ndarray
    teacher20: np.ndarray
    base20: np.ndarray
    timestamp: float


class TeacherBuffer:
    """Small timestamped buffer for camera-derived teacher poses."""

    def __init__(self, maxlen: int = 120) -> None:
        self._frames: Deque[_TeacherFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, timestamp: float, angles20: np.ndarray) -> None:
        angles = np.asarray(angles20, dtype=np.float32).reshape(-1)
        if angles.shape[0] < 20 or not np.isfinite(angles[:20]).all():
            return
        with self._lock:
            self._frames.append(_TeacherFrame(float(timestamp), angles[:20].copy()))

    def nearest(self, timestamp: float, tolerance_s: float) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if not self._frames:
                return None
            frames = list(self._frames)
        best = min(frames, key=lambda item: abs(item.timestamp - timestamp))
        age = abs(best.timestamp - timestamp)
        if age > tolerance_s:
            return None
        return best.angles20.copy(), age

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class FullModelOnlineTrainer:
    """Background full-parameter online updates for the local small EMG model.

    The trainer uses the exact filtered window consumed by realtime inference
    and a timestamp-matched WiLoR/mapper teacher angle vector. It updates all
    trainable model parameters, so conservative defaults are intentional.
    """

    def __init__(
        self,
        streamer: LocalSmallStreamer,
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 4,
        min_samples: int = 8,
        buffer_size: int = 256,
        match_tolerance_s: float = 0.12,
        update_interval_s: float = 1.0,
        steps_per_update: int = 1,
        grad_clip: float = 1.0,
        keep_weight: float = 0.05,
    ) -> None:
        self.streamer = streamer
        self.device = streamer.device
        self.batch_size = int(batch_size)
        self.min_samples = int(min_samples)
        self.match_tolerance_s = float(match_tolerance_s)
        self.update_interval_s = float(update_interval_s)
        self.steps_per_update = int(steps_per_update)
        self.grad_clip = float(grad_clip)
        self.keep_weight = float(keep_weight)

        params = [p for p in self.streamer.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
        self.teacher = TeacherBuffer()
        self._samples: Deque[_TrainSample] = deque(maxlen=int(buffer_size))
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._step_count = 0
        self._last_loss: float | None = None
        self._last_teacher_age_s: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="emg-online-full-adapt", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def add_teacher(self, timestamp: float, angles20: np.ndarray) -> None:
        self.teacher.add(timestamp, angles20)

    def observe_prediction(self, filtered_window_8ch: np.ndarray, pred: Prediction) -> None:
        match = self.teacher.nearest(pred.timestamp, self.match_tolerance_s)
        if match is None:
            return
        teacher20, age_s = match
        base20 = np.asarray(pred.angles[:20], dtype=np.float32).copy()
        sample = _TrainSample(
            window8=np.asarray(filtered_window_8ch, dtype=np.float32).copy(),
            teacher20=teacher20,
            base20=base20,
            timestamp=float(pred.timestamp),
        )
        with self._samples_lock:
            self._samples.append(sample)
        self._last_teacher_age_s = float(age_s)
        self._wake.set()

    def status(self) -> OnlineAdaptStatus:
        with self._samples_lock:
            sample_count = len(self._samples)
        return OnlineAdaptStatus(
            teacher_count=len(self.teacher),
            sample_count=sample_count,
            step_count=self._step_count,
            last_loss=self._last_loss,
            last_teacher_age_s=self._last_teacher_age_s,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.streamer.model.state_dict(),
            "online_adapt": {
                "step_count": self._step_count,
                "last_loss": self._last_loss,
                "last_teacher_age_s": self._last_teacher_age_s,
            },
        }
        torch.save(payload, Path(path))

    def _loop(self) -> None:
        last_update = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop.is_set():
                break
            now = time.monotonic()
            if now - last_update < self.update_interval_s:
                continue
            with self._samples_lock:
                enough = len(self._samples) >= self.min_samples
            if not enough:
                continue
            for _ in range(max(1, self.steps_per_update)):
                if self._stop.is_set():
                    break
                self._train_step()
            last_update = time.monotonic()

    def _sample_batch(self) -> list[_TrainSample]:
        with self._samples_lock:
            items = list(self._samples)
        if len(items) <= self.batch_size:
            return items
        return random.sample(items, self.batch_size)

    def _train_step(self) -> None:
        batch = self._sample_batch()
        if not batch:
            return
        emg_np = np.stack(
            [prepare_small_input(item.window8, self.streamer.model).T for item in batch],
            axis=0,
        ).astype(np.float32)
        teacher_np = np.stack([item.teacher20 for item in batch], axis=0).astype(np.float32)
        base_np = np.stack([item.base20 for item in batch], axis=0).astype(np.float32)

        emg = torch.from_numpy(emg_np).to(self.device)
        teacher = torch.from_numpy(teacher_np).to(self.device)
        base = torch.from_numpy(base_np).to(self.device)
        delay_samples = int(round(self.streamer.output_delay_s * self.streamer.sample_rate_hz))
        target_len = batch[0].window8.shape[0] - self.streamer.left_context
        pred_idx = max(0, target_len - 1 - delay_samples)

        with self.streamer.model_lock, torch.enable_grad():
            self.streamer.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            output = self.streamer.model({"emg": emg})
            if isinstance(output, tuple):
                output = output[0]
            output = F.interpolate(output, size=target_len, mode="linear")
            pred20 = output[:, :20, pred_idx]
            teacher_loss = F.smooth_l1_loss(pred20, teacher)
            if self.keep_weight > 0.0:
                keep_loss = F.mse_loss(pred20, base)
                loss = teacher_loss + self.keep_weight * keep_loss
            else:
                loss = teacher_loss
            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.streamer.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.streamer.model.eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._last_loss = float(loss.detach().cpu().item())
        self._step_count += 1
