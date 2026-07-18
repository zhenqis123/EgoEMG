"""ZMQ-based real-time EMG inference server.

Receives raw EMG samples via PUSH socket, runs the inference pipeline
(filter → channel map → normalize → model forward → FK), and publishes
predictions via PUB socket.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import zmq

from .buffer import SlidingWindowBuffer
from .inference import InferenceEngine, ModelConfig, ModelLoader
from .protocol import decode_upstream, encode_downstream


@dataclass
class ServerConfig:
    checkpoint_path: str
    device: str = "cuda"
    recv_port: int = 5555  # PULL: receives EMG samples
    send_port: int = 5556  # PUB: publishes predictions
    ctrl_port: int = 5557  # REP: control commands
    stride: int = 400  # samples between predictions (200ms @ 2kHz)
    compute_landmarks: bool = True
    channel_interpolate: bool | None = None  # None=auto-detect, True=ring, False=zero-pad


class InferenceServer:
    """ZMQ-based real-time EMG inference server."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.engine: InferenceEngine | None = None
        self.model_config: ModelConfig | None = None
        self.buffer: SlidingWindowBuffer | None = None
        self._running = False

    def setup(self) -> None:
        """Load model, create filter/buffer, bind ZMQ sockets."""
        print(f"Loading model from: {self.config.checkpoint_path}")
        loader = ModelLoader(
            self.config.checkpoint_path,
            self.config.device,
            channel_interpolate=self.config.channel_interpolate,
        )
        model, self.model_config = loader.load()

        device = loader.device
        print(f"  Variant: {self.model_config.variant}")
        print(f"  Input channels: {self.model_config.in_channels}")
        print(f"  Output channels: {self.model_config.out_channels}")
        print(f"  Window length: {self.model_config.window_length}")
        print(f"  Norm: mean={self.model_config.norm_mean}, std={self.model_config.norm_std}")
        print(f"  Channel interpolate: {self.model_config.channel_interpolate}")
        print(f"  Device: {device}")

        self.engine = InferenceEngine(
            model=model,
            config=self.model_config,
            device=device,
            compute_landmarks=self.config.compute_landmarks,
        )

        self.buffer = SlidingWindowBuffer(
            window_length=self.model_config.window_length,
            stride=self.config.stride,
            n_channels=8,
        )

    def run(self) -> None:
        """Main event loop: receive EMG, run inference, publish predictions."""
        if self.engine is None:
            raise RuntimeError("Call setup() before run()")

        ctx = zmq.Context()

        recv_sock = ctx.socket(zmq.PULL)
        recv_sock.setsockopt(zmq.RCVHWM, 5000)
        recv_sock.bind(f"tcp://*:{self.config.recv_port}")

        send_sock = ctx.socket(zmq.PUB)
        send_sock.setsockopt(zmq.SNDHWM, 100)
        send_sock.bind(f"tcp://*:{self.config.send_port}")

        ctrl_sock = ctx.socket(zmq.REP)
        ctrl_sock.setsockopt(zmq.RCVTIMEO, 50)
        ctrl_sock.bind(f"tcp://*:{self.config.ctrl_port}")

        print(f"\nServer listening:")
        print(f"  RECV (PULL): tcp://*:{self.config.recv_port}")
        print(f"  SEND (PUB):  tcp://*:{self.config.send_port}")
        print(f"  CTRL (REP):  tcp://*:{self.config.ctrl_port}")
        print(f"  Stride: {self.config.stride} samples ({self.config.stride / 2000 * 1000:.0f}ms)")
        print(f"\nWaiting for EMG data... (Ctrl+C to stop)\n")

        self._running = True
        poller = zmq.Poller()
        poller.register(recv_sock, zmq.POLLIN)
        poller.register(ctrl_sock, zmq.POLLIN)

        pred_seq = 0
        total_samples = 0
        total_predictions = 0
        t_start = time.monotonic()
        last_status = t_start

        try:
            while self._running:
                events = dict(poller.poll(timeout=100))

                # Handle control messages
                if ctrl_sock in events:
                    try:
                        msg = ctrl_sock.recv_json(flags=zmq.NOBLOCK)
                        cmd = msg.get("cmd")
                        if cmd == "ping":
                            ctrl_sock.send_json({
                                "status": "ok",
                                "variant": self.model_config.variant,
                                "window_length": self.model_config.window_length,
                                "out_channels": self.model_config.out_channels,
                            })
                        elif cmd == "stop":
                            ctrl_sock.send_json({"status": "stopping"})
                            self._running = False
                        else:
                            ctrl_sock.send_json({"status": "unknown_cmd"})
                    except zmq.Again:
                        pass

                # Handle incoming EMG data
                if recv_sock in events:
                    try:
                        frame = recv_sock.recv(flags=zmq.NOBLOCK)
                        seq, ts_ns, samples = decode_upstream(frame)
                        self.buffer.push(samples)
                        total_samples += samples.shape[0]

                        # Check if we have complete windows to predict
                        while self.buffer.has_window():
                            t0 = time.perf_counter()
                            raw_window = self.buffer.get_window()
                            angles, landmarks = self.engine.predict(raw_window)
                            inference_ms = (time.perf_counter() - t0) * 1000

                            pred = encode_downstream(
                                seq=pred_seq,
                                timestamp_ns=ts_ns,
                                angles=angles,
                                landmarks=landmarks,
                                inference_ms=inference_ms,
                            )
                            send_sock.send_json(pred)
                            pred_seq += 1
                            total_predictions += 1

                    except zmq.Again:
                        pass

                # Periodic status
                now = time.monotonic()
                if now - last_status > 5.0:
                    elapsed = now - t_start
                    fps = total_predictions / max(elapsed, 0.001)
                    sample_rate = total_samples / max(elapsed, 0.001)
                    print(
                        f"  Samples: {total_samples:>10d} | "
                        f"Predictions: {total_predictions:>6d} | "
                        f"Rate: {sample_rate:.0f} Hz | "
                        f"Pred/s: {fps:.1f}"
                    )
                    last_status = now

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            elapsed = time.monotonic() - t_start
            print(f"\nSession summary:")
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Total samples: {total_samples}")
            print(f"  Total predictions: {total_predictions}")
            if elapsed > 0:
                print(f"  Avg sample rate: {total_samples / elapsed:.0f} Hz")
                print(f"  Avg pred rate: {total_predictions / elapsed:.1f} pred/s")

            recv_sock.close()
            send_sock.close()
            ctrl_sock.close()
            ctx.term()
