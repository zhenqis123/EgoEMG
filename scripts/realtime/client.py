"""ZMQ streaming client for sending EMG data and receiving predictions.

Runs in a background thread. Sends batched raw EMG samples via PUSH socket,
subscribes to prediction results via SUB socket, and supports control
commands (ping/stop) via REQ socket.
"""

from __future__ import annotations

import queue
import time
from collections import deque
from threading import Event, Thread

import numpy as np
import zmq

from .protocol import encode_upstream


class ZmqStreamClient(Thread):
    """Background thread that streams EMG to server and collects predictions.

    Args:
        server_host: Server hostname or IP.
        recv_port: Server's PULL port (client PUSH to this).
        send_port: Server's PUB port (client SUB to this).
        ctrl_port: Server's REP port (client REQ to this).
        batch_size: Number of samples per upstream frame (default 40 = 20ms).
    """

    def __init__(
        self,
        server_host: str = "localhost",
        recv_port: int = 5555,
        send_port: int = 5556,
        ctrl_port: int = 5557,
        batch_size: int = 40,
    ):
        super().__init__(daemon=True)
        self.server_host = server_host
        self.recv_port = recv_port
        self.send_port = send_port
        self.ctrl_port = ctrl_port
        self.batch_size = batch_size

        self._sample_queue: queue.Queue = queue.Queue(maxsize=50000)
        self._predictions: deque[dict] = deque(maxlen=1000)
        self._running = Event()
        self._running.set()

        # Status
        self.connected = False
        self.server_info: dict = {}
        self.samples_sent = 0
        self.predictions_received = 0

    def enqueue_sample(self, timestamp: float, emg: list[int | float]) -> None:
        """Called by SerialCollector for each EMG sample.

        Args:
            timestamp: time.time() timestamp.
            emg: List of 8 channel values.
        """
        try:
            self._sample_queue.put_nowait((timestamp, emg))
        except queue.Full:
            pass  # drop sample — backpressure protection

    def get_predictions(self) -> list[dict]:
        """Drain and return all buffered predictions."""
        preds = list(self._predictions)
        self._predictions.clear()
        return preds

    def get_latest_prediction(self) -> dict | None:
        """Return the most recent prediction and clear older ones."""
        if not self._predictions:
            return None
        # Keep only the latest, discard older buffered predictions
        latest = self._predictions[-1]
        self._predictions.clear()
        return latest

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        ctx = zmq.Context()

        # PUSH: send samples to server
        push = ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.SNDHWM, 1000)
        push.setsockopt(zmq.LINGER, 0)
        push.connect(f"tcp://{self.server_host}:{self.recv_port}")

        # SUB: receive predictions
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVHWM, 100)
        sub.connect(f"tcp://{self.server_host}:{self.send_port}")

        # REQ: control
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 3000)
        req.setsockopt(zmq.SNDTIMEO, 3000)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(f"tcp://{self.server_host}:{self.ctrl_port}")

        # Verify connection via ping
        try:
            req.send_json({"cmd": "ping"})
            resp = req.recv_json()
            self.connected = resp.get("status") == "ok"
            self.server_info = resp
        except zmq.Again:
            self.connected = False

        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)

        batch: list[tuple[float, list]] = []
        seq = 0

        try:
            while self._running.is_set():
                # Drain sample queue into batch
                try:
                    while len(batch) < self.batch_size:
                        ts, emg = self._sample_queue.get(timeout=0.005)
                        batch.append((ts, emg))
                except queue.Empty:
                    pass

                # Send batch if full
                if len(batch) >= self.batch_size:
                    ts_ns = int(batch[-1][0] * 1e9)
                    samples = np.array(
                        [s[1] for s in batch], dtype=np.float32
                    )
                    frame = encode_upstream(seq, ts_ns, samples)
                    try:
                        push.send(frame, zmq.NOBLOCK)
                        self.samples_sent += len(batch)
                    except zmq.Again:
                        pass  # server not keeping up
                    batch.clear()
                    seq += 1

                # Check for predictions (non-blocking)
                events = dict(poller.poll(timeout=1))
                if sub in events:
                    try:
                        pred = sub.recv_json(flags=zmq.NOBLOCK)
                        self._predictions.append(pred)
                        self.predictions_received += 1
                    except zmq.Again:
                        pass

        finally:
            push.close()
            sub.close()
            req.close()
            ctx.term()
