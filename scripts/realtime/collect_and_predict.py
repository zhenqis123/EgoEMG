#!/usr/bin/env python3
"""EMG data collector with optional real-time inference.

Extends collect.py to stream raw EMG to a remote inference server and
receive joint angle predictions for real-time visualization.

Usage:
  # Real-time inference mode (server must be running)
  python scripts/realtime/collect_and_predict.py \\
    --server 192.168.1.100 --com-port COM3 --out data

  # Pure data collection mode (no server needed)
  python scripts/realtime/collect_and_predict.py --com-port COM3 --out data

  # With Manus glove (same as collect.py)
  python scripts/realtime/collect_and_predict.py \\
    --server 192.168.1.100 --com-port COM3 \\
    --redis-host localhost --out data

Over SSH tunnel:
  ssh -L 5555:localhost:5555 -L 5556:localhost:5556 -L 5557:localhost:5557 user@server
  python scripts/realtime/collect_and_predict.py --server localhost --com-port COM3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import numpy as np

# Add project root to path for collect.py imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# Import from collect.py
from collect import (
    HEADER,
    PKT_LEN,
    TYPE_AA,
    TYPE_BB,
    MAX_BUF,
    ManusCollector,
    make_session_dir,
    fmt_duration,
    FLUSH_INTERVAL_SAMPLES,
    TIMEBASE_INFO,
)

from realtime.client import ZmqStreamClient
from realtime.visualizer import HandVisualizer

# ==================== CONFIG ====================
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"  # Linux; override with --com-port on Windows
DEFAULT_SERIAL_BAUD = 921600
DEFAULT_SERIAL_TIMEOUT = 0.05
STATUS_INTERVAL = 0.25
# ===============================================


class StreamingSerialCollector(Thread):
    """Reads EMG+IMU from serial and optionally streams EMG to ZMQ client.

    Identical to collect.py's SerialCollector but adds sample_callback
    for real-time streaming.
    """

    def __init__(
        self,
        port: str,
        baud: int,
        timeout: float,
        emg_path: str,
        imu_path: str,
        remove_emg_mean: bool = True,
        sample_callback=None,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.emg_path = emg_path
        self.imu_path = imu_path
        self.remove_emg_mean = remove_emg_mean
        self.sample_callback = sample_callback
        self._running = Event()
        self._running.set()

        self.emg_count = 0
        self.imu_count = 0
        self.start_time: float = 0.0
        self.last_activity: float = 0.0
        self.status_msg = "initializing"

    @staticmethod
    def _int24_be_signed(b0: int, b1: int, b2: int) -> int:
        u = (b0 << 16) | (b1 << 8) | b2
        if u & 0x800000:
            u -= 1 << 24
        return u

    @staticmethod
    def _int16_be_signed(b0: int, b1: int) -> int:
        u = (b0 << 8) | b1
        if u & 0x8000:
            u -= 1 << 16
        return u

    def run(self) -> None:
        import serial as serial_mod

        self.start_time = time.time()
        self.last_activity = self.start_time

        ser = self._open_serial(serial_mod)
        if ser is None:
            return

        emg_fh = open(self.emg_path, "w", newline="")
        imu_fh = open(self.imu_path, "w", newline="")
        emg_writer = csv.writer(emg_fh)
        imu_writer = csv.writer(imu_fh)
        emg_writer.writerow(["timestamp", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"])
        imu_writer.writerow(["timestamp", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])

        buf = bytearray()
        retry_attempt = 0

        try:
            while self._running.is_set():
                if buf.find(HEADER) == -1:
                    if len(buf) > MAX_BUF:
                        buf.clear()
                    chunk = ser.read(256)
                    if chunk:
                        buf.extend(chunk)
                        retry_attempt = 0
                    else:
                        retry_attempt += 1
                        ser = self._reconnect(ser, serial_mod, retry_attempt)
                    continue

                header_idx = buf.find(HEADER)
                needed = header_idx + PKT_LEN
                while self._running.is_set() and len(buf) < needed:
                    chunk = ser.read(needed - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)

                if len(buf) < needed:
                    continue

                pkt = bytes(buf[header_idx : header_idx + PKT_LEN])
                del buf[: header_idx + PKT_LEN]

                if pkt[:3] != HEADER:
                    continue

                typ = pkt[3]
                payload = pkt[5:]
                ts = time.time()
                retry_attempt = 0
                self.last_activity = ts

                if typ == TYPE_AA and len(payload) == 24:
                    emg = []
                    for i in range(0, 24, 3):
                        emg.append(self._int24_be_signed(payload[i], payload[i + 1], payload[i + 2]))
                    if self.remove_emg_mean:
                        mean = sum(emg) // 8
                        emg = [v - mean for v in emg]
                    emg_writer.writerow([f"{ts:.9f}"] + emg)
                    self.emg_count += 1

                    # Stream to ZMQ client
                    if self.sample_callback is not None:
                        self.sample_callback(ts, emg)

                    if self.emg_count % FLUSH_INTERVAL_SAMPLES == 0:
                        emg_fh.flush()
                        os.fsync(emg_fh.fileno())

                elif typ == TYPE_BB and len(payload) == 24:
                    raw = []
                    for i in range(0, 12, 2):
                        raw.append(self._int16_be_signed(payload[i], payload[i + 1]))
                    gr_x, gr_y, gr_z, acc_x, acc_y, acc_z = raw
                    imu_writer.writerow([
                        f"{ts:.9f}",
                        0.0012 * gr_x, 0.0012 * gr_y, 0.0012 * gr_z,
                        0.0005978 * acc_x, 0.0005978 * acc_y, 0.0005978 * acc_z,
                    ])
                    self.imu_count += 1
                    if self.imu_count % FLUSH_INTERVAL_SAMPLES == 0:
                        imu_fh.flush()
                        os.fsync(imu_fh.fileno())

                self.status_msg = "active"

        except Exception as exc:
            self.status_msg = f"error: {exc}"
        finally:
            for fh in (emg_fh, imu_fh):
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                    fh.close()
                except Exception:
                    pass
            if ser and ser.is_open:
                try:
                    ser.close()
                except Exception:
                    pass

    def stop(self) -> None:
        self._running.clear()

    def _open_serial(self, serial_mod):
        attempt = 0
        while self._running.is_set():
            try:
                ser = serial_mod.Serial(self.port, baudrate=self.baud, timeout=self.timeout)
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                self.status_msg = "connected"
                return ser
            except Exception:
                attempt += 1
                self.status_msg = f"waiting for {self.port} (attempt {attempt})"
                delay = min(0.5 * (2 ** min(attempt - 1, 4)), 5.0)
                time.sleep(delay)
        return None

    def _reconnect(self, old_ser, serial_mod, attempt: int):
        if old_ser and old_ser.is_open:
            try:
                old_ser.close()
            except Exception:
                pass
        delay = min(0.5 * (2 ** min(attempt, 4)), 5.0)
        self.status_msg = f"serial disconnected, reconnecting in {delay:.1f}s"
        time.sleep(delay)
        return self._open_serial(serial_mod)


class StreamingSessionManager:
    """Manages a collection session with optional real-time inference."""

    def __init__(
        self,
        output_root: str,
        serial_port: str,
        server_host: str | None = None,
        server_recv_port: int = 5555,
        server_send_port: int = 5556,
        server_ctrl_port: int = 5557,
        redis_host: str | None = None,
        redis_port: int = 6379,
    ):
        self.output_root = output_root
        self.serial_port = serial_port
        self.server_host = server_host
        self.server_recv_port = server_recv_port
        self.server_send_port = server_send_port
        self.server_ctrl_port = server_ctrl_port
        self.redis_host = redis_host
        self.redis_port = redis_port

        self.session_dir: Optional[Path] = None
        self.serial: Optional[StreamingSerialCollector] = None
        self.manus: Optional[ManusCollector] = None
        self.zmq_client: Optional[ZmqStreamClient] = None
        self.info: dict = {}
        self.info_path: Optional[Path] = None

    def start(self) -> Path:
        self.session_dir = make_session_dir(self.output_root)
        self.info_path = self.session_dir / "session_info.json"
        self.info = {
            "session": self.session_dir.name,
            "start_time": time.time(),
            "config": {
                "serial_port": self.serial_port,
                "serial_baud": DEFAULT_SERIAL_BAUD,
                "server": self.server_host,
                "redis": f"{self.redis_host}:{self.redis_port}" if self.redis_host else None,
            },
            "timebase": TIMEBASE_INFO,
        }
        self.info_path.write_text(json.dumps(self.info, indent=2))

        # Setup ZMQ client first (if server is specified)
        callback = None
        if self.server_host:
            self.zmq_client = ZmqStreamClient(
                server_host=self.server_host,
                recv_port=self.server_recv_port,
                send_port=self.server_send_port,
                ctrl_port=self.server_ctrl_port,
            )
            self.zmq_client.start()
            time.sleep(0.5)  # Give time for connection
            if self.zmq_client.connected:
                print(f"  Connected to inference server: {self.server_host}")
                print(f"  Server info: {self.zmq_client.server_info}")
                callback = self.zmq_client.enqueue_sample
            else:
                print(f"  WARNING: Could not connect to inference server")
                print(f"  Running in offline collection mode")
                self.zmq_client.stop()
                self.zmq_client = None

        self.serial = StreamingSerialCollector(
            port=self.serial_port,
            baud=DEFAULT_SERIAL_BAUD,
            timeout=DEFAULT_SERIAL_TIMEOUT,
            emg_path=str(self.session_dir / "emg.csv"),
            imu_path=str(self.session_dir / "imu.csv"),
            sample_callback=callback,
        )

        if self.redis_host:
            self.manus = ManusCollector(
                host=self.redis_host,
                port=self.redis_port,
                key_left="manus:left",
                key_right="manus:right",
                left_path=str(self.session_dir / "manus_left.jsonl"),
                right_path=str(self.session_dir / "manus_right.jsonl"),
            )

        self.serial.start()
        if self.manus:
            self.manus.start()
        return self.session_dir

    def stop(self) -> dict:
        if self.serial:
            self.serial.stop()
        if self.manus:
            self.manus.stop()
        if self.zmq_client:
            self.zmq_client.stop()

        if self.serial:
            self.serial.join(timeout=5)
        if self.manus:
            self.manus.join(timeout=5)

        end_time = time.time()
        duration = end_time - self.info.get("start_time", end_time)

        if self.serial:
            self.info["serial"] = {
                "emg_samples": self.serial.emg_count,
                "imu_samples": self.serial.imu_count,
                "emg_fps": round(self.serial.emg_count / max(duration, 0.001), 1),
                "imu_fps": round(self.serial.imu_count / max(duration, 0.001), 1),
            }
        if self.manus:
            self.info["manus"] = {
                "left_frames": self.manus.left_count,
                "right_frames": self.manus.right_count,
            }
        if self.zmq_client:
            self.info["predictions"] = {
                "received": self.zmq_client.predictions_received,
            }
            # Save predictions to JSONL
            preds = self.zmq_client.get_predictions()
            if preds and self.session_dir:
                pred_path = self.session_dir / "predictions.jsonl"
                with open(pred_path, "w") as f:
                    for p in preds:
                        f.write(json.dumps(p) + "\n")

        self.info["end_time"] = end_time
        self.info["duration_s"] = round(duration, 3)

        if self.info_path:
            self.info_path.write_text(json.dumps(self.info, indent=2))

        return self.info

    def is_active(self) -> bool:
        return self.serial is not None and self.serial.is_alive()


def run_session(
    args: argparse.Namespace,
    visualizer: HandVisualizer | None,
) -> None:
    """Run a single collection session with optional visualization."""
    mgr = StreamingSessionManager(
        output_root=args.out,
        serial_port=args.com_port,
        server_host=args.server,
        server_recv_port=args.recv_port,
        server_send_port=args.send_port,
        server_ctrl_port=args.ctrl_port,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )

    session_dir = mgr.start()
    print(f"\nSession: {session_dir}")
    print("Collecting... Press Ctrl+C to stop.\n")

    shutdown = False

    def on_signal(signum, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Get window_length from server info for progress display
    window_length = 7790  # default
    if mgr.zmq_client and mgr.zmq_client.connected:
        window_length = mgr.zmq_client.server_info.get("window_length", 7790)

    try:
        while not shutdown:
            if visualizer and mgr.zmq_client and mgr.zmq_client.connected:
                # Visualization mode: show latest prediction
                pred = mgr.zmq_client.get_latest_prediction()
                if pred and "landmarks" in pred:
                    landmarks = np.array(pred["landmarks"], dtype=np.float32)
                    angles = np.array(pred["angles"], dtype=np.float32)
                    if not visualizer.update(
                        landmarks, angles, pred.get("inference_ms", 0)
                    ):
                        break  # window closed
                else:
                    # No prediction yet, show live progress
                    if mgr.serial and mgr.serial.emg_count > 0:
                        elapsed = time.time() - mgr.serial.start_time
                        n = mgr.serial.emg_count
                        pct = min(100, n * 100 // window_length)
                        eta = max(0, (window_length - n) / max(
                            n / max(elapsed, 0.001), 1
                        ))
                        sys.stdout.write(
                            f"\r  EMG: {n:>6d}/{window_length} ({pct:3d}%) | "
                            f"{n / max(elapsed, 0.001):>6.0f} Hz | "
                            f"First prediction in ~{eta:.1f}s  "
                        )
                        sys.stdout.flush()
                time.sleep(0.02)  # ~50 FPS status update
            else:
                # Terminal-only status
                if mgr.is_active():
                    s = mgr.serial
                    elapsed = time.time() - s.start_time if s.start_time > 0 else 0
                    sys.stdout.write(
                        f"\r  EMG: {s.emg_count:>8d} | "
                        f"{s.emg_count / max(elapsed, 0.001):>7.0f} Hz | "
                        f"IMU: {s.imu_count:>8d} | "
                        f"[{s.status_msg}]"
                    )
                    if mgr.zmq_client:
                        z = mgr.zmq_client
                        sys.stdout.write(
                            f" | Pred: {z.predictions_received:>5d}"
                        )
                    sys.stdout.flush()
                time.sleep(STATUS_INTERVAL)

    except KeyboardInterrupt:
        pass

    info = mgr.stop()
    duration = info.get("duration_s", 0)
    s = info.get("serial", {})
    p = info.get("predictions", {})
    print(f"\n\nSession saved: {mgr.session_dir}")
    print(f"  Duration:  {fmt_duration(duration)}")
    print(f"  EMG: {s.get('emg_samples', 0):,} samples")
    print(f"  IMU: {s.get('imu_samples', 0):,} samples")
    if p:
        print(f"  Predictions: {p.get('received', 0):,}")
    print(f"  Output: {mgr.session_dir.resolve()}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG collector with optional real-time inference"
    )
    parser.add_argument("--out", default="data", help="Output root directory")
    parser.add_argument(
        "--com-port",
        default=DEFAULT_SERIAL_PORT,
        help=f"Serial port (default: {DEFAULT_SERIAL_PORT}, use COM3 etc. on Windows)",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Inference server IP/hostname (omit for offline collection)",
    )
    parser.add_argument("--recv-port", type=int, default=5555)
    parser.add_argument("--send-port", type=int, default=5556)
    parser.add_argument("--ctrl-port", type=int, default=5557)
    parser.add_argument(
        "--redis-host",
        default=None,
        help="Redis host for Manus glove (omit to skip Manus)",
    )
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable Open3D GUI, use terminal display only",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single session mode (collect until Ctrl+C)",
    )
    args = parser.parse_args()

    # Setup visualizer if server is connected
    visualizer = None
    if args.server and not args.no_gui:
        visualizer = HandVisualizer(use_gui=True)
    elif args.server:
        visualizer = HandVisualizer(use_gui=False)

    print("=" * 55)
    print("  EMG Data Collector + Real-time Inference")
    print("=" * 55)
    print(f"  Serial port: {args.com_port}")
    if args.server:
        print(f"  Server: {args.server}:{args.recv_port}")
    else:
        print(f"  Mode: offline collection (no server)")
    if args.redis_host:
        print(f"  Manus: {args.redis_host}:{args.redis_port}")
    print()

    # Run single session
    run_session(args, visualizer)

    if visualizer:
        visualizer.close()


if __name__ == "__main__":
    main()
