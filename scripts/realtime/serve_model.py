#!/usr/bin/env python3
"""Serve EMGFormer model for real-time inference over ZMQ.

Usage:
  python scripts/realtime/serve_model.py \\
    --checkpoint test_results/.../best.ckpt \\
    --device cuda \\
    --stride 400

Clients connect via ZMQ:
  PUSH tcp://host:5555  (send raw EMG)
  SUB  tcp://host:5556  (receive predictions)
  REQ  tcp://host:5557  (control: ping/stop)

Over SSH tunnel:
  ssh -L 5555:localhost:5555 -L 5556:localhost:5556 -L 5557:localhost:5557 user@server
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from realtime.server import InferenceServer, ServerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Serve EMGFormer model for real-time inference"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=400,
        help="Samples between predictions (default: 400 = 200ms @ 2kHz)",
    )
    p.add_argument(
        "--recv-port",
        type=int,
        default=5555,
        help="ZMQ PULL port for receiving EMG (default: 5555)",
    )
    p.add_argument(
        "--send-port",
        type=int,
        default=5556,
        help="ZMQ PUB port for sending predictions (default: 5556)",
    )
    p.add_argument(
        "--ctrl-port",
        type=int,
        default=5557,
        help="ZMQ REP port for control (default: 5557)",
    )
    p.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Disable UmeTrack FK landmark computation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = ServerConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        recv_port=args.recv_port,
        send_port=args.send_port,
        ctrl_port=args.ctrl_port,
        stride=args.stride,
        compute_landmarks=not args.no_landmarks,
    )

    server = InferenceServer(config)
    server.setup()
    server.run()


if __name__ == "__main__":
    main()
