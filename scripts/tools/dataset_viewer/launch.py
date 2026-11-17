"""Serve the EgoEMG dataset on loopback and refresh when sessions are published."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[2] / "data/EgoEMG_dataset/dataset.json"
DEFAULT_PORT = 8793
child = None
stopping = False


def stop(signum, frame):
    global stopping
    stopping = True


def signature():
    return hashlib.sha256(DATASET.read_bytes()).hexdigest()


def port_path(name, port):
    path = Path(name)
    base = HERE.parents[2] / "logs/tools/dataset_viewer"
    base.mkdir(parents=True, exist_ok=True)
    if port == DEFAULT_PORT:
        return base / name
    return base / f"{path.stem}_{port}{path.suffix}"


def save_state(state, port, **values):
    path = port_path("process.json", port)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(
        state=state, launcher_pid=os.getpid(), port=port,
        updated_unix=time.time(), **values
    ), indent=2) + "\n")
    temp.replace(path)


def stop_child():
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def wait_ready(port, process, log_path):
    for _ in range(100):
        if process.poll() is not None:
            with log_path.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 2000))
                detail = stream.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Dashboard server exited on port {port}. Log: {log_path}\n{detail}"
            )
        with socket.socket() as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"Dashboard server did not listen on port {port}; see {log_path}")


if __name__ == "__main__":
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Remote listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET.parent,
                        help="Session dataset directory (default: data/EgoEMG_dataset)")
    args = parser.parse_args()
    DATASET = args.dataset_dir.resolve() / "dataset.json"
    if not DATASET.is_file():
        parser.error(f"dataset manifest not found: {DATASET}")
    port = args.port
    if not 1 <= port <= 65535:
        parser.error("--port must be between 1 and 65535")
    lock = port_path("dashboard.lock", port).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"dashboard launcher is already running on port {port}")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            parser.error(f"port {port} is already in use")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = os.environ.copy()
    environment.setdefault(
        "COLLECTOR_RVL_PLUGIN_DIR", str(HERE.parent / "depth_rvl/native/lib")
    )
    generation = None
    try:
        log_path = port_path("server.log", port)
        with log_path.open("ab") as log:
            while not stopping:
                latest = signature()
                if child is None or child.poll() is not None or latest != generation:
                    stop_child()
                    child = subprocess.Popen([
                        sys.executable, "-u", str(HERE / "server.py"),
                        "--dataset", f"EgoEMG_dataset={DATASET.parent}",
                        "--port", str(port)
                    ], cwd=HERE.parents[2], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                    wait_ready(port, child, log_path)
                    generation = latest
                    save_state("serving", port, server_pid=child.pid,
                               dataset_sha256=generation)
                    print(f"EgoEMG dashboard (remote host): http://127.0.0.1:{port}/", flush=True)
                    print(
                        f"Local browser: forward remote port {port} in VS Code's Ports view, "
                        "then open the Local Address shown there.",
                        flush=True,
                    )
                time.sleep(5)
    finally:
        stop_child()
        save_state("stopped", port)
