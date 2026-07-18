#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATA_COLLECT_ROOT = _PROJECT_ROOT / "data_collect"
if _DATA_COLLECT_ROOT.exists() and str(_DATA_COLLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECT_ROOT))

from emg2pose.realtime_local import FullModelOnlineTrainer, LocalSmallStreamer
from emg2pose.realtime_local.mesh_visualizer import RealtimeMeshVisualizer
from emg2pose.realtime_local.serial import SerialEmgReader, SerialProtocol


def _make_writer(path: str | None):
    if path is None:
        return None, None
    fh = open(path, "w", encoding="utf-8")

    def write(pred):
        fh.write(json.dumps(pred.to_jsonable()) + "\n")
        fh.flush()

    return fh, write


def _load_replay_memmap(root: Path) -> np.memmap:
    with open(root / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    spec = manifest["fields"]["emg_right_raw"]
    return np.memmap(
        root / spec["filename"],
        dtype=np.dtype(spec["dtype"]),
        mode="r",
        shape=tuple(spec["shape"]),
    )


def _log(args: argparse.Namespace, message: str) -> None:
    print(message, flush=True)
    path = getattr(args, "online_log_path", None)
    if not path:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{stamp} {message}\n")


class OnlineTeacherRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        trainer: FullModelOnlineTrainer,
        teacher_visualizer: RealtimeMeshVisualizer | None = None,
    ) -> None:
        self.args = args
        self.trainer = trainer
        self.teacher_visualizer = teacher_visualizer
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frames = 0
        self.results = 0
        self.last_inference_ms = 0.0
        self._preview_lock = threading.Lock()
        self._latest_preview: np.ndarray | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="wilor-online-teacher", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            self.thread = None

    def latest_preview(self) -> np.ndarray | None:
        with self._preview_lock:
            if self._latest_preview is None:
                return None
            return self._latest_preview.copy()

    def _set_preview(self, frame_bgr: np.ndarray, result) -> None:
        if not self.args.show_teacher_camera:
            return
        import cv2

        frame = frame_bgr.copy()
        if result is not None:
            x1, y1, x2, y2 = [int(v) for v in result.bbox.xyxy]
            color = (60, 220, 60) if result.bbox.detected else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = (
                f"teacher WiLoR+mapper {result.inference_ms:.1f}ms "
                f"frames={self.frames} results={self.results}"
            )
        else:
            text = f"teacher no hand frames={self.frames} results={self.results}"
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3)
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 1)
        with self._preview_lock:
            self._latest_preview = frame

    def _run(self) -> None:
        import cv2

        from scripts.realtime.local_wilor_mapper_mesh import (
            RealtimeWilorMapper,
            _open_camera,
        )

        try:
            _log(self.args, "[online-adapt] starting WiLoR teacher")
            predictor = RealtimeWilorMapper(
                mapper_checkpoint=self.args.teacher_mapper_checkpoint,
                hand=self.args.teacher_hand,
                device=self.args.teacher_device,
                dtype=self.args.teacher_dtype,
                yolo_conf=self.args.teacher_yolo_conf,
                yolo_input_height=self.args.teacher_yolo_input_height,
                wilor_pretrained_dir=self.args.teacher_wilor_pretrained_dir,
                detect_interval=self.args.teacher_detect_interval,
                max_bbox_age=self.args.teacher_max_bbox_age,
                pose_source="mapper",
                lbfgs_max_iter=1,
                lbfgs_lr=0.5,
                lbfgs_history_size=4,
                visualize_mano_mesh=False,
            )
            cap = _open_camera(
                self.args.teacher_camera,
                self.args.teacher_width,
                self.args.teacher_height,
                self.args.teacher_fps,
                self.args.teacher_backend,
            )
            _log(self.args, f"[online-adapt] teacher camera opened: {self.args.teacher_camera}")
        except Exception as exc:
            _log(self.args, f"[online-adapt] teacher failed to start: {type(exc).__name__}: {exc}")
            return
        try:
            while not self.stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                self.frames += 1
                ts = time.time()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = predictor.predict(frame_rgb)
                if result is None:
                    self._set_preview(frame_bgr, None)
                    continue
                self.results += 1
                self.last_inference_ms = result.inference_ms
                self.trainer.add_teacher(ts, result.angles[:20])
                if self.teacher_visualizer is not None:
                    self.teacher_visualizer.update(result.angles[:20], result.inference_ms)
                self._set_preview(frame_bgr, result)
        finally:
            cap.release()
            _log(self.args, "[online-adapt] teacher stopped")


def _maybe_start_online_adapt(
    args: argparse.Namespace,
    streamer: LocalSmallStreamer,
) -> tuple[FullModelOnlineTrainer | None, OnlineTeacherRunner | None, RealtimeMeshVisualizer | None]:
    if not args.online_full_adapt:
        return None, None, None
    trainer = FullModelOnlineTrainer(
        streamer,
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
        batch_size=args.adapt_batch_size,
        min_samples=args.adapt_min_samples,
        buffer_size=args.adapt_buffer_size,
        match_tolerance_s=args.adapt_match_tolerance_s,
        update_interval_s=args.adapt_update_interval_s,
        steps_per_update=args.adapt_steps_per_update,
        grad_clip=args.adapt_grad_clip,
        keep_weight=args.adapt_keep_weight,
    )
    streamer.window_callback = trainer.observe_prediction
    trainer.start()
    teacher_visualizer = _make_visualizer(
        args.visualize_teacher_mesh,
        window_name="Teacher WiLoR UmeTrack Mesh",
    )
    teacher = OnlineTeacherRunner(args, trainer, teacher_visualizer)
    teacher.start()
    _log(
        args,
        "Online full-model adaptation enabled: "
        f"lr={args.adapt_lr:g}, batch={args.adapt_batch_size}, "
        f"teacher_camera={args.teacher_camera}, "
        f"show_camera={args.show_teacher_camera}, "
        f"teacher_mesh={args.visualize_teacher_mesh}",
    )
    if args.online_log_path:
        _log(args, f"Online adaptation log: {args.online_log_path}")
    return trainer, teacher, teacher_visualizer


def _format_adapt_log(
    trainer: FullModelOnlineTrainer | None,
    teacher: OnlineTeacherRunner | None,
) -> str:
    if trainer is None:
        return ""
    status = trainer.status()
    loss = "nan" if status.last_loss is None else f"{status.last_loss:.5f}"
    age = "nan" if status.last_teacher_age_s is None else f"{status.last_teacher_age_s:.3f}s"
    teacher_extra = ""
    if teacher is not None:
        teacher_extra = (
            f" teacher_frames={teacher.frames}"
            f" teacher_results={teacher.results}"
            f" teacher_infer={teacher.last_inference_ms:.1f}ms"
        )
    return (
        "[online-adapt] "
        f"steps={status.step_count} "
        f"samples={status.sample_count} "
        f"teacher_buf={status.teacher_count} "
        f"match_age={age} "
        f"loss={loss}"
        f"{teacher_extra}"
    )


def run_replay(args: argparse.Namespace) -> None:
    writer_fh, writer = _make_writer(args.save_jsonl)
    visualizer = _make_visualizer(args.visualize_mesh, window_name="EMG UmeTrack Mesh")
    input_scale = 1.0 if args.input_scale is None else args.input_scale

    def on_prediction(pred):
        if writer is not None:
            writer(pred)
        if visualizer is not None:
            visualizer.update(pred.angles, pred.inference_ms)

    streamer = LocalSmallStreamer(
        checkpoint_path=args.checkpoint,
        stride_samples=args.stride_samples,
        device=args.device,
        noise_floor_path=args.noise_floor,
        input_scale=input_scale,
        remove_sample_mean=args.remove_sample_mean,
        output_delay_s=args.output_delay_s,
        compile_model=args.compile,
        callback=on_prediction,
    )
    trainer, teacher, teacher_visualizer = _maybe_start_online_adapt(args, streamer)
    root = Path(args.replay_memmap)
    raw = _load_replay_memmap(root)
    start = int(args.replay_start)
    end = min(raw.shape[0], start + int(args.replay_samples))
    chunk = int(args.chunk_samples)
    n_pred = 0
    t0 = time.monotonic()
    last_adapt_log_t = 0.0
    try:
        for s in range(start, end, chunk):
            samples = np.asarray(raw[s : min(end, s + chunk)], dtype=np.float32)
            preds = streamer.push_samples(samples, timestamp=time.time())
            n_pred += len(preds)
            shutdown = _poll_debug_windows(visualizer, teacher_visualizer, teacher)
            if shutdown:
                break
            now = time.monotonic()
            if trainer is not None and now - last_adapt_log_t >= args.adapt_log_interval_s:
                _log(args, "\n" + _format_adapt_log(trainer, teacher))
                last_adapt_log_t = now
            for pred in preds:
                print(
                    f"\rpred={n_pred:5d} sample={pred.sample_index:8d} "
                    f"inference={pred.inference_ms:7.2f}ms "
                    f"angle0={pred.angles[0]: .3f}",
                    end="",
                    flush=True,
                )
    finally:
        if teacher is not None:
            teacher.stop()
        if trainer is not None:
            if args.adapt_save_checkpoint:
                trainer.save_checkpoint(args.adapt_save_checkpoint)
                print(f"\nSaved online-adapted checkpoint: {args.adapt_save_checkpoint}")
            trainer.stop()
        if visualizer is not None:
            visualizer.close()
        if teacher_visualizer is not None:
            teacher_visualizer.close()
        if args.show_teacher_camera:
            try:
                import cv2

                cv2.destroyWindow("Online teacher camera")
            except Exception:
                pass
        if writer_fh is not None:
            writer_fh.close()
    elapsed = time.monotonic() - t0
    print(f"\nReplay done: samples={end-start:,} predictions={n_pred:,} elapsed={elapsed:.2f}s")


def run_serial(args: argparse.Namespace) -> None:
    writer_fh, writer = _make_writer(args.save_jsonl)
    visualizer = _make_visualizer(args.visualize_mesh, window_name="EMG UmeTrack Mesh")
    input_scale = 0.001 if args.input_scale is None else args.input_scale
    n_pred = 0

    def on_prediction(pred):
        nonlocal n_pred
        n_pred += 1
        if writer is not None:
            writer(pred)
        if visualizer is not None:
            visualizer.update(pred.angles, pred.inference_ms)
        print(
            f"\rpred={n_pred:5d} sample={pred.sample_index:8d} "
            f"inference={pred.inference_ms:7.2f}ms "
            f"angle0={pred.angles[0]: .3f}",
            end="",
            flush=True,
        )

    streamer = LocalSmallStreamer(
        checkpoint_path=args.checkpoint,
        stride_samples=args.stride_samples,
        device=args.device,
        noise_floor_path=args.noise_floor,
        input_scale=input_scale,
        remove_sample_mean=args.remove_sample_mean,
        output_delay_s=args.output_delay_s,
        compile_model=args.compile,
        callback=on_prediction,
    )
    trainer, teacher, teacher_visualizer = _maybe_start_online_adapt(args, streamer)
    protocol = SerialProtocol.from_collect_or_args(
        header_hex=args.header_hex,
        packet_len=args.packet_len,
        emg_type=args.emg_type,
        imu_type=args.imu_type,
        payload_offset=args.payload_offset,
    )

    def on_sample(ts: float, emg: np.ndarray) -> None:
        streamer.push_samples(emg, timestamp=ts)

    reader = SerialEmgReader(
        port=args.com_port,
        baud=args.baud,
        timeout=args.timeout,
        protocol=protocol,
        on_sample=on_sample,
    )
    shutdown = False

    def _stop(_signum, _frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    reader.start()
    t0 = time.monotonic()
    last_status_t = 0.0
    last_adapt_log_t = 0.0
    try:
        while not shutdown and reader.is_alive():
            if _poll_debug_windows(visualizer, teacher_visualizer, teacher):
                shutdown = True
                break
            now = time.monotonic()
            if now - last_status_t >= 0.25:
                elapsed = now - t0
                rate = reader.samples_read / max(elapsed, 1e-6)
                adapt = ""
                if trainer is not None:
                    status = trainer.status()
                    loss = "nan" if status.last_loss is None else f"{status.last_loss:.4f}"
                    adapt = (
                        f" adapt_steps={status.step_count:4d}"
                        f" adapt_buf={status.sample_count:3d}"
                        f" teacher={status.teacher_count:3d}"
                        f" loss={loss}"
                    )
                print(
                    f"\rserial={reader.status} samples={reader.samples_read:8d} "
                    f"rate={rate:7.0f}Hz pred={n_pred:5d}{adapt}",
                    end="",
                    flush=True,
                )
                last_status_t = now
            if trainer is not None and now - last_adapt_log_t >= args.adapt_log_interval_s:
                _log(args, "\n" + _format_adapt_log(trainer, teacher))
                last_adapt_log_t = now
            has_windows = (
                visualizer is not None
                or teacher_visualizer is not None
                or (teacher is not None and args.show_teacher_camera)
            )
            time.sleep(1.0 / 60.0 if has_windows else 0.25)
    finally:
        reader.stop()
        reader.join(timeout=2.0)
        if teacher is not None:
            teacher.stop()
        if trainer is not None:
            if args.adapt_save_checkpoint:
                trainer.save_checkpoint(args.adapt_save_checkpoint)
                print(f"\nSaved online-adapted checkpoint: {args.adapt_save_checkpoint}")
            trainer.stop()
        if visualizer is not None:
            visualizer.close()
        if teacher_visualizer is not None:
            teacher_visualizer.close()
        if args.show_teacher_camera:
            try:
                import cv2

                cv2.destroyWindow("Online teacher camera")
            except Exception:
                pass
        if writer_fh is not None:
            writer_fh.close()
    print("\nStopped.")


def _poll_debug_windows(
    visualizer: RealtimeMeshVisualizer | None,
    teacher_visualizer: RealtimeMeshVisualizer | None,
    teacher: OnlineTeacherRunner | None,
) -> bool:
    if visualizer is not None and not visualizer.poll():
        return True
    if teacher_visualizer is not None and not teacher_visualizer.poll():
        return True
    if teacher is not None and teacher.args.show_teacher_camera:
        preview = teacher.latest_preview()
        if preview is not None:
            import cv2

            cv2.imshow("Online teacher camera", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
    return False


def _make_visualizer(
    enabled: bool,
    window_name: str,
) -> RealtimeMeshVisualizer | None:
    if not enabled:
        return None
    try:
        return RealtimeMeshVisualizer(window_name=window_name)
    except RuntimeError as exc:
        print(f"{window_name} disabled: {exc}", file=sys.stderr, flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Local small EMGFormer streaming inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride-samples", type=int, default=200)
    parser.add_argument("--noise-floor", default=None)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help=(
            "Scale applied before filtering. Defaults to 0.001 for serial "
            "device int24 values and 1.0 for memmap replay."
        ),
    )
    parser.add_argument("--remove-sample-mean", action="store_true")
    parser.add_argument("--output-delay-s", type=float, default=0.5)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--save-jsonl", default=None)
    parser.add_argument("--visualize-mesh", action="store_true")
    parser.add_argument("--online-full-adapt", action="store_true")
    parser.add_argument("--adapt-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapt-batch-size", type=int, default=4)
    parser.add_argument("--adapt-min-samples", type=int, default=8)
    parser.add_argument("--adapt-buffer-size", type=int, default=256)
    parser.add_argument("--adapt-match-tolerance-s", type=float, default=0.12)
    parser.add_argument("--adapt-update-interval-s", type=float, default=1.0)
    parser.add_argument("--adapt-steps-per-update", type=int, default=1)
    parser.add_argument("--adapt-grad-clip", type=float, default=1.0)
    parser.add_argument("--adapt-keep-weight", type=float, default=0.05)
    parser.add_argument("--adapt-save-checkpoint", default=None)
    parser.add_argument("--adapt-log-interval-s", type=float, default=2.0)
    parser.add_argument("--online-log-path", default=None)
    parser.add_argument("--show-teacher-camera", action="store_true")
    parser.add_argument("--visualize-teacher-mesh", action="store_true")
    parser.add_argument("--teacher-camera", type=int, default=1)
    parser.add_argument("--teacher-width", type=int, default=1280)
    parser.add_argument("--teacher-height", type=int, default=720)
    parser.add_argument("--teacher-fps", type=float, default=30.0)
    parser.add_argument("--teacher-backend", choices=["dshow", "any"], default="dshow")
    parser.add_argument("--teacher-hand", choices=["right", "left"], default="right")
    parser.add_argument("--teacher-device", default="cuda")
    parser.add_argument("--teacher-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--teacher-mapper-checkpoint",
        default="pretrained_models/mano_to_umetrack_mapper.pt",
    )
    parser.add_argument("--teacher-wilor-pretrained-dir", default=None)
    parser.add_argument("--teacher-yolo-conf", type=float, default=0.1)
    parser.add_argument("--teacher-yolo-input-height", type=int, default=512)
    parser.add_argument("--teacher-detect-interval", type=int, default=1)
    parser.add_argument("--teacher-max-bbox-age", type=int, default=3)

    parser.add_argument("--replay-memmap", default=None)
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--replay-samples", type=int, default=60_000)
    parser.add_argument("--chunk-samples", type=int, default=50)

    parser.add_argument("--com-port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--header-hex", default=None)
    parser.add_argument("--packet-len", type=int, default=None)
    parser.add_argument("--emg-type", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--imu-type", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--payload-offset", type=int, default=5)
    args = parser.parse_args()

    if args.replay_memmap:
        run_replay(args)
    else:
        run_serial(args)


if __name__ == "__main__":
    main()
