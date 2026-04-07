"""Live webcam inference helpers for stage-3 models."""

from __future__ import annotations

import json
import queue
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch

from src.stage2.vision_encoder import FrozenEfficientNetStreamEncoder
from src.stage3.generation import generate_prediction_from_features
from src.stage3.manifest import STAGE3_SYSTEM_PROMPT
from src.stage3.model import build_stage3_lora_streamvlm


@dataclass
class RollingFeatureBuffer:
    """Rolling in-memory feature buffer for live stage-3 decoding."""

    history_sec: float
    spatial_res: list[int]
    feature_steps: list[torch.Tensor] = field(default_factory=list)
    feature_timestamps: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_step(self, feature_step: torch.Tensor, timestamp_sec: float) -> None:
        with self._lock:
            self.feature_steps.append(feature_step.detach().cpu())
            self.feature_timestamps.append(float(timestamp_sec))
            self._prune_locked()

    def _prune_locked(self) -> None:
        if not self.feature_timestamps:
            return
        cutoff = float(self.feature_timestamps[-1] - self.history_sec)
        keep_from = 0
        while keep_from < len(self.feature_timestamps) and self.feature_timestamps[keep_from] < cutoff:
            keep_from += 1
        if keep_from > 0:
            self.feature_steps = self.feature_steps[keep_from:]
            self.feature_timestamps = self.feature_timestamps[keep_from:]

    @property
    def ready(self) -> bool:
        with self._lock:
            return bool(self.feature_steps)

    @property
    def feature_count(self) -> int:
        with self._lock:
            return len(self.feature_steps)

    @property
    def duration_sec(self) -> float:
        with self._lock:
            return self._duration_locked()

    def _duration_locked(self) -> float:
        if len(self.feature_timestamps) < 2:
            return 0.0
        return float(self.feature_timestamps[-1] - self.feature_timestamps[0])

    def snapshot(self) -> tuple[torch.Tensor, list[float], list[int], float] | None:
        with self._lock:
            if not self.feature_steps:
                return None
            buffer_start = float(self.feature_timestamps[0])
            rel_timestamps = [float(timestamp - buffer_start) for timestamp in self.feature_timestamps]
            return (
                torch.stack(self.feature_steps, dim=0),
                rel_timestamps,
                list(self.spatial_res),
                buffer_start,
            )


@dataclass
class WorkerStatus:
    """Thread-safe status bookkeeping for the live webcam loop."""

    encoder_state: str = "idle"
    decoder_state: str = "idle"
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_encoder_state(self, value: str) -> None:
        with self._lock:
            self.encoder_state = value

    def set_decoder_state(self, value: str) -> None:
        with self._lock:
            self.decoder_state = value

    def set_error(self, value: str) -> None:
        with self._lock:
            self.last_error = value

    def snapshot(self) -> tuple[str, str, str | None]:
        with self._lock:
            return self.encoder_state, self.decoder_state, self.last_error


def _resolve_device_arg(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip().lower()
    if normalized == "auto":
        return None
    return normalized


def _load_stage3_live_model(
    stage2_checkpoint_dir: str | Path,
    stage3_adapter_dir: str | Path,
    device: str | None,
):
    model = build_stage3_lora_streamvlm(
        stage2_checkpoint_dir=stage2_checkpoint_dir,
        device=_resolve_device_arg(device),
    )
    model.load_lora_adapter(stage3_adapter_dir, is_trainable=False)
    model.eval()
    return model


def _wrap_overlay_text(text: str, width: int = 44) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def _draw_overlay(
    frame,
    latest_feedback: str | None,
    latest_feedback_expires_at: float,
    stream_elapsed_sec: float,
    status_line: str,
) -> None:
    height, width = frame.shape[:2]
    cv2.putText(
        frame,
        status_line,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (64, 255, 64),
        2,
        cv2.LINE_AA,
    )
    if not latest_feedback or stream_elapsed_sec > latest_feedback_expires_at:
        return

    lines = _wrap_overlay_text(latest_feedback, width=30)
    if not lines:
        return

    line_height = 48
    text_scale = 1.2
    text_thickness = 3
    box_height = 32 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, height - box_height - 8),
        (width - 8, height - 8),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)

    y = height - box_height + 18
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        y += line_height


def _emit_feedback_event(
    event_log: list[dict[str, Any]],
    save_log_path: str | Path | None,
    timestamp_sec: float,
    text: str,
) -> None:
    event = {
        "timestamp_sec": float(timestamp_sec),
        "text": text,
    }
    event_log.append(event)
    print(f"[{timestamp_sec:7.2f}s] {text}", flush=True)
    if save_log_path is not None:
        save_path = Path(save_log_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _enqueue_latest_frame(
    frame_queue: queue.Queue[tuple[Any, float]],
    item: tuple[Any, float],
) -> None:
    while True:
        try:
            frame_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return


def _encoder_worker(
    *,
    frame_queue: queue.Queue[tuple[Any, float]],
    feature_buffer: RollingFeatureBuffer,
    vision_encoder: FrozenEfficientNetStreamEncoder,
    rotate_90_cw: bool,
    status: WorkerStatus,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            pending_frame, pending_timestamp = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            status.set_encoder_state("encoding")
            feature_step = vision_encoder.encode_stream_frame(
                pending_frame,
                rotate_90_cw=rotate_90_cw,
            )
            if feature_step is not None:
                feature_buffer.add_step(feature_step, pending_timestamp)
        except Exception as exc:  # pragma: no cover - runtime safety
            status.set_error(f"encoder: {exc}")
            stop_event.set()
            break
        finally:
            status.set_encoder_state("idle")


def _decoder_worker(
    *,
    model,
    feature_buffer: RollingFeatureBuffer,
    result_queue: queue.Queue[dict[str, Any]],
    status: WorkerStatus,
    stop_event: threading.Event,
    warmup_sec: float,
    decode_interval_sec: float,
    latest_only_decode: bool,
    min_observation_sec: float,
    post_feedback_min_next_steps: int,
    max_feedback_tokens: int,
    max_total_new_tokens: int,
) -> None:
    next_decode_deadline = time.perf_counter()
    last_emitted_feedback_abs_sec = -1e9

    while not stop_event.is_set():
        now = time.perf_counter()
        if now < next_decode_deadline:
            time.sleep(0.05)
            continue

        snapshot = feature_buffer.snapshot()
        if snapshot is None:
            time.sleep(0.05)
            continue

        full_vision_feats, rel_feature_timestamps, spatial_res, buffer_start_abs_sec = snapshot
        if len(rel_feature_timestamps) < 2 or rel_feature_timestamps[-1] < float(warmup_sec):
            time.sleep(0.05)
            continue

        decode_started_at = now
        try:
            status.set_decoder_state("decoding")
            prediction = generate_prediction_from_features(
                model,
                full_vision_feats=full_vision_feats,
                feature_timestamps=rel_feature_timestamps,
                spatial_res=spatial_res,
                system_prompt=STAGE3_SYSTEM_PROMPT,
                segment_id="live:webcam:000",
                video_id="webcam",
                exercise_name="live_webcam",
                min_observation_sec=min_observation_sec,
                post_feedback_min_next_steps=post_feedback_min_next_steps,
                max_feedback_tokens=max_feedback_tokens,
                max_total_new_tokens=max_total_new_tokens,
                return_debug=False,
            )
            for text, relative_timestamp in zip(
                prediction["pred_feedbacks"],
                prediction["pred_feedback_timestamps"],
            ):
                absolute_timestamp = float(buffer_start_abs_sec + float(relative_timestamp))
                if absolute_timestamp <= last_emitted_feedback_abs_sec + 0.5:
                    continue
                result_queue.put_nowait(
                    {
                        "timestamp_sec": absolute_timestamp,
                        "text": text,
                    }
                )
                last_emitted_feedback_abs_sec = absolute_timestamp
        except Exception as exc:  # pragma: no cover - runtime safety
            status.set_error(f"decoder: {exc}")
            stop_event.set()
            break
        finally:
            status.set_decoder_state("idle")
            if latest_only_decode:
                next_decode_deadline = time.perf_counter() + float(decode_interval_sec)
            else:
                next_decode_deadline = decode_started_at + float(decode_interval_sec)


def _print_runtime_info(model, vision_encoder) -> None:
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    print(
        (
            "runtime: "
            f"model_device={getattr(model, 'device', 'unknown')} "
            f"vision_device={getattr(vision_encoder, 'device', 'unknown')} "
            f"mps_available={mps_available} "
            f"mps_built={mps_built}"
        ),
        flush=True,
    )


def run_live_webcam_inference(
    *,
    stage2_checkpoint_dir: str | Path,
    stage3_adapter_dir: str | Path,
    vision_checkpoint: str | Path,
    camera_index: int = 0,
    device: str | None = "auto",
    capture_fps: float = 8.0,
    history_sec: float = 12.0,
    warmup_sec: float = 2.0,
    encode_interval_sec: float = 3.0,
    decode_interval_sec: float = 5.0,
    min_observation_sec: float = 2.0,
    post_feedback_min_next_steps: int = 1,
    max_feedback_tokens: int = 32,
    max_total_new_tokens: int = 96,
    overlay_hold_sec: float = 5.0,
    save_log: str | Path | None = None,
    frame_timeout_sec: float = 10.0,
    rotate_90_cw: bool = False,
    frame_queue_size: int = 8,
    latest_only_decode: bool = True,
    print_runtime_info: bool = True,
) -> list[dict[str, Any]]:
    """Run live webcam inference with a rolling feature history."""
    if capture_fps <= 0:
        raise ValueError("capture_fps must be positive")
    if frame_queue_size <= 0:
        raise ValueError("frame_queue_size must be positive")

    model = _load_stage3_live_model(
        stage2_checkpoint_dir=stage2_checkpoint_dir,
        stage3_adapter_dir=stage3_adapter_dir,
        device=device,
    )
    vision_encoder = FrozenEfficientNetStreamEncoder(
        checkpoint_path=vision_checkpoint,
        device=_resolve_device_arg(device),
    )
    if print_runtime_info:
        _print_runtime_info(model, vision_encoder)

    feature_buffer = RollingFeatureBuffer(
        history_sec=float(history_sec),
        spatial_res=list(vision_encoder.spatial_res),
    )
    status = WorkerStatus()
    stop_event = threading.Event()
    frame_queue: queue.Queue[tuple[Any, float]] = queue.Queue(maxsize=int(frame_queue_size))
    result_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    encoder_thread = threading.Thread(
        target=_encoder_worker,
        kwargs={
            "frame_queue": frame_queue,
            "feature_buffer": feature_buffer,
            "vision_encoder": vision_encoder,
            "rotate_90_cw": rotate_90_cw,
            "status": status,
            "stop_event": stop_event,
        },
        daemon=True,
        name="fitcoach-encoder",
    )
    decoder_thread = threading.Thread(
        target=_decoder_worker,
        kwargs={
            "model": model,
            "feature_buffer": feature_buffer,
            "result_queue": result_queue,
            "status": status,
            "stop_event": stop_event,
            "warmup_sec": warmup_sec,
            "decode_interval_sec": decode_interval_sec,
            "latest_only_decode": latest_only_decode,
            "min_observation_sec": min_observation_sec,
            "post_feedback_min_next_steps": post_feedback_min_next_steps,
            "max_feedback_tokens": max_feedback_tokens,
            "max_total_new_tokens": max_total_new_tokens,
        },
        daemon=True,
        name="fitcoach-decoder",
    )
    encoder_thread.start()
    decoder_thread.start()

    capture = cv2.VideoCapture(int(camera_index))
    if not capture.isOpened():
        stop_event.set()
        raise RuntimeError(f"Unable to open webcam at camera index {camera_index}")

    stream_start = time.perf_counter()
    last_frame_time = stream_start
    next_capture_time = stream_start
    latest_feedback_text: str | None = None
    latest_feedback_expires_at = 0.0
    pending_overlay_events: deque[dict[str, Any]] = deque()
    event_log: list[dict[str, Any]] = []

    try:
        while True:
            success, frame = capture.read()
            now = time.perf_counter()
            stream_elapsed_sec = now - stream_start

            if not success:
                if (now - last_frame_time) > float(frame_timeout_sec):
                    raise RuntimeError("Webcam frame capture timed out")
                time.sleep(0.01)
                continue

            last_frame_time = now
            preview_frame = frame.copy()

            if now >= next_capture_time:
                _enqueue_latest_frame(frame_queue, (frame.copy(), stream_elapsed_sec))
                capture_period = 1.0 / float(capture_fps)
                while next_capture_time <= now:
                    next_capture_time += capture_period

            while True:
                try:
                    feedback_event = result_queue.get_nowait()
                except queue.Empty:
                    break
                _emit_feedback_event(
                    event_log=event_log,
                    save_log_path=save_log,
                    timestamp_sec=float(feedback_event["timestamp_sec"]),
                    text=str(feedback_event["text"]),
                )
                pending_overlay_events.append(
                    {
                        "text": str(feedback_event["text"]),
                        "timestamp_sec": float(feedback_event["timestamp_sec"]),
                    }
                )

            if (
                (latest_feedback_text is None or stream_elapsed_sec > latest_feedback_expires_at)
                and pending_overlay_events
            ):
                next_overlay_event = pending_overlay_events.popleft()
                latest_feedback_text = str(next_overlay_event["text"])
                latest_feedback_expires_at = stream_elapsed_sec + float(overlay_hold_sec)

            encoder_state, decoder_state, last_error = status.snapshot()
            if last_error:
                raise RuntimeError(last_error)

            decoder_mode = "latest" if latest_only_decode else "queue"
            status_line = (
                f"t={stream_elapsed_sec:5.1f}s  feature_steps={feature_buffer.feature_count}  "
                f"history={feature_buffer.duration_sec:4.1f}s  enc={encoder_state}  "
                f"dec={decoder_state}/{decoder_mode}"
            )
            _draw_overlay(
                preview_frame,
                latest_feedback_text,
                latest_feedback_expires_at,
                stream_elapsed_sec,
                status_line,
            )
            cv2.imshow("FitCoach Live Webcam", preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_event.set()
        capture.release()
        cv2.destroyAllWindows()
        encoder_thread.join(timeout=1.0)
        decoder_thread.join(timeout=1.0)

    return event_log
