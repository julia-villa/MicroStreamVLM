#!/usr/bin/env python3
"""Run live webcam inference with a stage-3 checkpoint and adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-checkpoint-dir",
        default="outputs/stage2_xattn_llamainstruct_a40/final",
        help="Path to the validated stage-2 checkpoint directory",
    )
    parser.add_argument(
        "--stage3-adapter-dir",
        default="outputs/stage3_llama_instruct_v2-1/final_adapter",
        help="Path to the trained stage-3 LoRA adapter directory",
    )
    parser.add_argument(
        "--vision-checkpoint",
        required=True,
        help="Path to the EfficientNet vision checkpoint used for stage-2/stage-3",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device: auto, cpu, cuda, or mps",
    )
    parser.add_argument("--capture-fps", type=float, default=8.0, help="Live sampling FPS")
    parser.add_argument("--history-sec", type=float, default=12.0, help="Rolling feature-history length")
    parser.add_argument("--warmup-sec", type=float, default=2.0, help="Minimum observed history before decode")
    parser.add_argument(
        "--encode-interval-sec",
        type=float,
        default=3.0,
        help="How often to encode newly captured frames",
    )
    parser.add_argument(
        "--decode-interval-sec",
        type=float,
        default=5.0,
        help="How often to rerun stage-3 decoding over the rolling history",
    )
    parser.add_argument(
        "--min-observation-sec",
        type=float,
        default=2.0,
        help="Minimum observed seconds before feedback may open",
    )
    parser.add_argument(
        "--post-feedback-min-next-steps",
        type=int,
        default=1,
        help="Require this many <next> steps after </feedback> before reopening",
    )
    parser.add_argument(
        "--max-feedback-tokens",
        type=int,
        default=32,
        help="Maximum generated tokens per feedback span",
    )
    parser.add_argument(
        "--max-total-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens per live decode pass",
    )
    parser.add_argument(
        "--overlay-hold-sec",
        type=float,
        default=5.0,
        help="How long to keep the latest feedback overlay visible",
    )
    parser.add_argument(
        "--save-log",
        default=None,
        help="Optional JSONL path for emitted live feedback events",
    )
    parser.add_argument(
        "--frame-timeout-sec",
        type=float,
        default=10.0,
        help="Stop if the webcam stops delivering frames for this long",
    )
    parser.add_argument(
        "--rotate-90-cw",
        action="store_true",
        help="Rotate the captured frames 90 degrees clockwise before encoding",
    )
    parser.add_argument(
        "--frame-queue-size",
        type=int,
        default=8,
        help="Bounded queued live frames waiting for background encoding",
    )
    parser.add_argument(
        "--latest-only-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip overlapping decode cycles and only keep the latest live state",
    )
    parser.add_argument(
        "--print-runtime-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print resolved model/vision devices and MPS availability at startup",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.stage3.live import run_live_webcam_inference

    run_live_webcam_inference(
        stage2_checkpoint_dir=args.stage2_checkpoint_dir,
        stage3_adapter_dir=args.stage3_adapter_dir,
        vision_checkpoint=args.vision_checkpoint,
        camera_index=args.camera_index,
        device=args.device,
        capture_fps=args.capture_fps,
        history_sec=args.history_sec,
        warmup_sec=args.warmup_sec,
        encode_interval_sec=args.encode_interval_sec,
        decode_interval_sec=args.decode_interval_sec,
        min_observation_sec=args.min_observation_sec,
        post_feedback_min_next_steps=args.post_feedback_min_next_steps,
        max_feedback_tokens=args.max_feedback_tokens,
        max_total_new_tokens=args.max_total_new_tokens,
        overlay_hold_sec=args.overlay_hold_sec,
        save_log=args.save_log,
        frame_timeout_sec=args.frame_timeout_sec,
        rotate_90_cw=args.rotate_90_cw,
        frame_queue_size=args.frame_queue_size,
        latest_only_decode=args.latest_only_decode,
        print_runtime_info=args.print_runtime_info,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
