"""Standalone stage-3 segment manifest and segmentation utilities."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LEGACY_STAGE3_SYSTEM_PROMPT = (
    "<system> You are an expert fitness coaching AI who coaches users as they exercise. "
    "You assess their performance, and proactively provide feedback. </system>"
)

SHORT_STAGE3_SYSTEM_PROMPT = (
    "<system> You are a fitness coach. Watch the exercise and give short, specific coaching feedback. </system>"
)

STAGE3_SYSTEM_PROMPT = LEGACY_STAGE3_SYSTEM_PROMPT
STAGE3_STYLE_PROMPT = ""


def resolve_stage3_system_prompt(system_prompt: str | None = None) -> str:
    prompt = (system_prompt or "").strip()
    if (
        not prompt
        or prompt == LEGACY_STAGE3_SYSTEM_PROMPT
        or prompt == SHORT_STAGE3_SYSTEM_PROMPT
    ):
        return STAGE3_SYSTEM_PROMPT
    return prompt


def build_stage3_text_prefix(system_prompt: str | None = None) -> str:
    return f"{resolve_stage3_system_prompt(system_prompt)}\n"


def load_video_timestamps(file_path: str | Path) -> np.ndarray:
    """Load frame timestamps from the saved `.npy` file and convert to seconds."""
    timestamps = np.load(file_path).astype(np.double)
    timestamps = np.array([int(xx / 1e9) + (xx / 1e9) % 1 for xx in timestamps]) + 28800.0
    return timestamps


@dataclass(frozen=True)
class Stage3SegmentRecord:
    """Single long-range exercise segment with optional cached feature metadata."""

    segment_id: str
    split: str
    video_id: str
    video_path: str
    video_timestamps_path: str
    exercise_name: str
    exercise_start_timestamp: float
    exercise_end_timestamp: float
    system_prompt: str
    feedbacks: tuple[str, ...]
    feedback_timestamps: tuple[float, ...]
    rotate_90_cw: bool = False
    cached_features_path: str | None = None
    feature_timestamps: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["feedbacks"] = list(self.feedbacks)
        record["feedback_timestamps"] = list(self.feedback_timestamps)
        record["feature_timestamps"] = list(self.feature_timestamps)
        return record

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Stage3SegmentRecord":
        return cls(
            segment_id=payload["segment_id"],
            split=payload["split"],
            video_id=payload["video_id"],
            video_path=payload["video_path"],
            video_timestamps_path=payload["video_timestamps_path"],
            exercise_name=payload["exercise_name"],
            exercise_start_timestamp=float(payload["exercise_start_timestamp"]),
            exercise_end_timestamp=float(payload["exercise_end_timestamp"]),
            system_prompt=resolve_stage3_system_prompt(payload.get("system_prompt", STAGE3_SYSTEM_PROMPT)),
            feedbacks=tuple(payload.get("feedbacks", [])),
            feedback_timestamps=tuple(float(x) for x in payload.get("feedback_timestamps", [])),
            rotate_90_cw=bool(payload.get("rotate_90_cw", False)),
            cached_features_path=payload.get("cached_features_path"),
            feature_timestamps=tuple(float(x) for x in payload.get("feature_timestamps", [])),
        )

    def with_cache(
        self,
        cached_features_path: str | Path,
        feature_timestamps: Sequence[float],
    ) -> "Stage3SegmentRecord":
        return Stage3SegmentRecord(
            segment_id=self.segment_id,
            split=self.split,
            video_id=self.video_id,
            video_path=self.video_path,
            video_timestamps_path=self.video_timestamps_path,
            exercise_name=self.exercise_name,
            exercise_start_timestamp=self.exercise_start_timestamp,
            exercise_end_timestamp=self.exercise_end_timestamp,
            system_prompt=self.system_prompt,
            feedbacks=self.feedbacks,
            feedback_timestamps=self.feedback_timestamps,
            rotate_90_cw=self.rotate_90_cw,
            cached_features_path=str(cached_features_path),
            feature_timestamps=tuple(float(x) for x in feature_timestamps),
        )


def extract_exercise_name(transition_feedback: str) -> str:
    """Extract the exercise name from a transition feedback string."""
    return (
        transition_feedback.replace("First up are ", "")
        .replace("Moving on to ", "")
        .replace("!", "")
        .strip()
    )


def get_feedback_spans(
    feedbacks: Sequence[str],
) -> list[tuple[str, int, int]]:
    """Collapse a dense per-frame feedback sequence into temporally grouped spans."""
    feedback_spans: list[tuple[str, int, int]] = []
    current_feedback = None
    start_idx = None

    for index, feedback in enumerate(feedbacks):
        if current_feedback is not None and current_feedback != feedback:
            feedback_spans.append((current_feedback, int(start_idx), index - 1))
            current_feedback = None
            start_idx = None
        if feedback and not current_feedback:
            current_feedback = feedback
            start_idx = index

    if current_feedback is not None and start_idx is not None:
        feedback_spans.append((current_feedback, int(start_idx), len(feedbacks) - 1))

    return feedback_spans


def _resolve_record_paths(
    video_dir: str | Path,
    relative_video_path: str,
    relative_timestamps_path: str,
) -> tuple[str, str]:
    video_dir_path = Path(video_dir)
    video_name = Path(relative_video_path).name
    timestamps_name = Path(relative_timestamps_path).name
    video_path = video_dir_path / video_name
    timestamps_path = video_dir_path / timestamps_name
    if not video_path.exists():
        raise FileNotFoundError(f"Long-range video does not exist: {video_path}")
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Long-range timestamp file does not exist: {timestamps_path}")
    return str(video_path), str(timestamps_path)


def _normalize_relative_timestamps(
    timestamps: Sequence[float],
    start_timestamp: float,
) -> tuple[float, ...]:
    return tuple(float(timestamp - start_timestamp) for timestamp in timestamps)


def segment_long_range_record(
    record: dict[str, Any],
    video_dir: str | Path,
    split: str,
    system_prompt: str = STAGE3_SYSTEM_PROMPT,
) -> list[Stage3SegmentRecord]:
    """Split one long-range workout record into exercise-length segment records."""
    video_path, video_timestamps_path = _resolve_record_paths(
        video_dir=video_dir,
        relative_video_path=record["long_range_video_file"],
        relative_timestamps_path=record["video_timestamps"],
    )
    feedbacks = [feedback.replace("armcrosschest", "deltoid stretch") for feedback in record["feedbacks"]]
    feedback_timestamps = [float(timestamp) for timestamp in record["feedback_timestamps"]]
    feedback_spans = get_feedback_spans(feedbacks)
    transition_flags = list(record["is_transition"])

    if len(feedback_spans) != len(feedback_timestamps) or len(feedback_spans) != len(transition_flags):
        video_name = Path(video_path).name
        warnings.warn(
            "Skipping malformed long-range record because grouped feedback spans do not align "
            f"with timestamps/transitions: video={video_name}, "
            f"spans={len(feedback_spans)}, timestamps={len(feedback_timestamps)}, "
            f"transitions={len(transition_flags)}"
        )
        return []

    transition_indices = np.where(np.array(transition_flags, dtype=bool))[0].tolist()
    if len(transition_indices) < 2:
        return []

    video_timestamps = load_video_timestamps(video_timestamps_path)
    exercises = [extract_exercise_name(feedback_spans[idx][0]) for idx in transition_indices][:-1]

    segments: list[Stage3SegmentRecord] = []
    for segment_index in range(len(transition_indices) - 1):
        current_transition = feedback_spans[transition_indices[segment_index]]
        next_transition = feedback_spans[transition_indices[segment_index + 1]]
        segment_feedbacks = [
            span[0]
            for span in feedback_spans[
                (transition_indices[segment_index] + 1) : transition_indices[segment_index + 1]
            ]
        ]
        segment_feedback_timestamps = feedback_timestamps[
            (transition_indices[segment_index] + 1) : transition_indices[segment_index + 1]
        ]
        if not segment_feedbacks:
            continue

        start_frame_idx = current_transition[1]
        end_frame_idx = next_transition[1] + (next_transition[2] - next_transition[1]) // 2
        start_timestamp = float(video_timestamps[start_frame_idx])
        end_timestamp = float(video_timestamps[end_frame_idx])
        video_id = Path(video_path).stem
        segment_id = f"{split}:{video_id}:{segment_index:03d}"
        exercise_name = exercises[segment_index] if segment_index < len(exercises) else "unknown"

        segments.append(
            Stage3SegmentRecord(
                segment_id=segment_id,
                split=split,
                video_id=video_id,
                video_path=video_path,
                video_timestamps_path=video_timestamps_path,
                exercise_name=exercise_name,
                exercise_start_timestamp=start_timestamp,
                exercise_end_timestamp=end_timestamp,
                system_prompt=system_prompt,
                feedbacks=tuple(segment_feedbacks),
                feedback_timestamps=_normalize_relative_timestamps(segment_feedback_timestamps, start_timestamp),
            )
        )

    return segments


def load_long_range_segments(
    metadata_path: str | Path,
    video_dir: str | Path,
    split: str,
    system_prompt: str = STAGE3_SYSTEM_PROMPT,
) -> list[Stage3SegmentRecord]:
    """Load one split of long-range metadata and expand it into single-exercise segments."""
    metadata = json.loads(Path(metadata_path).read_text())
    segments: list[Stage3SegmentRecord] = []
    for record in metadata:
        segments.extend(
            segment_long_range_record(
                record=record,
                video_dir=video_dir,
                split=split,
                system_prompt=system_prompt,
            )
        )
    return segments


def split_train_validation_segments(
    records: Sequence[Stage3SegmentRecord],
    val_fraction: float = 0.1,
    seed: int = 469,
    max_train_segments: int | None = None,
    max_val_segments: int | None = 128,
) -> tuple[list[Stage3SegmentRecord], list[Stage3SegmentRecord]]:
    """Create a deterministic train/validation split from stage-3 segment records."""
    if not records:
        return [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    requested_val_count = int(math.ceil(len(records) * val_fraction))
    val_count = max(1, requested_val_count) if len(records) > 1 else 0
    if max_val_segments is not None:
        val_count = min(val_count, max_val_segments)

    val_indices = indices[:val_count].tolist()
    train_indices = indices[val_count:].tolist()

    if max_train_segments is not None:
        train_indices = train_indices[:max_train_segments]

    train_records = [records[idx] for idx in train_indices]
    val_records = [records[idx] for idx in val_indices]
    return train_records, val_records


def limit_records(
    records: Sequence[Stage3SegmentRecord],
    max_records: int | None,
) -> list[Stage3SegmentRecord]:
    """Apply an optional cap to a record list without changing record order."""
    if max_records is None:
        return list(records)
    return list(records[:max_records])


def save_segment_manifest(records: Sequence[Stage3SegmentRecord], output_path: str | Path) -> None:
    """Save segment records to a JSON manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([record.to_dict() for record in records], indent=2), encoding="utf-8")


def load_segment_manifest(manifest_path: str | Path) -> list[Stage3SegmentRecord]:
    """Load segment records from a JSON manifest."""
    payload = json.loads(Path(manifest_path).read_text())
    return [Stage3SegmentRecord.from_dict(record) for record in payload]


__all__ = [
    "LEGACY_STAGE3_SYSTEM_PROMPT",
    "STAGE3_STYLE_PROMPT",
    "STAGE3_SYSTEM_PROMPT",
    "Stage3SegmentRecord",
    "build_stage3_text_prefix",
    "extract_exercise_name",
    "get_feedback_spans",
    "limit_records",
    "load_long_range_segments",
    "load_segment_manifest",
    "load_video_timestamps",
    "resolve_stage3_system_prompt",
    "save_segment_manifest",
    "segment_long_range_record",
    "split_train_validation_segments",
]
