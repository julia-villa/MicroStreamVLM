"""Stage-3 segment feature-cache helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch
from tqdm.auto import tqdm

from src.stage2.vision_encoder import FrozenEfficientNetStreamEncoder
from src.stage3.dataset import Stage3SegmentRecord, save_segment_manifest


def _safe_segment_filename(segment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", segment_id)


def get_segment_cache_path(cache_dir: str | Path, record: Stage3SegmentRecord) -> Path:
    """Return the stable cache file path for a segment record."""
    return Path(cache_dir) / record.split / f"{_safe_segment_filename(record.segment_id)}.pt"


def load_cached_segment(cache_path: str | Path) -> dict:
    """Load one cached segment feature blob."""
    return torch.load(Path(cache_path), map_location="cpu")


def build_segment_feature_cache(
    records: Sequence[Stage3SegmentRecord],
    vision_encoder: FrozenEfficientNetStreamEncoder,
    cache_dir: str | Path,
    overwrite: bool = False,
    manifest_path: str | Path | None = None,
    progress_label: str = "building stage3 cache",
) -> list[Stage3SegmentRecord]:
    """Encode and cache every segment record with the frozen vision encoder."""
    cache_dir_path = Path(cache_dir)
    cached_records: list[Stage3SegmentRecord] = []

    for record in tqdm(records, desc=progress_label):
        cache_path = get_segment_cache_path(cache_dir_path, record)
        if cache_path.exists() and not overwrite:
            cached = load_cached_segment(cache_path)
            feature_timestamps = cached.get("feature_timestamps", [])
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_segment = vision_encoder.encode_segment(
                video_path=record.video_path,
                timestamps_path=record.video_timestamps_path,
                start_timestamp_sec=record.exercise_start_timestamp,
                end_timestamp_sec=record.exercise_end_timestamp,
                rotate_90_cw=record.rotate_90_cw,
            )
            feature_timestamps = encoded_segment.get("feature_timestamps", [])
            torch.save(
                {
                    "segment_id": record.segment_id,
                    "split": record.split,
                    "video_id": record.video_id,
                    "exercise_name": record.exercise_name,
                    "exercise_start_timestamp": record.exercise_start_timestamp,
                    "exercise_end_timestamp": record.exercise_end_timestamp,
                    "feedbacks": list(record.feedbacks),
                    "feedback_timestamps": list(record.feedback_timestamps),
                    "feature_timestamps": list(feature_timestamps),
                    "spatial_res": encoded_segment["spatial_res"],
                    "feats": encoded_segment["feats"].cpu(),
                },
                cache_path,
            )

        cached_records.append(record.with_cache(cache_path, feature_timestamps))

    if manifest_path is not None:
        save_segment_manifest(cached_records, manifest_path)
    return cached_records
