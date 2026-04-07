"""Torch dataset wrappers plus backward-compatible stage-3 manifest re-exports."""

from __future__ import annotations

from typing import Any, Sequence

from torch.utils.data import Dataset

from src.stage3.manifest import (
    LEGACY_STAGE3_SYSTEM_PROMPT,
    STAGE3_STYLE_PROMPT,
    STAGE3_SYSTEM_PROMPT,
    Stage3SegmentRecord,
    build_stage3_text_prefix,
    extract_exercise_name,
    get_feedback_spans,
    limit_records,
    load_long_range_segments,
    load_segment_manifest,
    load_video_timestamps,
    resolve_stage3_system_prompt,
    save_segment_manifest,
    segment_long_range_record,
    split_train_validation_segments,
)


class Stage3SegmentDataset(Dataset):
    """Torch dataset wrapper for stage-3 segment records."""

    def __init__(self, records: Sequence[Stage3SegmentRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index].to_dict()


def stage3_collate(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate record dicts into the list-of-fields structure used by stage-2 utilities."""
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}


__all__ = [
    "LEGACY_STAGE3_SYSTEM_PROMPT",
    "STAGE3_STYLE_PROMPT",
    "STAGE3_SYSTEM_PROMPT",
    "Stage3SegmentDataset",
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
    "stage3_collate",
]
