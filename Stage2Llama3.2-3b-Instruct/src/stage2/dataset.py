"""Stage-2 dataset helpers for xattn-only StreamVLM training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert fitness coaching AI who coaches users as they exercise. "
    "You observe them silently, assess their performance, and answer any questions they have."
)
FEEDBACK_REQUEST = "Please provide a feedback for the user."
SUPPORTED_TASK_TYPES = {"feedback", "qa"}
SUPPORTED_SPLITS = {"train", "test"}
QUESTION_SECTIONS = ("high_level", "fine_grain")
MISMATCH_RATE_ERROR_THRESHOLD = 0.05


@dataclass(frozen=True)
class Stage2Sample:
    """Single stage-2 training sample."""

    sample_id: str
    video_path: str
    task_type: str
    answer: str
    question: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rotate_90_cw: bool = False
    annotation_source: str | None = None
    query_type: str | None = None
    labels: list[str] | None = None
    labels_descriptive: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for DataLoader compatibility."""
        return {
            "sample_id": self.sample_id,
            "video_path": self.video_path,
            "task_type": self.task_type,
            "answer": self.answer,
            "question": self.question,
            "system_prompt": self.system_prompt,
            "rotate_90_cw": self.rotate_90_cw,
            "annotation_source": self.annotation_source,
            "query_type": self.query_type,
            "labels": self.labels,
            "labels_descriptive": self.labels_descriptive,
        }


@dataclass(frozen=True)
class ClipRecord:
    """Registry entry for one short clip."""

    video_path_key: str
    absolute_video_path: str
    split: str
    labels: list[str]
    labels_descriptive: list[str]


def _normalize_video_path(video_path: str) -> str:
    """Normalize path keys across annotation files."""
    path = str(Path(video_path).as_posix())
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("short_clips/"):
        path = path[len("short_clips/") :]
    return path


def _load_json(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a top-level list")
    return data


def get_request_text(sample: Stage2Sample | dict[str, Any]) -> str:
    """Return the request/question text for a training sample."""
    task_type = sample.task_type if isinstance(sample, Stage2Sample) else sample["task_type"]
    if task_type == "feedback":
        return FEEDBACK_REQUEST

    question = sample.question if isinstance(sample, Stage2Sample) else sample.get("question")
    if not question:
        raise ValueError("qa sample is missing a question")
    return str(question)


class Stage2QEVDFit300KDataset(Dataset):
    """Native QEVD-FIT-300K short-clip dataset for stage-2 training."""

    def __init__(self, data_root: str | Path, split: str = "train") -> None:
        self.data_root = Path(data_root)
        self.short_clips_root = self.data_root / "short_clips"
        self.split = split.strip().lower()
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(SUPPORTED_SPLITS)}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        if not self.short_clips_root.exists():
            raise FileNotFoundError(f"short_clips directory not found: {self.short_clips_root}")

        self.stats = {
            "feedback_missing_video_paths": 0,
            "questions_missing_video_paths": 0,
            "feedback_records_total": 0,
            "question_records_total": 0,
        }

        self.clip_registry = self._build_clip_registry()
        self.samples = self._build_samples()

    def _build_clip_registry(self) -> dict[str, ClipRecord]:
        labels_path = self.data_root / "fine_grained_labels.json"
        entries = _load_json(labels_path)
        registry: dict[str, ClipRecord] = {}

        for item in entries:
            video_path_raw = item.get("video_path")
            if not video_path_raw:
                continue

            item_split = str(item.get("split", "")).strip().lower()
            if item_split != self.split:
                continue

            normalized_key = _normalize_video_path(video_path_raw)
            absolute_video_path = self.short_clips_root / normalized_key
            if not absolute_video_path.exists():
                raise FileNotFoundError(
                    f"Video from fine_grained_labels.json not found under short_clips/: "
                    f"{absolute_video_path}"
                )

            registry[normalized_key] = ClipRecord(
                video_path_key=normalized_key,
                absolute_video_path=str(absolute_video_path),
                split=item_split,
                labels=[str(label) for label in item.get("labels", [])],
                labels_descriptive=[str(label) for label in item.get("labels_descriptive", [])],
            )

        if not registry:
            raise ValueError(
                f"No clips found for split={self.split!r} in {labels_path}"
            )

        return registry

    def _make_sample(
        self,
        clip_record: ClipRecord,
        task_type: str,
        answer: str,
        question: str | None,
        annotation_source: str,
        query_type: str | None = None,
        sample_suffix: str = "",
    ) -> Stage2Sample:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task_type={task_type!r}")

        sample_id = f"{clip_record.video_path_key}:{annotation_source}{sample_suffix}"
        return Stage2Sample(
            sample_id=sample_id,
            video_path=clip_record.absolute_video_path,
            task_type=task_type,
            answer=answer.strip(),
            question=question.strip() if question is not None else None,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            rotate_90_cw=False,
            annotation_source=annotation_source,
            query_type=query_type,
            labels=clip_record.labels,
            labels_descriptive=clip_record.labels_descriptive,
        )

    def _expand_feedback_samples(self) -> list[Stage2Sample]:
        feedbacks_path = self.data_root / "feedbacks_short_clips.json"
        entries = _load_json(feedbacks_path)
        samples: list[Stage2Sample] = []

        for item in entries:
            self.stats["feedback_records_total"] += 1
            item_split = str(item.get("split", "")).strip().lower()
            if item_split != self.split:
                continue

            video_path_raw = item.get("video_path")
            if not video_path_raw:
                continue

            normalized_key = _normalize_video_path(video_path_raw)
            clip_record = self.clip_registry.get(normalized_key)
            if clip_record is None:
                self.stats["feedback_missing_video_paths"] += 1
                continue

            for feedback_index, feedback_text in enumerate(item.get("feedback", [])):
                feedback_text = str(feedback_text).strip()
                if not feedback_text:
                    continue
                samples.append(
                    self._make_sample(
                        clip_record=clip_record,
                        task_type="feedback",
                        answer=feedback_text,
                        question=None,
                        annotation_source="feedbacks_short_clips",
                        sample_suffix=f":{feedback_index}",
                    )
                )

        return samples

    def _expand_question_section(
        self,
        clip_record: ClipRecord,
        section_name: str,
        section_data: dict[str, Any],
    ) -> list[Stage2Sample]:
        queries = [str(query).strip() for query in section_data.get("query", [])]
        responses = [str(response).strip() for response in section_data.get("response", [])]
        query_types = [str(query_type).strip() for query_type in section_data.get("query_type", [])]

        if not (len(queries) == len(responses) == len(query_types)):
            raise ValueError(
                f"questions.json has misaligned arrays for {clip_record.video_path_key} "
                f"section={section_name}"
            )

        samples: list[Stage2Sample] = []
        for qa_index, (query, response, query_type) in enumerate(zip(queries, responses, query_types)):
            if not query or not response:
                continue
            samples.append(
                self._make_sample(
                    clip_record=clip_record,
                    task_type="qa",
                    answer=response,
                    question=query,
                    annotation_source=f"questions.{section_name}",
                    query_type=query_type,
                    sample_suffix=f":{qa_index}",
                )
            )

        return samples

    def _expand_question_samples(self) -> list[Stage2Sample]:
        questions_path = self.data_root / "questions.json"
        entries = _load_json(questions_path)
        samples: list[Stage2Sample] = []

        for item in entries:
            self.stats["question_records_total"] += 1
            item_split = str(item.get("split", "")).strip().lower()
            if item_split != self.split:
                continue

            video_path_raw = item.get("video_path")
            if not video_path_raw:
                continue

            normalized_key = _normalize_video_path(video_path_raw)
            clip_record = self.clip_registry.get(normalized_key)
            if clip_record is None:
                self.stats["questions_missing_video_paths"] += 1
                continue

            for section_name in QUESTION_SECTIONS:
                section_data = item.get(section_name, {})
                if not isinstance(section_data, dict):
                    continue
                samples.extend(self._expand_question_section(clip_record, section_name, section_data))

        return samples

    def _validate_join_mismatch_rate(self) -> None:
        total_annotations = (
            self.stats["feedback_records_total"] + self.stats["question_records_total"]
        )
        total_missing = (
            self.stats["feedback_missing_video_paths"] + self.stats["questions_missing_video_paths"]
        )
        if total_annotations == 0:
            raise ValueError("No feedback/question annotation records were found for the requested split")

        mismatch_rate = total_missing / total_annotations
        if mismatch_rate > MISMATCH_RATE_ERROR_THRESHOLD:
            raise ValueError(
                "Annotation/video_path join mismatch rate is too high: "
                f"{total_missing}/{total_annotations} ({mismatch_rate:.2%}). "
                "Check that video_path values match fine_grained_labels.json and short_clips/."
            )

    def _build_samples(self) -> list[Stage2Sample]:
        samples = self._expand_feedback_samples()
        samples.extend(self._expand_question_samples())
        self._validate_join_mismatch_rate()

        if not samples:
            raise ValueError(
                f"No stage-2 samples built for split={self.split!r} under {self.data_root}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index].to_dict()


def stage2_collate(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Collate dataset rows into a dict of lists."""
    if not samples:
        return {}

    collated: dict[str, list[Any]] = {key: [] for key in samples[0]}
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated
