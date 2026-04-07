"""Reusable stage-3 prediction-file evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.stage3.manifest import Stage3SegmentRecord, load_segment_manifest
from src.stage3.predictions import load_predictions


def _get_alignment_matrix(
    gt_feedback_timestamps: np.ndarray,
    pred_feedback_timestamps: np.ndarray,
    pred_feedbacks: list[str],
    tolerance: float = 3.0,
) -> tuple[list[int], list[int]]:
    matching_row_idxs, matching_col_idxs = [], []
    last_match_idx = -1
    for row_index, timestamp in enumerate(gt_feedback_timestamps):
        if len(pred_feedback_timestamps) == 0:
            break
        min_idx = int(np.argmin((pred_feedback_timestamps - timestamp) ** 2))
        if (
            np.abs(timestamp - pred_feedback_timestamps[min_idx]) < (tolerance / 2.0)
            and min_idx > last_match_idx
            and (min_idx not in matching_col_idxs)
            and pred_feedbacks[min_idx] != ""
        ):
            matching_row_idxs.append(row_index)
            matching_col_idxs.append(min_idx)
            last_match_idx = min_idx
    return matching_row_idxs, matching_col_idxs


def _get_temporally_aligned_feedbacks(
    gt_feedback_timestamps: list[float],
    pred_feedback_timestamps: list[float],
    gt_feedbacks: list[str],
    pred_feedbacks: list[str],
    tolerance: float = 3.0,
) -> tuple[int, int, list[tuple[str, str]]]:
    gt_feedback_timestamps_arr = np.array(gt_feedback_timestamps, dtype=np.float64)
    pred_feedback_timestamps_arr = np.array(pred_feedback_timestamps, dtype=np.float64)

    matched_feedbacks = []
    matched_idxs_gt = []
    matched_idxs_pred = []
    matching_row_idxs, matching_col_idxs = [], []
    if len(pred_feedback_timestamps_arr) > 0:
        matching_row_idxs, matching_col_idxs = _get_alignment_matrix(
            gt_feedback_timestamps_arr,
            pred_feedback_timestamps_arr,
            pred_feedbacks,
            tolerance=tolerance,
        )

    for match_idx, match_jdx in zip(matching_row_idxs, matching_col_idxs):
        matched_feedbacks.append((gt_feedbacks[match_idx], pred_feedbacks[match_jdx]))
        matched_idxs_gt.append(match_idx)
        matched_idxs_pred.append(match_jdx)

    return len(matched_idxs_gt), len(matched_idxs_pred), matched_feedbacks


def _compute_temporal_fscore_running(
    gt_feedbacks: list[str],
    pred_feedbacks: list[str],
    gt_feedback_timestamps: list[float],
    pred_feedback_timestamps: list[float],
    running_stats: dict[str, float],
    tolerance: float = 3.0,
) -> tuple[float, list[tuple[str, str]], dict[str, float]]:
    num_matched_gt, _, matched_feedbacks = _get_temporally_aligned_feedbacks(
        gt_feedback_timestamps,
        pred_feedback_timestamps,
        gt_feedbacks,
        pred_feedbacks,
        tolerance=tolerance,
    )
    _, num_matched_preds, _ = _get_temporally_aligned_feedbacks(
        pred_feedback_timestamps,
        gt_feedback_timestamps,
        pred_feedbacks,
        gt_feedbacks,
        tolerance=tolerance,
    )

    running_stats["total_matched_gt_feedbacks"] += num_matched_gt
    running_stats["total_matched_pred_feedbacks"] += num_matched_preds
    running_stats["total_num_gt_feedbacks"] += len(gt_feedbacks)
    running_stats["total_num_pred_feedbacks"] += len(pred_feedbacks)

    eps = 1e-12
    precision = running_stats["total_matched_pred_feedbacks"] / (
        running_stats["total_num_pred_feedbacks"] + eps
    )
    recall = running_stats["total_matched_gt_feedbacks"] / (
        running_stats["total_num_gt_feedbacks"] + eps
    )
    f_score = 2 * ((precision * recall) / (precision + recall + eps))
    return f_score, matched_feedbacks, running_stats


def _load_reference_records(
    references: str | Path | Sequence[Stage3SegmentRecord] | Sequence[dict[str, Any]],
) -> list[Stage3SegmentRecord]:
    if isinstance(references, (str, Path)):
        return load_segment_manifest(references)
    if not references:
        return []
    first_item = references[0]
    if isinstance(first_item, Stage3SegmentRecord):
        return list(references)
    return [Stage3SegmentRecord.from_dict(record) for record in references]


def evaluate_predictions(
    predictions: str | Path | Sequence[dict[str, Any]],
    references: str | Path | Sequence[Stage3SegmentRecord] | Sequence[dict[str, Any]],
    output_path: str | Path | None = None,
    tolerance: float = 3.0,
) -> dict[str, Any]:
    """Evaluate a stage-3 predictions file against reference segment records."""
    try:
        import evaluate
    except Exception as exc:
        raise RuntimeError(
            "The stage-3 metrics stack requires `evaluate` and a NumPy 1.x runtime. "
            "Install compatible packages with "
            "`pip install \"numpy<2\" evaluate rouge_score bert-score nltk datasets` "
            "and restart the kernel before rerunning the metrics cell."
        ) from exc

    prediction_records = load_predictions(predictions)
    reference_records = _load_reference_records(references)
    reference_lookup = {record.segment_id: record for record in reference_records}

    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    try:
        bert_metric = evaluate.load("bertscore")
    except (RuntimeError, ImportError, FileNotFoundError):
        bert_metric = None

    meteor_scores: list[float] = []
    rouge_scores: list[float] = []
    bert_scores: list[float] = []
    ttft_values: list[float] = []
    last_token_values: list[float] = []
    matched_feedback_examples: list[dict[str, Any]] = []
    running_stats = {
        "total_num_gt_feedbacks": 0.0,
        "total_num_pred_feedbacks": 0.0,
        "total_matched_gt_feedbacks": 0.0,
        "total_matched_pred_feedbacks": 0.0,
    }

    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_tokens = 0
    total_generation_wall_time = 0.0
    missing_reference_segment_ids: list[str] = []

    for prediction in prediction_records:
        segment_id = prediction["segment_id"]
        reference = reference_lookup.get(segment_id)
        if reference is None:
            missing_reference_segment_ids.append(segment_id)
            continue

        gt_feedbacks = list(reference.feedbacks)
        gt_feedback_timestamps = list(reference.feedback_timestamps)
        pred_feedbacks = [str(text) for text in prediction.get("pred_feedbacks", [])]
        pred_feedback_timestamps = [float(ts) for ts in prediction.get("pred_feedback_timestamps", [])]

        temporal_fscore, matched_feedbacks, running_stats = _compute_temporal_fscore_running(
            gt_feedbacks=gt_feedbacks,
            pred_feedbacks=pred_feedbacks,
            gt_feedback_timestamps=gt_feedback_timestamps,
            pred_feedback_timestamps=pred_feedback_timestamps,
            running_stats=running_stats,
            tolerance=tolerance,
        )

        for gt_feedback, pred_feedback in matched_feedbacks:
            meteor_scores.append(
                meteor_metric.compute(references=[gt_feedback], predictions=[pred_feedback])["meteor"]
            )
            rouge_scores.append(
                rouge_metric.compute(references=[gt_feedback], predictions=[pred_feedback])["rougeL"]
            )
            if bert_metric is not None:
                bert_scores.extend(
                    bert_metric.compute(
                        references=[gt_feedback],
                        predictions=[pred_feedback],
                        lang="en",
                    )["f1"]
                )
            matched_feedback_examples.append(
                {
                    "segment_id": segment_id,
                    "temporal_fscore_running": temporal_fscore,
                    "gt_feedback": gt_feedback,
                    "pred_feedback": pred_feedback,
                }
            )

        for timing_event in prediction.get("timing_events", []):
            if timing_event.get("ttft_sec") is not None:
                ttft_values.append(float(timing_event["ttft_sec"]))
            if timing_event.get("time_to_last_token_sec") is not None:
                last_token_values.append(float(timing_event["time_to_last_token_sec"]))

        total_prompt_tokens += int(prediction.get("prompt_tokens", 0))
        total_generated_tokens += int(prediction.get("generated_tokens", 0))
        total_tokens += int(prediction.get("total_tokens", 0))
        total_generation_wall_time += float(prediction.get("generation_wall_time_sec", 0.0))

    eps = 1e-12
    precision = running_stats["total_matched_pred_feedbacks"] / (
        running_stats["total_num_pred_feedbacks"] + eps
    )
    recall = running_stats["total_matched_gt_feedbacks"] / (
        running_stats["total_num_gt_feedbacks"] + eps
    )
    temporal_f_score = 2 * ((precision * recall) / (precision + recall + eps))
    metrics = {
        "num_prediction_segments": len(prediction_records),
        "num_reference_segments": len(reference_records),
        "missing_reference_segment_ids": missing_reference_segment_ids,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "rougeL": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "bert_score": float(np.mean(bert_scores)) if bert_scores else None,
        "temporal_f_score": float(temporal_f_score),
        "mean_ttft_sec": float(np.mean(ttft_values)) if ttft_values else 0.0,
        "mean_time_to_last_token_sec": float(np.mean(last_token_values)) if last_token_values else 0.0,
        "tokens_per_second": float(total_generated_tokens / (total_generation_wall_time + eps)),
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "total_tokens": total_tokens,
            "mean_prompt_tokens": float(total_prompt_tokens / max(1, len(prediction_records))),
            "mean_generated_tokens": float(total_generated_tokens / max(1, len(prediction_records))),
            "mean_total_tokens": float(total_tokens / max(1, len(prediction_records))),
        },
        "matched_feedback_examples": matched_feedback_examples[:100],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
