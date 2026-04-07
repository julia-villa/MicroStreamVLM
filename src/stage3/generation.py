"""Stage-3 benchmark prediction generation utilities."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from src.chat_format import (
    build_chat_aligned_prefix,
    contains_role_leakage,
    get_instruct_role_guard_token_ids,
    trim_role_leakage_text,
)
from src.constants import INFERENCE_SPEED
from src.stage3.cache import load_cached_segment
from src.stage3.manifest import (
    STAGE3_SYSTEM_PROMPT,
    Stage3SegmentRecord,
    build_stage3_text_prefix,
    limit_records,
    resolve_stage3_system_prompt,
)
from src.stage3.predictions import load_predictions, save_predictions


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _build_generation_prefix(system_prompt: str, model) -> list[int]:
    model_name_or_path = getattr(model, "base_llm_model_name_or_path", None)
    if model_name_or_path and "instruct" in model_name_or_path.lower():
        prefix_ids = build_chat_aligned_prefix(
            tokenizer=model.tokenizer,
            system_prompt=system_prompt,
            user_content="",
            model_name_or_path=model_name_or_path,
            add_generation_prompt=True,
        )
    else:
        bos_ids = [model.tokenizer.bos_token_id] if model.tokenizer.bos_token_id is not None else []
        system_ids = _encode_text(model.tokenizer, build_stage3_text_prefix(system_prompt))
        prefix_ids = bos_ids + system_ids
    return prefix_ids + [model.timeline_token_ids["next"]]


def _load_generation_inputs(
    model,
    sample: Stage3SegmentRecord,
) -> tuple[torch.Tensor, list[float], list[int], list[int]]:
    if sample.cached_features_path is None:
        raise ValueError(f"Sample {sample.segment_id} is missing cached_features_path")

    cached = load_cached_segment(sample.cached_features_path)
    feature_timestamps = [float(x) for x in cached.get("feature_timestamps", sample.feature_timestamps)]
    if not feature_timestamps:
        raise ValueError(f"Cached benchmark segment {sample.segment_id} has no feature timestamps")

    spatial_res = cached.get("spatial_res", [5, 7])
    full_vision_feats = cached["feats"].to(model.device)
    prefix_ids = _build_generation_prefix(sample.system_prompt, model)
    return full_vision_feats, feature_timestamps, spatial_res, prefix_ids


def _append_token(
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_xattn_mask: torch.Tensor,
    token_id: int,
    vision_mask_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=generated_ids.device)
    generated_ids = torch.cat([generated_ids, token_tensor], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(token_tensor)], dim=1)
    vision_xattn_mask = torch.cat(
        [vision_xattn_mask, torch.full_like(token_tensor, vision_mask_value)],
        dim=1,
    )
    return generated_ids, attention_mask, vision_xattn_mask


def _predict_next_token(
    model,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_xattn_mask: torch.Tensor,
    visible_vision_feats: torch.Tensor,
    spatial_res: list[int],
    allowed_token_ids: Sequence[int] | None = None,
    blocked_token_ids: Sequence[int] | None = None,
    forbidden_token_ids: Sequence[int] | None = None,
    repetition_penalty: float = 1.0,
    penalized_token_ids: Sequence[int] | None = None,
    token_biases: dict[int, float] | None = None,
) -> int:
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        vision_feats={"feats": visible_vision_feats, "spatial_res": spatial_res},
        vision_xattn_mask=vision_xattn_mask,
        labels=None,
    )
    logits = outputs.logits[:, -1, :]
    masked_logits = _apply_generation_logit_mask(
        logits,
        allowed_token_ids=allowed_token_ids,
        blocked_token_ids=blocked_token_ids,
        forbidden_token_ids=forbidden_token_ids,
        repetition_penalty=repetition_penalty,
        penalized_token_ids=penalized_token_ids,
        token_biases=token_biases,
    )
    return int(masked_logits.argmax(dim=-1)[0].item())


def _apply_generation_logit_mask(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int] | None = None,
    blocked_token_ids: Sequence[int] | None = None,
    forbidden_token_ids: Sequence[int] | None = None,
    repetition_penalty: float = 1.0,
    penalized_token_ids: Sequence[int] | None = None,
    token_biases: dict[int, float] | None = None,
) -> torch.Tensor:
    masked_logits = logits.clone()
    if allowed_token_ids is not None:
        allowed_tensor = torch.tensor(list(allowed_token_ids), device=logits.device, dtype=torch.long)
        keep_mask = torch.zeros_like(masked_logits, dtype=torch.bool)
        keep_mask[:, allowed_tensor] = True
        masked_logits = masked_logits.masked_fill(~keep_mask, float("-inf"))
    if blocked_token_ids is not None:
        blocked_tensor = torch.tensor(list(blocked_token_ids), device=logits.device, dtype=torch.long)
        masked_logits[:, blocked_tensor] = float("-inf")
    if forbidden_token_ids is not None:
        forbidden_tensor = torch.tensor(list(forbidden_token_ids), device=logits.device, dtype=torch.long)
        masked_logits[:, forbidden_tensor] = float("-inf")
    if repetition_penalty > 1.0 and penalized_token_ids:
        penalized_tensor = torch.tensor(
            sorted(set(int(token_id) for token_id in penalized_token_ids)),
            device=logits.device,
            dtype=torch.long,
        )
        penalized_logits = masked_logits[:, penalized_tensor]
        masked_logits[:, penalized_tensor] = torch.where(
            penalized_logits > 0,
            penalized_logits / repetition_penalty,
            penalized_logits * repetition_penalty,
        )
    if token_biases:
        for token_id, bias in token_biases.items():
            masked_logits[:, int(token_id)] += float(bias)
    return masked_logits


def _predict_next_token_with_topk(
    model,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_xattn_mask: torch.Tensor,
    visible_vision_feats: torch.Tensor,
    spatial_res: list[int],
    topk: int = 5,
    allowed_token_ids: Sequence[int] | None = None,
    blocked_token_ids: Sequence[int] | None = None,
    forbidden_token_ids: Sequence[int] | None = None,
    repetition_penalty: float = 1.0,
    penalized_token_ids: Sequence[int] | None = None,
    token_biases: dict[int, float] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        vision_feats={"feats": visible_vision_feats, "spatial_res": spatial_res},
        vision_xattn_mask=vision_xattn_mask,
        labels=None,
    )
    logits = outputs.logits[:, -1, :]
    masked_logits = _apply_generation_logit_mask(
        logits,
        allowed_token_ids=allowed_token_ids,
        blocked_token_ids=blocked_token_ids,
        forbidden_token_ids=forbidden_token_ids,
        repetition_penalty=repetition_penalty,
        penalized_token_ids=penalized_token_ids,
        token_biases=token_biases,
    )
    probs = torch.softmax(masked_logits, dim=-1)
    if allowed_token_ids is not None:
        topk = min(topk, len(tuple(allowed_token_ids)))
    top_values, top_indices = torch.topk(probs[0], k=min(topk, probs.shape[-1]))
    top_tokens = [
        {
            "token_id": int(token_id),
            "token": model.tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "prob": float(prob),
        }
        for token_id, prob in zip(top_indices.tolist(), top_values.tolist())
    ]
    return int(masked_logits.argmax(dim=-1)[0].item()), top_tokens


def _observation_allowed_token_ids(model, include_next: bool = True) -> list[int]:
    token_ids = [model.timeline_token_ids["feedback_begin"]]
    if include_next:
        token_ids.insert(0, model.timeline_token_ids["next"])
    return token_ids


def _feedback_blocked_token_ids(model) -> list[int]:
    feedback_end_id = model.timeline_token_ids["feedback_end"]
    blocked_ids = set(getattr(model.tokenizer, "all_special_ids", []))
    blocked_ids.update(
        [
            model.timeline_token_ids["next"],
            model.timeline_token_ids["feedback_begin"],
        ]
    )
    blocked_ids.discard(feedback_end_id)
    return sorted(int(token_id) for token_id in blocked_ids)


def _estimate_feats_frequency(feature_timestamps: list[float]) -> float:
    if len(feature_timestamps) < 2:
        return 1.0

    deltas = []
    for left, right in zip(feature_timestamps[:-1], feature_timestamps[1:]):
        delta = float(right - left)
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return 1.0
    return 1.0 / (sum(deltas) / len(deltas))


def _observed_visible_sec(
    feature_timestamps: list[float],
    selected_step_indices: list[int],
) -> float:
    if not feature_timestamps or not selected_step_indices:
        return 0.0
    last_visible_idx = selected_step_indices[-1]
    if last_visible_idx < 0:
        return 0.0
    return float(feature_timestamps[min(last_visible_idx, len(feature_timestamps) - 1)])


def _feedback_no_repeat_ngram_forbidden_tokens(
    generated_feedback_token_ids: Sequence[int],
    no_repeat_ngram_size: int,
) -> list[int]:
    if no_repeat_ngram_size <= 1 or len(generated_feedback_token_ids) < (no_repeat_ngram_size - 1):
        return []
    prefix = tuple(generated_feedback_token_ids[-(no_repeat_ngram_size - 1) :])
    forbidden = set()
    limit = len(generated_feedback_token_ids) - no_repeat_ngram_size + 1
    for index in range(max(0, limit)):
        if tuple(generated_feedback_token_ids[index : index + no_repeat_ngram_size - 1]) == prefix:
            forbidden.add(int(generated_feedback_token_ids[index + no_repeat_ngram_size - 1]))
    return sorted(forbidden)


def _resolve_soft_feedback_token_budget(model, soft_feedback_token_budget: int | None) -> int:
    if soft_feedback_token_budget is not None:
        return int(soft_feedback_token_budget)
    return 24


def _extract_pred_feedbacks(
    output_ids: Sequence[int],
    model,
    feats_frequency: float,
) -> tuple[list[str], list[float]]:
    next_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]

    try:
        first_next_idx = list(output_ids).index(next_id)
    except ValueError:
        return [], []

    output = list(output_ids)[first_next_idx:]
    feedback_begin_idxs = [idx for idx, token_id in enumerate(output) if token_id == feedback_begin_id]
    feedback_end_idxs = [idx for idx, token_id in enumerate(output) if token_id == feedback_end_id]

    responses: list[str] = []
    timestamps: list[float] = []
    cumulative_timestamp = 0.0
    previous_answer_generation_time = 0.0

    for idx in range(min(len(feedback_begin_idxs), len(feedback_end_idxs))):
        answer_begin_idx = feedback_begin_idxs[idx]
        answer_end_idx = feedback_end_idxs[idx]
        previous_answer_end_idx = feedback_end_idxs[idx - 1] if idx > 0 else -1

        response = model.tokenizer.decode(
            output[answer_begin_idx + 1 : answer_end_idx],
            skip_special_tokens=True,
        ).strip()
        response = trim_role_leakage_text(response)
        if not response:
            previous_answer_generation_time = 0.0
            continue

        timestep = (answer_begin_idx - previous_answer_end_idx - 1) / max(feats_frequency, 1e-6)
        timestep += previous_answer_generation_time
        cumulative_timestamp += timestep

        responses.append(response)
        timestamps.append(float(cumulative_timestamp))
        previous_answer_generation_time = (answer_end_idx - answer_begin_idx) / INFERENCE_SPEED

    return responses, timestamps


def _build_debug_payload(
    output_ids: Sequence[int],
    model,
    first_feedback_visible_step: int | None = None,
    first_feedback_visible_sec: float | None = None,
    feedback_blocked_before_threshold: bool = False,
    min_observation_sec: float = 0.0,
    soft_feedback_token_budget: int = 20,
    feedback_stop_events: Sequence[dict[str, Any]] | None = None,
    feedback_followed_by_next: bool = False,
    cooldown_blocked_immediate_reopen: bool = False,
    post_feedback_min_next_steps: int = 0,
    role_leakage_detected: bool = False,
) -> dict[str, Any]:
    next_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]

    output_ids = list(output_ids)
    try:
        first_next_idx = output_ids.index(next_id)
    except ValueError:
        first_next_idx = 0

    stream_output_ids = output_ids[first_next_idx:]
    action_trace = []
    for index, token_id in enumerate(stream_output_ids):
        token_label = None
        if token_id == next_id:
            token_label = "<next>"
        elif token_id == feedback_begin_id:
            token_label = "<feedback>"
        elif token_id == feedback_end_id:
            token_label = "</feedback>"
        if token_label is not None:
            action_trace.append({"stream_index": index, "token": token_label, "token_id": int(token_id)})

    return {
        "raw_output_ids": output_ids,
        "raw_decoded_text": model.tokenizer.decode(output_ids, skip_special_tokens=False),
        "raw_stream_ids": stream_output_ids,
        "raw_stream_text": model.tokenizer.decode(stream_output_ids, skip_special_tokens=False),
        "action_token_counts": {
            "next": sum(token_id == next_id for token_id in stream_output_ids),
            "feedback_begin": sum(token_id == feedback_begin_id for token_id in stream_output_ids),
            "feedback_end": sum(token_id == feedback_end_id for token_id in stream_output_ids),
        },
        "action_trace": action_trace,
        "first_feedback_visible_step": first_feedback_visible_step,
        "first_feedback_visible_sec": first_feedback_visible_sec,
        "feedback_blocked_before_threshold": bool(feedback_blocked_before_threshold),
        "min_observation_sec": float(min_observation_sec),
        "soft_feedback_token_budget": int(soft_feedback_token_budget),
        "feedback_stop_events": list(feedback_stop_events or []),
        "feedback_followed_by_next": bool(feedback_followed_by_next),
        "cooldown_blocked_immediate_reopen": bool(cooldown_blocked_immediate_reopen),
        "post_feedback_min_next_steps": int(post_feedback_min_next_steps),
        "role_leakage_detected": bool(role_leakage_detected),
    }


def _generate_prediction_from_inputs(
    model,
    *,
    full_vision_feats: torch.Tensor,
    feature_timestamps: Sequence[float],
    spatial_res: Sequence[int],
    prefix_ids: Sequence[int],
    segment_id: str,
    video_id: str,
    exercise_name: str,
    max_feedback_tokens: int = 64,
    max_total_new_tokens: int = 1024,
    max_feedbacks_per_segment: int | None = None,
    min_observation_sec: float = 0.0,
    min_feedback_tokens_before_end: int = 4,
    soft_feedback_token_budget: int | None = 24,
    feedback_end_logit_bias: float = 1.5,
    no_repeat_ngram_size: int = 3,
    feedback_repetition_penalty: float = 1.1,
    post_feedback_min_next_steps: int = 0,
    return_debug: bool = False,
) -> dict[str, Any]:
    feature_timestamps = [float(x) for x in feature_timestamps]
    spatial_res = [int(x) for x in spatial_res]
    full_vision_feats = full_vision_feats.to(model.device)
    if full_vision_feats.ndim != 3:
        raise ValueError(
            f"Expected full_vision_feats to have shape [T, HW, C], got {tuple(full_vision_feats.shape)}"
        )
    if not feature_timestamps:
        raise ValueError("feature_timestamps must not be empty")
    if int(full_vision_feats.shape[0]) != len(feature_timestamps):
        raise ValueError(
            "full_vision_feats and feature_timestamps length mismatch: "
            f"{int(full_vision_feats.shape[0])} vs {len(feature_timestamps)}"
        )

    generated_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(generated_ids)
    prefix_vision_mask = [0] * (len(prefix_ids) - 1) + [2]
    vision_xattn_mask = torch.tensor([prefix_vision_mask], dtype=torch.long, device=model.device)

    next_token_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]

    selected_step_indices = [0]
    next_step_idx = 1
    feedback_mode = False
    curr_response_len = 0
    generated_token_count = 0
    prompt_tokens = int(generated_ids.shape[1])
    timing_events: list[dict[str, Any]] = []
    current_timing_event: dict[str, Any] | None = None
    current_feedback_token_ids: list[int] = []
    feats_frequency = _estimate_feats_frequency(feature_timestamps)
    resolved_soft_feedback_token_budget = _resolve_soft_feedback_token_budget(model, soft_feedback_token_budget)
    first_feedback_visible_step: int | None = None
    first_feedback_visible_sec: float | None = None
    feedback_blocked_before_threshold = False
    feedback_stop_events: list[dict[str, Any]] = []
    current_feedback_stop_event: dict[str, Any] | None = None
    post_feedback_next_steps_remaining = 0
    feedback_followed_by_next = False
    waiting_for_post_feedback_next = False
    cooldown_blocked_immediate_reopen = False
    role_guard_token_ids = get_instruct_role_guard_token_ids(
        model.tokenizer,
        getattr(model, "base_llm_model_name_or_path", None),
    )

    segment_start_wall_time = time.perf_counter()
    model.eval()

    while generated_token_count < max_total_new_tokens:
        visible_vision_feats = full_vision_feats[selected_step_indices].unsqueeze(0)
        hard_cap_forced_end = False
        if feedback_mode:
            feedback_blocked_ids = sorted(set(_feedback_blocked_token_ids(model) + role_guard_token_ids))
            if curr_response_len < min_feedback_tokens_before_end:
                feedback_blocked_ids = sorted(set(feedback_blocked_ids + [feedback_end_id]))
            forbidden_ngram_tokens = _feedback_no_repeat_ngram_forbidden_tokens(
                current_feedback_token_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            token_biases = (
                {feedback_end_id: float(feedback_end_logit_bias)}
                if curr_response_len >= resolved_soft_feedback_token_budget
                else None
            )
            penalized_token_ids = list(dict.fromkeys(current_feedback_token_ids))
            next_token = _predict_next_token(
                model,
                generated_ids,
                attention_mask,
                vision_xattn_mask,
                visible_vision_feats,
                spatial_res,
                blocked_token_ids=feedback_blocked_ids,
                forbidden_token_ids=forbidden_ngram_tokens,
                repetition_penalty=feedback_repetition_penalty,
                penalized_token_ids=penalized_token_ids,
                token_biases=token_biases,
            )
            if current_feedback_stop_event is not None:
                current_feedback_stop_event["soft_end_bias_applied"] = bool(
                    current_feedback_stop_event["soft_end_bias_applied"] or token_biases
                )
                current_feedback_stop_event["no_repeat_ngram_block_applied"] = bool(
                    current_feedback_stop_event["no_repeat_ngram_block_applied"]
                    or forbidden_ngram_tokens
                )
                current_feedback_stop_event["repetition_penalty_applied"] = bool(
                    current_feedback_stop_event["repetition_penalty_applied"] or penalized_token_ids
                )
            hard_cap_forced_end = curr_response_len >= max_feedback_tokens
            if hard_cap_forced_end:
                next_token = feedback_end_id
        else:
            include_next = next_step_idx < len(feature_timestamps)
            observed_visible_sec = _observed_visible_sec(feature_timestamps, selected_step_indices)
            feedback_allowed = observed_visible_sec >= float(min_observation_sec)
            if post_feedback_next_steps_remaining > 0 and not include_next:
                break
            if post_feedback_next_steps_remaining > 0 and include_next:
                cooldown_blocked_immediate_reopen = True
            if not include_next and max_feedbacks_per_segment is not None and len(timing_events) >= max_feedbacks_per_segment:
                break
            if not feedback_allowed and include_next:
                feedback_blocked_before_threshold = True
            next_token = _predict_next_token(
                model,
                generated_ids,
                attention_mask,
                vision_xattn_mask,
                visible_vision_feats,
                spatial_res,
                allowed_token_ids=(
                    [next_token_id]
                    if ((not feedback_allowed and include_next) or (post_feedback_next_steps_remaining > 0 and include_next))
                    else _observation_allowed_token_ids(model, include_next=include_next)
                ),
            )
            if not include_next and next_token != feedback_begin_id:
                break

        generated_ids, attention_mask, vision_xattn_mask = _append_token(
            generated_ids,
            attention_mask,
            vision_xattn_mask,
            next_token,
            2 if next_token == next_token_id else 0,
        )
        generated_token_count += 1

        if next_token == next_token_id:
            selected_step_indices.append(next_step_idx)
            next_step_idx += 1
            if post_feedback_next_steps_remaining > 0:
                post_feedback_next_steps_remaining -= 1
            if waiting_for_post_feedback_next:
                feedback_followed_by_next = True
                waiting_for_post_feedback_next = False
            continue

        if next_token == feedback_begin_id:
            feedback_mode = True
            curr_response_len = 0
            current_feedback_token_ids = []
            waiting_for_post_feedback_next = False
            if first_feedback_visible_step is None:
                first_feedback_visible_step = int(selected_step_indices[-1]) if selected_step_indices else None
                first_feedback_visible_sec = _observed_visible_sec(feature_timestamps, selected_step_indices)
            current_feedback_stop_event = {
                "feedback_index": len(feedback_stop_events),
                "generated_feedback_token_count": 0,
                "ended_before_hard_cap": False,
                "terminated_naturally": False,
                "soft_end_bias_applied": False,
                "no_repeat_ngram_block_applied": False,
                "repetition_penalty_applied": False,
            }
            current_timing_event = {
                "ttft_sec": None,
                "time_to_last_token_sec": None,
            }
            feedback_start_wall_time = time.perf_counter()
            continue

        if next_token == feedback_end_id:
            feedback_mode = False
            feedback_text = model.tokenizer.decode(
                current_feedback_token_ids,
                skip_special_tokens=True,
            ).strip()
            if feedback_text and current_timing_event is not None:
                current_timing_event["text"] = feedback_text
                current_timing_event["generated_token_count"] = len(current_feedback_token_ids)
                current_timing_event["ttft_sec"] = float(current_timing_event["ttft_sec"] or 0.0)
                current_timing_event["time_to_last_token_sec"] = float(
                    current_timing_event["time_to_last_token_sec"] or 0.0
                )
                timing_events.append(current_timing_event)
            if current_feedback_stop_event is not None:
                current_feedback_stop_event["generated_feedback_token_count"] = len(current_feedback_token_ids)
                current_feedback_stop_event["ended_before_hard_cap"] = len(current_feedback_token_ids) < max_feedback_tokens
                current_feedback_stop_event["terminated_naturally"] = not hard_cap_forced_end
                feedback_stop_events.append(current_feedback_stop_event)
            skip_forward = math.floor((curr_response_len / INFERENCE_SPEED) * feats_frequency)
            next_step_idx = min(len(feature_timestamps), next_step_idx + skip_forward)
            curr_response_len = 0
            current_feedback_token_ids = []
            current_timing_event = None
            current_feedback_stop_event = None
            post_feedback_next_steps_remaining = max(0, int(post_feedback_min_next_steps))
            waiting_for_post_feedback_next = next_step_idx < len(feature_timestamps)
            if max_feedbacks_per_segment is not None and len(timing_events) >= max_feedbacks_per_segment:
                break
            continue

        current_feedback_token_ids.append(next_token)
        curr_response_len += 1
        if current_feedback_stop_event is not None:
            current_feedback_stop_event["generated_feedback_token_count"] = len(current_feedback_token_ids)
        if current_timing_event is not None:
            now = time.perf_counter()
            if current_timing_event["ttft_sec"] is None:
                current_timing_event["ttft_sec"] = now - feedback_start_wall_time
            current_timing_event["time_to_last_token_sec"] = now - feedback_start_wall_time

    if current_feedback_stop_event is not None:
        current_feedback_stop_event["generated_feedback_token_count"] = len(current_feedback_token_ids)
        current_feedback_stop_event["ended_before_hard_cap"] = False
        current_feedback_stop_event["terminated_naturally"] = False
        feedback_stop_events.append(current_feedback_stop_event)

    generation_wall_time_sec = time.perf_counter() - segment_start_wall_time
    pred_feedbacks, pred_feedback_timestamps = _extract_pred_feedbacks(
        generated_ids[0].tolist(),
        model=model,
        feats_frequency=feats_frequency,
    )

    for feedback_index, timing_event in enumerate(timing_events[: len(pred_feedbacks)]):
        timing_event["feedback_index"] = feedback_index
        timing_event["pred_timestamp_sec"] = float(pred_feedback_timestamps[feedback_index])
    role_leakage_detected = contains_role_leakage(
        model.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    )

    result = {
        "segment_id": segment_id,
        "video_id": video_id,
        "exercise_name": exercise_name,
        "pred_feedbacks": pred_feedbacks,
        "pred_feedback_timestamps": pred_feedback_timestamps,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_token_count,
        "total_tokens": prompt_tokens + generated_token_count,
        "generation_wall_time_sec": generation_wall_time_sec,
        "timing_events": timing_events[: len(pred_feedbacks)],
    }
    if return_debug:
        result.update(
            _build_debug_payload(
                generated_ids[0].tolist(),
                model,
                first_feedback_visible_step=first_feedback_visible_step,
                first_feedback_visible_sec=first_feedback_visible_sec,
                feedback_blocked_before_threshold=feedback_blocked_before_threshold,
                min_observation_sec=min_observation_sec,
                soft_feedback_token_budget=resolved_soft_feedback_token_budget,
                feedback_stop_events=feedback_stop_events,
                feedback_followed_by_next=feedback_followed_by_next,
                cooldown_blocked_immediate_reopen=cooldown_blocked_immediate_reopen,
                post_feedback_min_next_steps=post_feedback_min_next_steps,
                role_leakage_detected=role_leakage_detected,
            )
        )
    return result


@torch.no_grad()
def generate_prediction_from_features(
    model,
    full_vision_feats: torch.Tensor,
    feature_timestamps: Sequence[float],
    spatial_res: Sequence[int],
    system_prompt: str | None = None,
    segment_id: str = "live:webcam:000",
    video_id: str = "webcam",
    exercise_name: str = "live_webcam",
    max_feedback_tokens: int = 64,
    max_total_new_tokens: int = 1024,
    max_feedbacks_per_segment: int | None = None,
    min_observation_sec: float = 0.0,
    min_feedback_tokens_before_end: int = 4,
    soft_feedback_token_budget: int | None = 24,
    feedback_end_logit_bias: float = 1.5,
    no_repeat_ngram_size: int = 3,
    feedback_repetition_penalty: float = 1.1,
    post_feedback_min_next_steps: int = 0,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate stage-3 feedback from in-memory feature tensors."""
    prefix_ids = _build_generation_prefix(
        resolve_stage3_system_prompt(system_prompt or STAGE3_SYSTEM_PROMPT),
        model,
    )
    return _generate_prediction_from_inputs(
        model,
        full_vision_feats=full_vision_feats,
        feature_timestamps=feature_timestamps,
        spatial_res=spatial_res,
        prefix_ids=prefix_ids,
        segment_id=segment_id,
        video_id=video_id,
        exercise_name=exercise_name,
        max_feedback_tokens=max_feedback_tokens,
        max_total_new_tokens=max_total_new_tokens,
        max_feedbacks_per_segment=max_feedbacks_per_segment,
        min_observation_sec=min_observation_sec,
        min_feedback_tokens_before_end=min_feedback_tokens_before_end,
        soft_feedback_token_budget=soft_feedback_token_budget,
        feedback_end_logit_bias=feedback_end_logit_bias,
        no_repeat_ngram_size=no_repeat_ngram_size,
        feedback_repetition_penalty=feedback_repetition_penalty,
        post_feedback_min_next_steps=post_feedback_min_next_steps,
        return_debug=return_debug,
    )


@torch.no_grad()
def generate_segment_prediction(
    model,
    sample: Stage3SegmentRecord,
    max_feedback_tokens: int = 64,
    max_total_new_tokens: int = 1024,
    max_feedbacks_per_segment: int | None = None,
    min_observation_sec: float = 0.0,
    min_feedback_tokens_before_end: int = 4,
    soft_feedback_token_budget: int | None = 24,
    feedback_end_logit_bias: float = 1.5,
    no_repeat_ngram_size: int = 3,
    feedback_repetition_penalty: float = 1.1,
    post_feedback_min_next_steps: int = 0,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate interactive feedback for one cached benchmark segment."""
    full_vision_feats, feature_timestamps, spatial_res, prefix_ids = _load_generation_inputs(model, sample)
    return _generate_prediction_from_inputs(
        model,
        full_vision_feats=full_vision_feats,
        feature_timestamps=feature_timestamps,
        spatial_res=spatial_res,
        prefix_ids=prefix_ids,
        segment_id=sample.segment_id,
        video_id=sample.video_id,
        exercise_name=sample.exercise_name,
        max_feedback_tokens=max_feedback_tokens,
        max_total_new_tokens=max_total_new_tokens,
        max_feedbacks_per_segment=max_feedbacks_per_segment,
        min_observation_sec=min_observation_sec,
        min_feedback_tokens_before_end=min_feedback_tokens_before_end,
        soft_feedback_token_budget=soft_feedback_token_budget,
        feedback_end_logit_bias=feedback_end_logit_bias,
        no_repeat_ngram_size=no_repeat_ngram_size,
        feedback_repetition_penalty=feedback_repetition_penalty,
        post_feedback_min_next_steps=post_feedback_min_next_steps,
        return_debug=return_debug,
    )


@torch.no_grad()
def trace_generation_step_choices(
    model,
    sample: Stage3SegmentRecord,
    max_steps: int = 20,
    max_feedback_tokens: int = 64,
    min_observation_sec: float = 0.0,
    min_feedback_tokens_before_end: int = 4,
    soft_feedback_token_budget: int | None = 24,
    feedback_end_logit_bias: float = 1.5,
    no_repeat_ngram_size: int = 3,
    feedback_repetition_penalty: float = 1.1,
    post_feedback_min_next_steps: int = 0,
) -> dict[str, Any]:
    """Trace raw per-step token choices for one sample before control-token coercion."""
    full_vision_feats, feature_timestamps, spatial_res, prefix_ids = _load_generation_inputs(model, sample)

    generated_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(generated_ids)
    prefix_vision_mask = [0] * (len(prefix_ids) - 1) + [2]
    vision_xattn_mask = torch.tensor([prefix_vision_mask], dtype=torch.long, device=model.device)

    next_token_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]

    selected_step_indices = [0]
    next_step_idx = 1
    feedback_mode = False
    curr_response_len = 0
    current_feedback_token_ids: list[int] = []
    post_feedback_next_steps_remaining = 0
    trace: list[dict[str, Any]] = []
    resolved_soft_feedback_token_budget = _resolve_soft_feedback_token_budget(model, soft_feedback_token_budget)
    role_guard_token_ids = get_instruct_role_guard_token_ids(
        model.tokenizer,
        getattr(model, "base_llm_model_name_or_path", None),
    )

    for step_idx in range(max_steps):
        visible_vision_feats = full_vision_feats[selected_step_indices].unsqueeze(0)
        sanitization_reason = None
        observed_visible_sec = _observed_visible_sec(feature_timestamps, selected_step_indices)
        forbidden_ngram_tokens: list[int] = []
        penalized_token_ids: list[int] = []
        if feedback_mode:
            feedback_blocked_ids = sorted(set(_feedback_blocked_token_ids(model) + role_guard_token_ids))
            if curr_response_len < min_feedback_tokens_before_end:
                feedback_blocked_ids = sorted(set(feedback_blocked_ids + [feedback_end_id]))
            forbidden_ngram_tokens = _feedback_no_repeat_ngram_forbidden_tokens(
                current_feedback_token_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            token_biases = (
                {feedback_end_id: float(feedback_end_logit_bias)}
                if curr_response_len >= resolved_soft_feedback_token_budget
                else None
            )
            penalized_token_ids = list(dict.fromkeys(current_feedback_token_ids))
            raw_token, top5 = _predict_next_token_with_topk(
                model,
                generated_ids,
                attention_mask,
                vision_xattn_mask,
                visible_vision_feats,
                spatial_res,
                topk=5,
                blocked_token_ids=feedback_blocked_ids,
                forbidden_token_ids=forbidden_ngram_tokens,
                repetition_penalty=feedback_repetition_penalty,
                penalized_token_ids=penalized_token_ids,
                token_biases=token_biases,
            )
            sanitized_token = raw_token
            if curr_response_len >= max_feedback_tokens:
                sanitized_token = feedback_end_id
                sanitization_reason = "feedback_mode_forced_end"
        else:
            include_next = next_step_idx < len(feature_timestamps)
            feedback_allowed = observed_visible_sec >= float(min_observation_sec)
            if post_feedback_next_steps_remaining > 0 and not include_next:
                trace.append(
                    {
                        "step": step_idx,
                        "visible_steps": len(selected_step_indices),
                        "observed_visible_sec": observed_visible_sec,
                        "gate_active": bool(not feedback_allowed),
                        "feedback_mode": feedback_mode,
                        "raw_top1": None,
                        "sanitized": None,
                        "stop_reason": "end_of_stream_during_post_feedback_cooldown",
                        "top5": [],
                    }
                )
                break
            raw_token, top5 = _predict_next_token_with_topk(
                model,
                generated_ids,
                attention_mask,
                vision_xattn_mask,
                visible_vision_feats,
                spatial_res,
                topk=5,
                allowed_token_ids=(
                    [next_token_id]
                    if ((not feedback_allowed and include_next) or (post_feedback_next_steps_remaining > 0 and include_next))
                    else _observation_allowed_token_ids(model, include_next=include_next)
                ),
            )
            sanitized_token = raw_token
            if not include_next and raw_token != feedback_begin_id:
                trace.append(
                    {
                        "step": step_idx,
                        "visible_steps": len(selected_step_indices),
                        "observed_visible_sec": observed_visible_sec,
                        "gate_active": bool(not feedback_allowed),
                        "feedback_mode": feedback_mode,
                        "raw_top1": {
                            "token_id": int(raw_token),
                            "token": model.tokenizer.decode([raw_token], skip_special_tokens=False),
                        },
                        "sanitized": None,
                        "stop_reason": "end_of_stream_without_feedback",
                        "top5": top5,
                    }
                )
                break

        trace.append(
            {
                "step": step_idx,
                "visible_steps": len(selected_step_indices),
                "observed_visible_sec": observed_visible_sec,
                "gate_active": bool((not feedback_mode) and (observed_visible_sec < float(min_observation_sec))),
                "feedback_mode": feedback_mode,
                "soft_end_bias_active": bool(feedback_mode and curr_response_len >= resolved_soft_feedback_token_budget),
                "no_repeat_ngram_block_active": bool(feedback_mode and forbidden_ngram_tokens),
                "repetition_penalty_active": bool(feedback_mode and penalized_token_ids),
                "post_feedback_cooldown_active": bool((not feedback_mode) and post_feedback_next_steps_remaining > 0),
                "raw_top1": {
                    "token_id": int(raw_token),
                    "token": model.tokenizer.decode([raw_token], skip_special_tokens=False),
                },
                "sanitized": {
                    "token_id": int(sanitized_token),
                    "token": model.tokenizer.decode([sanitized_token], skip_special_tokens=False),
                },
                "sanitization_reason": sanitization_reason,
                "top5": top5,
            }
        )

        generated_ids, attention_mask, vision_xattn_mask = _append_token(
            generated_ids,
            attention_mask,
            vision_xattn_mask,
            sanitized_token,
            2 if sanitized_token == next_token_id else 0,
        )

        if sanitized_token == next_token_id:
            if next_step_idx >= len(feature_timestamps):
                break
            selected_step_indices.append(next_step_idx)
            next_step_idx += 1
            if post_feedback_next_steps_remaining > 0:
                post_feedback_next_steps_remaining -= 1
            continue

        if sanitized_token == feedback_begin_id:
            feedback_mode = True
            curr_response_len = 0
            current_feedback_token_ids = []
            continue

        if sanitized_token == feedback_end_id:
            feedback_mode = False
            curr_response_len = 0
            current_feedback_token_ids = []
            post_feedback_next_steps_remaining = max(0, int(post_feedback_min_next_steps))
            continue

        current_feedback_token_ids.append(sanitized_token)
        curr_response_len += 1

    return {
        "segment_id": sample.segment_id,
        "video_id": sample.video_id,
        "trace": trace,
        "prompt_text": model.tokenizer.decode(prefix_ids, skip_special_tokens=False),
        "prefix_ids": prefix_ids,
    }

def generate_benchmark_predictions(
    model,
    records: Sequence[Stage3SegmentRecord],
    output_path: str | Path | None = None,
    max_records: int | None = None,
    max_feedback_tokens: int = 64,
    max_total_new_tokens: int = 1024,
    max_feedbacks_per_segment: int | None = None,
    min_observation_sec: float = 0.0,
    min_feedback_tokens_before_end: int = 4,
    soft_feedback_token_budget: int | None = 24,
    feedback_end_logit_bias: float = 1.5,
    no_repeat_ngram_size: int = 3,
    feedback_repetition_penalty: float = 1.1,
    post_feedback_min_next_steps: int = 0,
) -> list[dict[str, Any]]:
    """Generate predictions for a benchmark segment set and optionally save them to JSON."""
    selected_records = limit_records(records, max_records=max_records)
    predictions: list[dict[str, Any]] = []
    progress = tqdm(selected_records, desc="generating benchmark predictions")
    for record in progress:
        progress.set_postfix(segment_id=record.segment_id)
        predictions.append(
            generate_segment_prediction(
            model=model,
            sample=record,
            max_feedback_tokens=max_feedback_tokens,
            max_total_new_tokens=max_total_new_tokens,
            max_feedbacks_per_segment=max_feedbacks_per_segment,
            min_observation_sec=min_observation_sec,
            min_feedback_tokens_before_end=min_feedback_tokens_before_end,
            soft_feedback_token_budget=soft_feedback_token_budget,
            feedback_end_logit_bias=feedback_end_logit_bias,
            no_repeat_ngram_size=no_repeat_ngram_size,
            feedback_repetition_penalty=feedback_repetition_penalty,
            post_feedback_min_next_steps=post_feedback_min_next_steps,
        )
        )
    if output_path is not None:
        save_predictions(predictions, output_path)
    return predictions
