"""Stage-3 LoRA fine-tuning utilities for long-range segments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.chat_format import build_chat_aligned_prefix
from src.stage2.runtime import resolve_runtime
from src.stage3.cache import load_cached_segment
from src.stage3.manifest import Stage3SegmentRecord, build_stage3_text_prefix


@dataclass(frozen=True)
class Stage3TrainingConfig:
    """Default stage-3 LoRA optimization settings."""

    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip_norm: float = 1.0
    epochs: int = 2
    effective_batch_size: int = 32
    micro_batch_size: int = 1
    log_every: int = 10
    eval_every_steps: int = 0
    feedback_action_weight: float | None = None
    auto_feedback_weight_cap: float = 8.0
    min_feedback_open_sec: float = 2.0
    feedback_open_margin: float = 0.75
    pre_feedback_next_margin: float = 0.25
    post_feedback_next_margin: float = 0.25
    feedback_open_margin_weight: float = 2.0
    pre_feedback_next_margin_weight: float = 1.0
    post_feedback_next_margin_weight: float = 1.0
    feedback_end_weight: float = 4.0

    @property
    def gradient_accumulation_steps(self) -> int:
        return max(1, self.effective_batch_size // self.micro_batch_size)


def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    raise RuntimeError(f"Non-finite tensor detected: {name}")


def _raise_if_nonfinite_trainable_gradients(model) -> None:
    for name, parameter in model.named_trainable_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient detected in trainable parameter: {name}")


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_length = max(len(sequence) for sequence in sequences)
    padded = torch.full((len(sequences), max_length), pad_value, dtype=dtype)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = torch.tensor(sequence, dtype=dtype)
    return padded


def _pad_vision_sequences(vision_sequences: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(sequence.shape[0] for sequence in vision_sequences)
    feature_dim_1 = vision_sequences[0].shape[1]
    feature_dim_2 = vision_sequences[0].shape[2]
    padded = torch.zeros(
        len(vision_sequences),
        max_length,
        feature_dim_1,
        feature_dim_2,
        dtype=vision_sequences[0].dtype,
    )
    for index, sequence in enumerate(vision_sequences):
        padded[index, : sequence.shape[0]] = sequence
    return padded


def _as_stage3_samples(batch: dict[str, list[Any]]) -> list[Stage3SegmentRecord]:
    return [Stage3SegmentRecord.from_dict({key: batch[key][index] for key in batch}) for index in range(len(batch["segment_id"]))]


def _first_feature_index_at_or_after_time(
    feature_timestamps: list[float],
    min_feedback_open_sec: float,
) -> int:
    if not feature_timestamps:
        return 0
    for index, timestamp in enumerate(feature_timestamps):
        if float(timestamp) >= float(min_feedback_open_sec):
            return index
    return len(feature_timestamps) - 1


def _plan_patience_aware_feedback_targets(
    feedbacks: tuple[str, ...],
    feedback_timestamps: tuple[float, ...],
    feature_timestamps: list[float],
    min_feedback_open_sec: float,
) -> list[tuple[int, str]]:
    if not feature_timestamps:
        return []

    feature_timestamp_tensor = torch.tensor(feature_timestamps, dtype=torch.float32)
    minimum_open_index = _first_feature_index_at_or_after_time(
        feature_timestamps=feature_timestamps,
        min_feedback_open_sec=min_feedback_open_sec,
    )
    planned_targets: list[tuple[int, str]] = []
    previous_assigned_index: int | None = None
    for feedback, feedback_timestamp in zip(feedbacks, feedback_timestamps):
        feedback_timestamp_t = torch.tensor(float(feedback_timestamp), dtype=torch.float32)
        nearest_index = int(torch.argmin(torch.abs(feature_timestamp_tensor - feedback_timestamp_t)).item())
        assigned_index = max(nearest_index, minimum_open_index)
        if previous_assigned_index is not None:
            assigned_index = max(assigned_index, previous_assigned_index + 1)
        assigned_index = min(assigned_index, len(feature_timestamps) - 1)
        planned_targets.append((assigned_index, feedback))
        previous_assigned_index = assigned_index
    return planned_targets


def _build_tokenized_stage3_sample(
    sample: Stage3SegmentRecord,
    model,
    tokenizer,
    timeline_token_ids: dict[str, int],
    feature_timestamps: list[float],
    min_feedback_open_sec: float,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int]]:
    model_name_or_path = getattr(model, "base_llm_model_name_or_path", None)
    if model_name_or_path and "instruct" in model_name_or_path.lower():
        prefix_ids = build_chat_aligned_prefix(
            tokenizer=tokenizer,
            system_prompt=sample.system_prompt,
            user_content="",
            model_name_or_path=model_name_or_path,
            add_generation_prompt=True,
        )
    else:
        bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        system_ids = _encode_text(tokenizer, build_stage3_text_prefix(sample.system_prompt))
        prefix_ids = bos_ids + system_ids

    planned_feedback_targets = _plan_patience_aware_feedback_targets(
        feedbacks=sample.feedbacks,
        feedback_timestamps=sample.feedback_timestamps,
        feature_timestamps=feature_timestamps,
        min_feedback_open_sec=min_feedback_open_sec,
    )
    aligned_feedbacks: dict[int, list[str]] = {}
    post_feedback_next_indices: set[int] = set()
    for feature_index, feedback in planned_feedback_targets:
        aligned_feedbacks.setdefault(feature_index, []).append(feedback)
        if feature_index + 1 < len(feature_timestamps):
            post_feedback_next_indices.add(feature_index + 1)

    timeline_ids: list[int] = []
    timeline_labels: list[int] = []
    timeline_mask: list[int] = []
    timeline_action_target_mask: list[int] = []
    timeline_text_target_mask: list[int] = []
    timeline_feedback_open_mask: list[int] = []
    timeline_pre_feedback_next_mask: list[int] = []
    timeline_feedback_end_mask: list[int] = []
    timeline_post_feedback_next_mask: list[int] = []
    for feature_index in range(len(feature_timestamps)):
        timeline_ids.append(timeline_token_ids["next"])
        timeline_labels.append(timeline_token_ids["next"])
        timeline_mask.append(2)
        timeline_action_target_mask.append(1)
        timeline_text_target_mask.append(0)
        timeline_feedback_open_mask.append(0)
        timeline_pre_feedback_next_mask.append(0)
        timeline_feedback_end_mask.append(0)
        timeline_post_feedback_next_mask.append(1 if feature_index in post_feedback_next_indices else 0)
        next_token_timeline_index = len(timeline_ids) - 1
        for feedback in aligned_feedbacks.get(feature_index, []):
            feedback_ids = _encode_text(tokenizer, feedback.strip())
            timeline_pre_feedback_next_mask[next_token_timeline_index] = 1
            timeline_ids.append(timeline_token_ids["feedback_begin"])
            timeline_labels.append(timeline_token_ids["feedback_begin"])
            timeline_mask.append(0)
            timeline_action_target_mask.append(1)
            timeline_text_target_mask.append(0)
            timeline_feedback_open_mask.append(1)
            timeline_pre_feedback_next_mask.append(0)
            timeline_feedback_end_mask.append(0)
            timeline_post_feedback_next_mask.append(0)
            timeline_ids.extend(feedback_ids)
            timeline_labels.extend(feedback_ids)
            timeline_mask.extend([0] * len(feedback_ids))
            timeline_action_target_mask.extend([0] * len(feedback_ids))
            timeline_text_target_mask.extend([1] * len(feedback_ids))
            timeline_feedback_open_mask.extend([0] * len(feedback_ids))
            timeline_pre_feedback_next_mask.extend([0] * len(feedback_ids))
            timeline_feedback_end_mask.extend([0] * len(feedback_ids))
            timeline_post_feedback_next_mask.extend([0] * len(feedback_ids))
            timeline_ids.append(timeline_token_ids["feedback_end"])
            timeline_labels.append(timeline_token_ids["feedback_end"])
            timeline_mask.append(0)
            timeline_action_target_mask.append(0)
            timeline_text_target_mask.append(1)
            timeline_feedback_open_mask.append(0)
            timeline_pre_feedback_next_mask.append(0)
            timeline_feedback_end_mask.append(1)
            timeline_post_feedback_next_mask.append(0)

    input_ids = prefix_ids + timeline_ids
    labels = ([-100] * len(prefix_ids)) + timeline_labels
    vision_xattn_mask = ([0] * len(prefix_ids)) + timeline_mask
    action_target_mask = ([0] * len(prefix_ids)) + timeline_action_target_mask
    text_target_mask = ([0] * len(prefix_ids)) + timeline_text_target_mask
    feedback_open_mask = ([0] * len(prefix_ids)) + timeline_feedback_open_mask
    pre_feedback_next_mask = ([0] * len(prefix_ids)) + timeline_pre_feedback_next_mask
    feedback_end_mask = ([0] * len(prefix_ids)) + timeline_feedback_end_mask
    post_feedback_next_mask = ([0] * len(prefix_ids)) + timeline_post_feedback_next_mask
    return (
        input_ids,
        labels,
        vision_xattn_mask,
        action_target_mask,
        text_target_mask,
        feedback_open_mask,
        pre_feedback_next_mask,
        feedback_end_mask,
        post_feedback_next_mask,
    )


def prepare_stage3_batch(
    batch: dict[str, list[Any]],
    model,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load cached segment features and build the interleaved stage-3 token sequence."""
    samples = _as_stage3_samples(batch)
    tokenizer = model.tokenizer
    device = torch.device(device or model.device)

    input_id_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    vision_mask_sequences: list[list[int]] = []
    action_target_sequences: list[list[int]] = []
    text_target_sequences: list[list[int]] = []
    feedback_open_sequences: list[list[int]] = []
    pre_feedback_next_sequences: list[list[int]] = []
    feedback_end_sequences: list[list[int]] = []
    post_feedback_next_sequences: list[list[int]] = []
    vision_sequences: list[torch.Tensor] = []
    spatial_res = None

    for sample in samples:
        if sample.cached_features_path is None:
            raise ValueError(f"Sample {sample.segment_id} is missing cached_features_path")

        cached = load_cached_segment(sample.cached_features_path)
        vision_sequence = cached["feats"]
        feature_timestamps = [float(x) for x in cached.get("feature_timestamps", sample.feature_timestamps)]
        if not feature_timestamps:
            raise ValueError(f"Cached segment {sample.segment_id} has no feature timestamps")

        (
            input_ids,
            labels,
            vision_mask,
            action_target_mask,
            text_target_mask,
            feedback_open_mask,
            pre_feedback_next_mask,
            feedback_end_mask,
            post_feedback_next_mask,
        ) = _build_tokenized_stage3_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            timeline_token_ids=model.timeline_token_ids,
            feature_timestamps=feature_timestamps,
            min_feedback_open_sec=float(getattr(model, "stage3_min_feedback_open_sec", 2.0)),
        )
        input_id_sequences.append(input_ids)
        label_sequences.append(labels)
        vision_mask_sequences.append(vision_mask)
        action_target_sequences.append(action_target_mask)
        text_target_sequences.append(text_target_mask)
        feedback_open_sequences.append(feedback_open_mask)
        pre_feedback_next_sequences.append(pre_feedback_next_mask)
        feedback_end_sequences.append(feedback_end_mask)
        post_feedback_next_sequences.append(post_feedback_next_mask)
        vision_sequences.append(vision_sequence)
        spatial_res = cached.get("spatial_res", [5, 7])

    feature_lengths = {sequence.shape[0] for sequence in vision_sequences}
    if len(feature_lengths) > 1:
        raise ValueError(
            "Variable-length cached segments cannot be mixed in one micro-batch with the current "
            "cross-attention implementation. Keep micro_batch_size=1."
        )

    padded_input_ids = _pad_sequences(
        input_id_sequences,
        pad_value=tokenizer.pad_token_id,
        dtype=torch.long,
    ).to(device)
    padded_labels = _pad_sequences(label_sequences, pad_value=-100, dtype=torch.long).to(device)
    padded_vision_mask = _pad_sequences(vision_mask_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_action_target_mask = _pad_sequences(action_target_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_text_target_mask = _pad_sequences(text_target_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_feedback_open_mask = _pad_sequences(feedback_open_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_pre_feedback_next_mask = _pad_sequences(pre_feedback_next_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_feedback_end_mask = _pad_sequences(feedback_end_sequences, pad_value=0, dtype=torch.long).to(device)
    padded_post_feedback_next_mask = _pad_sequences(post_feedback_next_sequences, pad_value=0, dtype=torch.long).to(device)
    attention_mask = (padded_input_ids != tokenizer.pad_token_id).long().to(device)
    padded_vision_feats = _pad_vision_sequences(vision_sequences).to(device)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "vision_xattn_mask": padded_vision_mask,
        "action_target_mask": padded_action_target_mask,
        "text_target_mask": padded_text_target_mask,
        "feedback_open_mask": padded_feedback_open_mask,
        "pre_feedback_next_mask": padded_pre_feedback_next_mask,
        "feedback_end_mask": padded_feedback_end_mask,
        "post_feedback_next_mask": padded_post_feedback_next_mask,
        "vision_feats": {
            "feats": padded_vision_feats,
            "spatial_res": spatial_res or [5, 7],
        },
        "samples": samples,
    }


def build_stage3_optimizer(model, config: Stage3TrainingConfig) -> AdamW:
    """Create the LoRA-only stage-3 optimizer."""
    trainable_parameters = model.trainable_parameters()
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found; LoRA attachment likely failed")

    return AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )


def assert_only_lora_has_gradients(model) -> None:
    """Fail if a non-LoRA parameter unexpectedly receives gradients."""
    invalid_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and "lora_" not in name
    ]
    if invalid_names:
        invalid_str = ", ".join(invalid_names)
        raise RuntimeError(f"Frozen parameters received gradients during stage-3: {invalid_str}")


def compute_stage3_action_statistics(dataloader, model) -> dict[str, float]:
    """Count `<next>` and `<feedback>` action targets over a stage-3 train subset."""
    next_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]

    next_action_targets = 0
    feedback_action_targets = 0
    total_action_targets = 0
    total_text_targets = 0
    feedback_open_targets = 0
    pre_feedback_next_targets = 0
    feedback_end_targets = 0
    post_feedback_next_targets = 0
    feedback_token_lengths: list[int] = []

    for batch in dataloader:
        prepared = prepare_stage3_batch(batch, model=model)
        action_mask = prepared["action_target_mask"].bool()
        text_mask = prepared["text_target_mask"].bool()
        labels = prepared["labels"]

        total_action_targets += int(action_mask.sum().item())
        total_text_targets += int(text_mask.sum().item())
        next_action_targets += int(((labels == next_id) & action_mask).sum().item())
        feedback_action_targets += int(((labels == feedback_begin_id) & action_mask).sum().item())
        feedback_open_targets += int(prepared["feedback_open_mask"].sum().item())
        pre_feedback_next_targets += int(prepared["pre_feedback_next_mask"].sum().item())
        feedback_end_targets += int(((labels == feedback_end_id) & prepared["feedback_end_mask"].bool()).sum().item())
        post_feedback_next_targets += int(prepared["post_feedback_next_mask"].sum().item())
        for sample in prepared["samples"]:
            for feedback in sample.feedbacks:
                feedback_token_lengths.append(len(_encode_text(model.tokenizer, feedback.strip())))

    raw_ratio = float(next_action_targets / max(feedback_action_targets, 1))
    if feedback_token_lengths:
        feedback_length_tensor = torch.tensor(feedback_token_lengths, dtype=torch.float32)
        feedback_token_length_p50 = float(torch.quantile(feedback_length_tensor, 0.50).item())
        feedback_token_length_p75 = float(torch.quantile(feedback_length_tensor, 0.75).item())
        feedback_token_length_p90 = float(torch.quantile(feedback_length_tensor, 0.90).item())
    else:
        feedback_token_length_p50 = 0.0
        feedback_token_length_p75 = 0.0
        feedback_token_length_p90 = 0.0
    return {
        "next_action_targets": float(next_action_targets),
        "feedback_action_targets": float(feedback_action_targets),
        "total_action_targets": float(total_action_targets),
        "total_text_targets": float(total_text_targets),
        "feedback_open_targets": float(feedback_open_targets),
        "pre_feedback_next_targets": float(pre_feedback_next_targets),
        "feedback_end_targets": float(feedback_end_targets),
        "post_feedback_next_targets": float(post_feedback_next_targets),
        "raw_ratio": raw_ratio,
        "feedback_token_length_p50": feedback_token_length_p50,
        "feedback_token_length_p75": feedback_token_length_p75,
        "feedback_token_length_p90": feedback_token_length_p90,
    }


def compute_stage3_loss_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_target_mask: torch.Tensor,
    text_target_mask: torch.Tensor,
    feedback_open_mask: torch.Tensor,
    pre_feedback_next_mask: torch.Tensor,
    feedback_end_mask: torch.Tensor,
    post_feedback_next_mask: torch.Tensor,
    model,
    feedback_action_weight: float = 1.0,
    feedback_open_margin: float = 0.75,
    pre_feedback_next_margin: float = 0.25,
    post_feedback_next_margin: float = 0.25,
    feedback_open_margin_weight: float = 2.0,
    pre_feedback_next_margin_weight: float = 1.0,
    post_feedback_next_margin_weight: float = 1.0,
    feedback_end_weight: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Compute constrained action loss plus boundary-aware text loss for stage 3."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_action_mask = action_target_mask[:, 1:].contiguous().bool()
    shift_text_mask = text_target_mask[:, 1:].contiguous().bool()
    shift_feedback_open_mask = feedback_open_mask[:, 1:].contiguous().bool()
    shift_pre_feedback_next_mask = pre_feedback_next_mask[:, 1:].contiguous().bool()
    shift_feedback_end_mask = feedback_end_mask[:, 1:].contiguous().bool()
    shift_post_feedback_next_mask = post_feedback_next_mask[:, 1:].contiguous().bool()
    valid_label_mask = shift_labels != -100

    next_id = model.timeline_token_ids["next"]
    feedback_begin_id = model.timeline_token_ids["feedback_begin"]
    feedback_end_id = model.timeline_token_ids["feedback_end"]
    action_candidate_ids = torch.tensor(
        [next_id, feedback_begin_id],
        device=shift_logits.device,
        dtype=torch.long,
    )

    action_mask = shift_action_mask & valid_label_mask
    text_mask = shift_text_mask & valid_label_mask

    zero = logits.sum() * 0.0

    action_loss_sum = zero
    action_count = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    weighted_action_count = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    weighted_text_count = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    next_action_loss = zero
    feedback_action_loss = zero
    feedback_open_margin_loss = zero
    pre_feedback_next_margin_loss = zero
    post_feedback_next_margin_loss = zero
    feedback_prob_on_feedback_targets = zero
    feedback_prob_on_next_targets = zero
    feedback_prob_on_feedback_open_targets = zero
    feedback_prob_on_pre_feedback_next_targets = zero
    feedback_prob_on_post_feedback_next_targets = zero
    if torch.any(action_mask):
        action_logits = shift_logits[action_mask][:, action_candidate_ids]
        action_labels = shift_labels[action_mask]
        remapped_action_labels = (action_labels == feedback_begin_id).long()
        class_weights = action_logits.new_tensor([1.0, float(feedback_action_weight)])
        action_losses = torch.nn.functional.cross_entropy(
            action_logits,
            remapped_action_labels,
            reduction="none",
        )
        sample_weights = class_weights[remapped_action_labels]
        action_loss_sum = (action_losses * sample_weights).sum()
        action_count = action_losses.new_tensor(float(action_losses.numel()))
        weighted_action_count = sample_weights.sum()
        next_mask = remapped_action_labels == 0
        feedback_mask = remapped_action_labels == 1
        action_probs = torch.softmax(action_logits, dim=-1)
        feedback_minus_next = action_logits[:, 1] - action_logits[:, 0]
        if torch.any(next_mask):
            next_action_loss = action_losses[next_mask].mean()
            feedback_prob_on_next_targets = action_probs[next_mask, 1].mean()
        if torch.any(feedback_mask):
            feedback_action_loss = action_losses[feedback_mask].mean()
            feedback_prob_on_feedback_targets = action_probs[feedback_mask, 1].mean()

        action_indices = torch.where(action_mask)
        action_positions = torch.stack(action_indices, dim=1)
        feedback_open_positions = torch.stack(torch.where(shift_feedback_open_mask & valid_label_mask), dim=1)
        pre_feedback_next_positions = torch.stack(
            torch.where(shift_pre_feedback_next_mask & valid_label_mask), dim=1
        )
        post_feedback_next_positions = torch.stack(
            torch.where(shift_post_feedback_next_mask & valid_label_mask), dim=1
        )

        if feedback_open_positions.numel() > 0:
            open_matches = (action_positions[:, None, :] == feedback_open_positions[None, :, :]).all(dim=-1).any(dim=1)
            if torch.any(open_matches):
                feedback_open_margin_loss = torch.relu(
                    action_logits.new_tensor(float(feedback_open_margin)) - feedback_minus_next[open_matches]
                ).mean()
                feedback_prob_on_feedback_open_targets = action_probs[open_matches, 1].mean()

        if pre_feedback_next_positions.numel() > 0:
            pre_next_matches = (
                (action_positions[:, None, :] == pre_feedback_next_positions[None, :, :]).all(dim=-1).any(dim=1)
            )
            if torch.any(pre_next_matches):
                next_minus_feedback = action_logits[:, 0] - action_logits[:, 1]
                pre_feedback_next_margin_loss = torch.relu(
                    action_logits.new_tensor(float(pre_feedback_next_margin)) - next_minus_feedback[pre_next_matches]
                ).mean()
                feedback_prob_on_pre_feedback_next_targets = action_probs[pre_next_matches, 1].mean()

        if post_feedback_next_positions.numel() > 0:
            post_next_matches = (
                (action_positions[:, None, :] == post_feedback_next_positions[None, :, :]).all(dim=-1).any(dim=1)
            )
            if torch.any(post_next_matches):
                next_minus_feedback = action_logits[:, 0] - action_logits[:, 1]
                post_feedback_next_margin_loss = torch.relu(
                    action_logits.new_tensor(float(post_feedback_next_margin)) - next_minus_feedback[post_next_matches]
                ).mean()
                feedback_prob_on_post_feedback_next_targets = action_probs[post_next_matches, 1].mean()

    text_loss_sum = zero
    text_count = torch.zeros((), device=shift_logits.device, dtype=shift_logits.dtype)
    feedback_end_loss = zero
    if torch.any(text_mask):
        text_logits = shift_logits[text_mask]
        text_labels = shift_labels[text_mask]
        text_losses = torch.nn.functional.cross_entropy(
            text_logits,
            text_labels,
            reduction="none",
        )
        text_sample_weights = torch.ones_like(text_losses)
        feedback_end_text_mask = (shift_feedback_end_mask & valid_label_mask)[text_mask]
        if torch.any(feedback_end_text_mask):
            text_sample_weights = text_sample_weights.clone()
            text_sample_weights[feedback_end_text_mask] = float(feedback_end_weight)
            feedback_end_loss = text_losses[feedback_end_text_mask].mean()
        text_loss_sum = (text_losses * text_sample_weights).sum()
        text_count = text_losses.new_tensor(float(text_losses.numel()))
        weighted_text_count = text_sample_weights.sum()

    total_count = weighted_action_count + weighted_text_count
    ce_loss = torch.where(
        total_count > 0,
        (action_loss_sum + text_loss_sum) / total_count,
        zero,
    )
    action_loss = torch.where(weighted_action_count > 0, action_loss_sum / weighted_action_count, zero)
    text_loss = torch.where(weighted_text_count > 0, text_loss_sum / weighted_text_count, zero)
    total_loss = (
        ce_loss
        + (float(feedback_open_margin_weight) * feedback_open_margin_loss)
        + (float(pre_feedback_next_margin_weight) * pre_feedback_next_margin_loss)
        + (float(post_feedback_next_margin_weight) * post_feedback_next_margin_loss)
    )

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "action_loss": action_loss,
        "text_loss": text_loss,
        "action_count": action_count,
        "weighted_action_count": weighted_action_count,
        "weighted_text_count": weighted_text_count,
        "text_count": text_count,
        "next_action_loss": next_action_loss,
        "feedback_action_loss": feedback_action_loss,
        "feedback_open_margin_loss": feedback_open_margin_loss,
        "pre_feedback_next_margin_loss": pre_feedback_next_margin_loss,
        "post_feedback_next_margin_loss": post_feedback_next_margin_loss,
        "feedback_end_loss": feedback_end_loss,
        "feedback_prob_on_feedback_targets": feedback_prob_on_feedback_targets,
        "feedback_prob_on_next_targets": feedback_prob_on_next_targets,
        "feedback_prob_on_feedback_open_targets": feedback_prob_on_feedback_open_targets,
        "feedback_prob_on_pre_feedback_next_targets": feedback_prob_on_pre_feedback_next_targets,
        "feedback_prob_on_post_feedback_next_targets": feedback_prob_on_post_feedback_next_targets,
    }


def _evaluate_validation_loss(
    model,
    dataloader,
    feedback_action_weight: float,
    config: Stage3TrainingConfig,
) -> dict[str, float]:
    total_loss = 0.0
    total_ce_loss = 0.0
    total_action_loss = 0.0
    total_text_loss = 0.0
    total_feedback_open_margin_loss = 0.0
    total_pre_feedback_next_margin_loss = 0.0
    total_post_feedback_next_margin_loss = 0.0
    total_feedback_end_loss = 0.0
    total_feedback_prob_on_feedback_targets = 0.0
    total_feedback_prob_on_next_targets = 0.0
    total_feedback_prob_on_feedback_open_targets = 0.0
    total_feedback_prob_on_pre_feedback_next_targets = 0.0
    total_feedback_prob_on_post_feedback_next_targets = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            prepared = prepare_stage3_batch(batch, model=model)
            outputs = model(
                input_ids=prepared["input_ids"],
                attention_mask=prepared["attention_mask"],
                vision_feats=prepared["vision_feats"],
                vision_xattn_mask=prepared["vision_xattn_mask"],
                labels=None,
            )
            _raise_if_nonfinite("validation.outputs.logits", outputs.logits)
            losses = compute_stage3_loss_metrics(
                logits=outputs.logits,
                labels=prepared["labels"],
                action_target_mask=prepared["action_target_mask"],
                text_target_mask=prepared["text_target_mask"],
                feedback_open_mask=prepared["feedback_open_mask"],
                pre_feedback_next_mask=prepared["pre_feedback_next_mask"],
                feedback_end_mask=prepared["feedback_end_mask"],
                post_feedback_next_mask=prepared["post_feedback_next_mask"],
                model=model,
                feedback_action_weight=feedback_action_weight,
                feedback_open_margin=config.feedback_open_margin,
                pre_feedback_next_margin=config.pre_feedback_next_margin,
                post_feedback_next_margin=config.post_feedback_next_margin,
                feedback_open_margin_weight=config.feedback_open_margin_weight,
                pre_feedback_next_margin_weight=config.pre_feedback_next_margin_weight,
                post_feedback_next_margin_weight=config.post_feedback_next_margin_weight,
                feedback_end_weight=config.feedback_end_weight,
            )
            _raise_if_nonfinite("validation.losses.loss", losses["loss"])
            total_loss += float(losses["loss"].item())
            total_ce_loss += float(losses["ce_loss"].item())
            total_action_loss += float(losses["action_loss"].item())
            total_text_loss += float(losses["text_loss"].item())
            total_feedback_open_margin_loss += float(losses["feedback_open_margin_loss"].item())
            total_pre_feedback_next_margin_loss += float(losses["pre_feedback_next_margin_loss"].item())
            total_post_feedback_next_margin_loss += float(losses["post_feedback_next_margin_loss"].item())
            total_feedback_end_loss += float(losses["feedback_end_loss"].item())
            total_feedback_prob_on_feedback_targets += float(losses["feedback_prob_on_feedback_targets"].item())
            total_feedback_prob_on_next_targets += float(losses["feedback_prob_on_next_targets"].item())
            total_feedback_prob_on_feedback_open_targets += float(losses["feedback_prob_on_feedback_open_targets"].item())
            total_feedback_prob_on_pre_feedback_next_targets += float(losses["feedback_prob_on_pre_feedback_next_targets"].item())
            total_feedback_prob_on_post_feedback_next_targets += float(losses["feedback_prob_on_post_feedback_next_targets"].item())
            batch_count += 1
    denom = max(1, batch_count)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce_loss / denom,
        "action_loss": total_action_loss / denom,
        "text_loss": total_text_loss / denom,
        "feedback_open_margin_loss": total_feedback_open_margin_loss / denom,
        "pre_feedback_next_margin_loss": total_pre_feedback_next_margin_loss / denom,
        "post_feedback_next_margin_loss": total_post_feedback_next_margin_loss / denom,
        "feedback_end_loss": total_feedback_end_loss / denom,
        "feedback_prob_on_feedback_targets": total_feedback_prob_on_feedback_targets / denom,
        "feedback_prob_on_next_targets": total_feedback_prob_on_next_targets / denom,
        "feedback_prob_on_feedback_open_targets": total_feedback_prob_on_feedback_open_targets / denom,
        "feedback_prob_on_pre_feedback_next_targets": total_feedback_prob_on_pre_feedback_next_targets / denom,
        "feedback_prob_on_post_feedback_next_targets": total_feedback_prob_on_post_feedback_next_targets / denom,
    }


def _run_probe_generation(
    model,
    probe_sample: Stage3SegmentRecord | None,
    probe_generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if probe_sample is None:
        return None

    from src.stage3.generation import generate_segment_prediction

    kwargs = {
        "max_feedback_tokens": 64,
        "max_total_new_tokens": 256,
        "max_feedbacks_per_segment": None,
    }
    if probe_generation_kwargs:
        kwargs.update(probe_generation_kwargs)
    return generate_segment_prediction(model=model, sample=probe_sample, **kwargs)


def _build_probe_epoch_preview(probe_prediction: dict[str, Any] | None) -> dict[str, Any] | None:
    if probe_prediction is None:
        return None
    return {
        "segment_id": probe_prediction["segment_id"],
        "video_id": probe_prediction["video_id"],
        "feedback_count": len(probe_prediction["pred_feedbacks"]),
        "pred_feedbacks": probe_prediction["pred_feedbacks"],
        "pred_feedback_timestamps": probe_prediction["pred_feedback_timestamps"],
        "generated_tokens": probe_prediction["generated_tokens"],
        "action_token_counts": probe_prediction.get("action_token_counts"),
        "raw_stream_text": probe_prediction.get("raw_stream_text"),
        "first_feedback_visible_step": probe_prediction.get("first_feedback_visible_step"),
        "first_feedback_visible_sec": probe_prediction.get("first_feedback_visible_sec"),
        "feedback_blocked_before_threshold": probe_prediction.get("feedback_blocked_before_threshold"),
        "feedback_stop_events": probe_prediction.get("feedback_stop_events"),
        "feedback_followed_by_next": probe_prediction.get("feedback_followed_by_next"),
        "cooldown_blocked_immediate_reopen": probe_prediction.get("cooldown_blocked_immediate_reopen"),
        "role_leakage_detected": probe_prediction.get("role_leakage_detected"),
    }


def train_stage3(
    model,
    dataloader,
    config: Stage3TrainingConfig,
    output_dir: str | Path | None = None,
    validation_dataloader=None,
    probe_sample: Stage3SegmentRecord | None = None,
    probe_generation_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Train stage-3 LoRA weights on cached long-range segments."""
    runtime = resolve_runtime(preferred_device=model.device.type)
    model.stage3_min_feedback_open_sec = float(config.min_feedback_open_sec)
    model.train()
    optimizer = build_stage3_optimizer(model, config)
    grad_accum_steps = config.gradient_accumulation_steps
    action_stats = compute_stage3_action_statistics(dataloader, model)
    model.stage3_train_feedback_token_length_p50 = float(action_stats["feedback_token_length_p50"])
    model.stage3_train_feedback_token_length_p75 = float(action_stats["feedback_token_length_p75"])
    model.stage3_train_feedback_token_length_p90 = float(action_stats["feedback_token_length_p90"])
    model.stage3_feedback_soft_budget = int(
        min(32, max(12, round(float(action_stats["feedback_token_length_p75"] or 20.0))))
    )
    if config.feedback_action_weight is None:
        feedback_action_weight = min(
            float(config.auto_feedback_weight_cap),
            float(action_stats["raw_ratio"] ** 0.5),
        )
    else:
        feedback_action_weight = float(config.feedback_action_weight)
    history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    global_step = 0
    output_dir_path = Path(output_dir) if output_dir is not None else None

    for epoch_index in range(config.epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_action_loss = 0.0
        running_text_loss = 0.0
        running_next_action_loss = 0.0
        running_feedback_action_loss = 0.0
        running_feedback_open_margin_loss = 0.0
        running_pre_feedback_next_margin_loss = 0.0
        running_post_feedback_next_margin_loss = 0.0
        running_feedback_end_loss = 0.0
        running_feedback_prob_on_feedback_targets = 0.0
        running_feedback_prob_on_next_targets = 0.0
        running_feedback_prob_on_feedback_open_targets = 0.0
        running_feedback_prob_on_pre_feedback_next_targets = 0.0
        running_feedback_prob_on_post_feedback_next_targets = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(dataloader, desc=f"epoch {epoch_index + 1}/{config.epochs}")

        for step_index, batch in enumerate(progress_bar, start=1):
            prepared = prepare_stage3_batch(batch, model=model)

            with torch.autocast(
                device_type=model.device.type,
                dtype=runtime.llm_dtype,
                enabled=runtime.use_autocast,
            ):
                outputs = model(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    vision_feats=prepared["vision_feats"],
                    vision_xattn_mask=prepared["vision_xattn_mask"],
                    labels=None,
                )
                _raise_if_nonfinite("outputs.logits", outputs.logits)
                losses = compute_stage3_loss_metrics(
                    logits=outputs.logits,
                    labels=prepared["labels"],
                    action_target_mask=prepared["action_target_mask"],
                    text_target_mask=prepared["text_target_mask"],
                    feedback_open_mask=prepared["feedback_open_mask"],
                    pre_feedback_next_mask=prepared["pre_feedback_next_mask"],
                    feedback_end_mask=prepared["feedback_end_mask"],
                    post_feedback_next_mask=prepared["post_feedback_next_mask"],
                    model=model,
                    feedback_action_weight=feedback_action_weight,
                    feedback_open_margin=config.feedback_open_margin,
                    pre_feedback_next_margin=config.pre_feedback_next_margin,
                    post_feedback_next_margin=config.post_feedback_next_margin,
                    feedback_open_margin_weight=config.feedback_open_margin_weight,
                    pre_feedback_next_margin_weight=config.pre_feedback_next_margin_weight,
                    post_feedback_next_margin_weight=config.post_feedback_next_margin_weight,
                    feedback_end_weight=config.feedback_end_weight,
                )
                _raise_if_nonfinite("losses.loss", losses["loss"])
                _raise_if_nonfinite("losses.ce_loss", losses["ce_loss"])
                _raise_if_nonfinite("losses.action_loss", losses["action_loss"])
                _raise_if_nonfinite("losses.text_loss", losses["text_loss"])
                loss = losses["loss"] / grad_accum_steps

            loss.backward()
            _raise_if_nonfinite_trainable_gradients(model)
            running_loss += float(losses["loss"].item())
            running_ce_loss += float(losses["ce_loss"].item())
            running_action_loss += float(losses["action_loss"].item())
            running_text_loss += float(losses["text_loss"].item())
            running_next_action_loss += float(losses["next_action_loss"].item())
            running_feedback_action_loss += float(losses["feedback_action_loss"].item())
            running_feedback_open_margin_loss += float(losses["feedback_open_margin_loss"].item())
            running_pre_feedback_next_margin_loss += float(losses["pre_feedback_next_margin_loss"].item())
            running_post_feedback_next_margin_loss += float(losses["post_feedback_next_margin_loss"].item())
            running_feedback_end_loss += float(losses["feedback_end_loss"].item())
            running_feedback_prob_on_feedback_targets += float(losses["feedback_prob_on_feedback_targets"].item())
            running_feedback_prob_on_next_targets += float(losses["feedback_prob_on_next_targets"].item())
            running_feedback_prob_on_feedback_open_targets += float(losses["feedback_prob_on_feedback_open_targets"].item())
            running_feedback_prob_on_pre_feedback_next_targets += float(losses["feedback_prob_on_pre_feedback_next_targets"].item())
            running_feedback_prob_on_post_feedback_next_targets += float(losses["feedback_prob_on_post_feedback_next_targets"].item())

            should_step = (step_index % grad_accum_steps == 0) or (step_index == len(dataloader))
            if should_step:
                clip_grad_norm_(model.trainable_parameters(), config.grad_clip_norm)
                assert_only_lora_has_gradients(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step_index % config.log_every == 0 or should_step:
                progress_bar.set_postfix(
                    loss=f"{running_loss / step_index:.4f}",
                    ce_loss=f"{running_ce_loss / step_index:.4f}",
                    action_loss=f"{running_action_loss / step_index:.4f}",
                    text_loss=f"{running_text_loss / step_index:.4f}",
                    feedback_end_loss=f"{running_feedback_end_loss / step_index:.4f}",
                )

            should_eval = (
                validation_dataloader is not None
                and config.eval_every_steps > 0
                and (step_index % config.eval_every_steps == 0)
            )
            if should_eval:
                model.eval()
                validation_metrics = _evaluate_validation_loss(
                    model,
                    validation_dataloader,
                    feedback_action_weight=feedback_action_weight,
                    config=config,
                )
                probe_prediction = _run_probe_generation(
                    model,
                    probe_sample=probe_sample,
                    probe_generation_kwargs=probe_generation_kwargs,
                )
                eval_record = {
                    "step": float((epoch_index * len(dataloader)) + step_index),
                    "epoch": float(epoch_index + (step_index / max(1, len(dataloader)))),
                    "running_loss": running_loss / step_index,
                    "running_ce_loss": running_ce_loss / step_index,
                    "running_action_loss": running_action_loss / step_index,
                    "running_text_loss": running_text_loss / step_index,
                    "running_next_action_loss": running_next_action_loss / step_index,
                    "running_feedback_action_loss": running_feedback_action_loss / step_index,
                    "running_feedback_open_margin_loss": running_feedback_open_margin_loss / step_index,
                    "running_pre_feedback_next_margin_loss": running_pre_feedback_next_margin_loss / step_index,
                    "running_post_feedback_next_margin_loss": running_post_feedback_next_margin_loss / step_index,
                    "running_feedback_end_loss": running_feedback_end_loss / step_index,
                    "running_feedback_prob_on_feedback_targets": running_feedback_prob_on_feedback_targets / step_index,
                    "running_feedback_prob_on_next_targets": running_feedback_prob_on_next_targets / step_index,
                    "running_feedback_prob_on_feedback_open_targets": running_feedback_prob_on_feedback_open_targets / step_index,
                    "running_feedback_prob_on_pre_feedback_next_targets": running_feedback_prob_on_pre_feedback_next_targets / step_index,
                    "running_feedback_prob_on_post_feedback_next_targets": running_feedback_prob_on_post_feedback_next_targets / step_index,
                    "validation_loss": validation_metrics["loss"],
                    "validation_ce_loss": validation_metrics["ce_loss"],
                    "validation_action_loss": validation_metrics["action_loss"],
                    "validation_text_loss": validation_metrics["text_loss"],
                    "validation_feedback_open_margin_loss": validation_metrics["feedback_open_margin_loss"],
                    "validation_pre_feedback_next_margin_loss": validation_metrics["pre_feedback_next_margin_loss"],
                    "validation_post_feedback_next_margin_loss": validation_metrics["post_feedback_next_margin_loss"],
                    "validation_feedback_end_loss": validation_metrics["feedback_end_loss"],
                    "validation_feedback_prob_on_feedback_targets": validation_metrics["feedback_prob_on_feedback_targets"],
                    "validation_feedback_prob_on_next_targets": validation_metrics["feedback_prob_on_next_targets"],
                    "validation_feedback_prob_on_feedback_open_targets": validation_metrics["feedback_prob_on_feedback_open_targets"],
                    "validation_feedback_prob_on_pre_feedback_next_targets": validation_metrics["feedback_prob_on_pre_feedback_next_targets"],
                    "validation_feedback_prob_on_post_feedback_next_targets": validation_metrics["feedback_prob_on_post_feedback_next_targets"],
                    "feedback_action_weight": feedback_action_weight,
                }
                if probe_prediction is not None:
                    eval_record["probe_prediction"] = probe_prediction
                eval_history.append(eval_record)
                probe_feedback_count = (
                    len(probe_prediction["pred_feedbacks"])
                    if probe_prediction is not None
                    else 0
                )
                probe_preview = (
                    probe_prediction["pred_feedbacks"][0][:120]
                    if probe_prediction is not None and probe_prediction["pred_feedbacks"]
                    else "<no feedback>"
                )
                tqdm.write(
                    f"[stage3 eval] step={int(eval_record['step'])} "
                    f"loss={eval_record['running_loss']:.4f} "
                    f"ce_loss={eval_record['running_ce_loss']:.4f} "
                    f"action_loss={eval_record['running_action_loss']:.4f} "
                    f"text_loss={eval_record['running_text_loss']:.4f} "
                    f"feedback_open_margin_loss={eval_record['running_feedback_open_margin_loss']:.4f} "
                    f"pre_feedback_next_margin_loss={eval_record['running_pre_feedback_next_margin_loss']:.4f} "
                    f"post_feedback_next_margin_loss={eval_record['running_post_feedback_next_margin_loss']:.4f} "
                    f"feedback_end_loss={eval_record['running_feedback_end_loss']:.4f} "
                    f"val_loss={validation_metrics['loss']:.4f} "
                    f"val_action_loss={validation_metrics['action_loss']:.4f} "
                    f"val_text_loss={validation_metrics['text_loss']:.4f} "
                    f"probe_feedbacks={probe_feedback_count} "
                    f"probe_preview={probe_preview}"
                )
                model.train()

        epoch_record = {
            "epoch": float(epoch_index + 1),
            "avg_loss": running_loss / max(1, len(dataloader)),
            "avg_ce_loss": running_ce_loss / max(1, len(dataloader)),
            "avg_action_loss": running_action_loss / max(1, len(dataloader)),
            "avg_text_loss": running_text_loss / max(1, len(dataloader)),
            "avg_next_action_loss": running_next_action_loss / max(1, len(dataloader)),
            "avg_feedback_action_loss": running_feedback_action_loss / max(1, len(dataloader)),
            "avg_feedback_open_margin_loss": running_feedback_open_margin_loss / max(1, len(dataloader)),
            "avg_pre_feedback_next_margin_loss": running_pre_feedback_next_margin_loss / max(1, len(dataloader)),
            "avg_post_feedback_next_margin_loss": running_post_feedback_next_margin_loss / max(1, len(dataloader)),
            "avg_feedback_end_loss": running_feedback_end_loss / max(1, len(dataloader)),
            "avg_feedback_action_prob_on_feedback_targets": (
                running_feedback_prob_on_feedback_targets / max(1, len(dataloader))
            ),
            "avg_feedback_action_prob_on_next_targets": (
                running_feedback_prob_on_next_targets / max(1, len(dataloader))
            ),
            "avg_feedback_prob_on_feedback_open_targets": (
                running_feedback_prob_on_feedback_open_targets / max(1, len(dataloader))
            ),
            "avg_feedback_prob_on_pre_feedback_next_targets": (
                running_feedback_prob_on_pre_feedback_next_targets / max(1, len(dataloader))
            ),
            "avg_feedback_prob_on_post_feedback_next_targets": (
                running_feedback_prob_on_post_feedback_next_targets / max(1, len(dataloader))
            ),
            "optimizer_steps": float(global_step),
            "feedback_action_weight": feedback_action_weight,
            "min_feedback_open_sec": float(config.min_feedback_open_sec),
            "feedback_open_margin": float(config.feedback_open_margin),
            "pre_feedback_next_margin": float(config.pre_feedback_next_margin),
            "post_feedback_next_margin": float(config.post_feedback_next_margin),
            "feedback_open_margin_weight": float(config.feedback_open_margin_weight),
            "pre_feedback_next_margin_weight": float(config.pre_feedback_next_margin_weight),
            "post_feedback_next_margin_weight": float(config.post_feedback_next_margin_weight),
            "feedback_end_weight": float(config.feedback_end_weight),
            "next_action_targets": action_stats["next_action_targets"],
            "feedback_action_targets": action_stats["feedback_action_targets"],
            "feedback_open_targets": action_stats["feedback_open_targets"],
            "pre_feedback_next_targets": action_stats["pre_feedback_next_targets"],
            "feedback_end_targets": action_stats["feedback_end_targets"],
            "post_feedback_next_targets": action_stats["post_feedback_next_targets"],
            "raw_action_ratio": action_stats["raw_ratio"],
            "feedback_token_length_p50": action_stats["feedback_token_length_p50"],
            "feedback_token_length_p75": action_stats["feedback_token_length_p75"],
            "feedback_token_length_p90": action_stats["feedback_token_length_p90"],
            "feedback_soft_budget": float(model.stage3_feedback_soft_budget),
        }
        if validation_dataloader is not None:
            model.eval()
            validation_metrics = _evaluate_validation_loss(
                model,
                validation_dataloader,
                feedback_action_weight=feedback_action_weight,
                config=config,
            )
            epoch_record["validation_loss"] = validation_metrics["loss"]
            epoch_record["validation_ce_loss"] = validation_metrics["ce_loss"]
            epoch_record["validation_action_loss"] = validation_metrics["action_loss"]
            epoch_record["validation_text_loss"] = validation_metrics["text_loss"]
            epoch_record["validation_feedback_open_margin_loss"] = validation_metrics["feedback_open_margin_loss"]
            epoch_record["validation_pre_feedback_next_margin_loss"] = validation_metrics["pre_feedback_next_margin_loss"]
            epoch_record["validation_post_feedback_next_margin_loss"] = validation_metrics["post_feedback_next_margin_loss"]
            epoch_record["validation_feedback_end_loss"] = validation_metrics["feedback_end_loss"]
            epoch_record["validation_feedback_action_prob_on_feedback_targets"] = validation_metrics["feedback_prob_on_feedback_targets"]
            epoch_record["validation_feedback_action_prob_on_next_targets"] = validation_metrics["feedback_prob_on_next_targets"]
            epoch_record["validation_feedback_prob_on_feedback_open_targets"] = validation_metrics["feedback_prob_on_feedback_open_targets"]
            epoch_record["validation_feedback_prob_on_pre_feedback_next_targets"] = validation_metrics["feedback_prob_on_pre_feedback_next_targets"]
            epoch_record["validation_feedback_prob_on_post_feedback_next_targets"] = validation_metrics["feedback_prob_on_post_feedback_next_targets"]
            model.train()

        model.eval()
        epoch_probe_prediction = _run_probe_generation(
            model,
            probe_sample=probe_sample,
            probe_generation_kwargs=(
                {
                    **(probe_generation_kwargs or {}),
                    "return_debug": True,
                }
                if probe_sample is not None
                else None
            ),
        )
        model.train()
        if epoch_probe_prediction is not None:
            epoch_probe = _build_probe_epoch_preview(epoch_probe_prediction)
            if epoch_probe is not None:
                first_feedback_visible_sec = epoch_probe.get("first_feedback_visible_sec")
                epoch_probe["feedback_opened_before_patience_target"] = bool(
                    first_feedback_visible_sec is not None
                    and float(first_feedback_visible_sec) < float(config.min_feedback_open_sec)
                )
            epoch_record["epoch_probe"] = epoch_probe

        history.append(epoch_record)

        if output_dir_path is not None:
            model.save_lora_adapter(
                output_dir_path / f"epoch_{epoch_index + 1}",
                extra_config={
                    "training_config": asdict(config),
                    "action_stats": action_stats,
                    "feedback_action_weight": feedback_action_weight,
                    "epoch_record": epoch_record,
                },
            )

    if output_dir_path is not None and eval_history:
        (output_dir_path / "eval_history.json").write_text(
            json.dumps(eval_history, indent=2),
            encoding="utf-8",
        )

    return history
