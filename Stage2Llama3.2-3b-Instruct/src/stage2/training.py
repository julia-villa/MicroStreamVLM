"""Stage-2 batch preparation and training utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.chat_format import (
    build_chat_aligned_prefix,
    build_repeated_special_token_string,
    token_mask_from_ids,
)
from src.stage2.dataset import DEFAULT_SYSTEM_PROMPT, Stage2Sample, get_request_text
from src.stage2.runtime import resolve_runtime


@dataclass(frozen=True)
class Stage2TrainingConfig:
    """Default stage-2 xattn-only optimization settings."""

    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip_norm: float = 1.0
    epochs: int = 2
    effective_batch_size: int = 32
    micro_batch_size: int = 1
    max_new_tokens_eval: int = 128
    log_every: int = 10
    eval_every_steps: int = 0

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


def _as_stage2_samples(batch: dict[str, list[Any]]) -> list[Stage2Sample]:
    return [
        Stage2Sample(
            sample_id=batch["sample_id"][index],
            video_path=batch["video_path"][index],
            task_type=batch["task_type"][index],
            answer=batch["answer"][index],
            question=batch.get("question", [None] * len(batch["sample_id"]))[index],
            system_prompt=batch.get("system_prompt", [DEFAULT_SYSTEM_PROMPT] * len(batch["sample_id"]))[
                index
            ]
            or DEFAULT_SYSTEM_PROMPT,
            rotate_90_cw=batch.get("rotate_90_cw", [False] * len(batch["sample_id"]))[index],
            annotation_source=batch.get("annotation_source", [None] * len(batch["sample_id"]))[index],
            query_type=batch.get("query_type", [None] * len(batch["sample_id"]))[index],
            labels=batch.get("labels", [None] * len(batch["sample_id"]))[index],
            labels_descriptive=batch.get("labels_descriptive", [None] * len(batch["sample_id"]))[
                index
            ],
        )
        for index in range(len(batch["sample_id"]))
    ]


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


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _build_tokenized_sample(
    sample: Stage2Sample,
    model,
    tokenizer,
    special_token_ids: dict[str, int],
    special_token_strings: dict[str, str],
    num_vision_tokens: int,
) -> tuple[list[int], list[int], list[int]]:
    request_text = get_request_text(sample)
    answer_ids = _encode_text(tokenizer, sample.answer.strip())
    answer_begin_id = special_token_ids["answer_begin"]
    answer_end_id = special_token_ids["answer_end"]
    model_name_or_path = getattr(model, "base_llm_model_name_or_path", None)

    if getattr(model, "base_llm_model_name_or_path", None) and "instruct" in model_name_or_path.lower():
        user_content = (
            f"{build_repeated_special_token_string(special_token_strings['vision'], num_vision_tokens)}"
            f"\n{request_text.strip()}\n"
        )
        prefix_ids = build_chat_aligned_prefix(
            tokenizer=tokenizer,
            system_prompt=sample.system_prompt,
            user_content=user_content,
            model_name_or_path=model_name_or_path,
            add_generation_prompt=True,
        ) + [answer_begin_id]
        prefix_vision_mask = token_mask_from_ids(prefix_ids, special_token_ids["vision"])
    else:
        system_ids = _encode_text(tokenizer, f"{sample.system_prompt.strip()}\n")
        request_ids = _encode_text(tokenizer, f"\n{request_text.strip()}\n")
        bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        vision_ids = [special_token_ids["vision"]] * num_vision_tokens
        prefix_ids = bos_ids + system_ids + vision_ids + request_ids + [answer_begin_id]
        prefix_vision_mask = (
            ([0] * (len(bos_ids) + len(system_ids)))
            + ([2] * len(vision_ids))
            + ([0] * (len(request_ids) + 1))
        )

    input_ids = prefix_ids + answer_ids + [answer_end_id]
    labels = ([-100] * len(prefix_ids)) + answer_ids + [answer_end_id]
    vision_xattn_mask = prefix_vision_mask + ([0] * (len(answer_ids) + 1))
    return input_ids, labels, vision_xattn_mask


def prepare_training_batch(
    batch: dict[str, list[Any]],
    model,
    vision_encoder,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Encode raw clips and build token-level stage-2 training tensors."""
    samples = _as_stage2_samples(batch)
    special_token_ids = model.special_token_ids
    special_token_strings = model.special_token_strings
    tokenizer = model.tokenizer
    device = torch.device(device or model.device)

    input_id_sequences: list[list[int]] = []
    label_sequences: list[list[int]] = []
    vision_mask_sequences: list[list[int]] = []
    vision_sequences: list[torch.Tensor] = []
    spatial_res = None

    for sample in samples:
        encoded_video = vision_encoder.encode_clip(
            video_path=sample.video_path,
            rotate_90_cw=sample.rotate_90_cw,
        )
        vision_sequence = encoded_video["feats"]
        spatial_res = encoded_video["spatial_res"]
        vision_sequences.append(vision_sequence)

        input_ids, labels, vision_mask = _build_tokenized_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
            special_token_strings=special_token_strings,
            num_vision_tokens=vision_sequence.shape[0],
        )
        input_id_sequences.append(input_ids)
        label_sequences.append(labels)
        vision_mask_sequences.append(vision_mask)

    feature_lengths = {sequence.shape[0] for sequence in vision_sequences}
    if len(feature_lengths) > 1:
        raise ValueError(
            "Variable-length clips cannot be mixed in one micro-batch with the current "
            "cross-attention implementation because padded vision timesteps are not masked. "
            "Use micro_batch_size=1 or bucket clips by identical feature length."
        )

    padded_input_ids = _pad_sequences(
        input_id_sequences,
        pad_value=tokenizer.pad_token_id,
        dtype=torch.long,
    ).to(device)
    padded_labels = _pad_sequences(label_sequences, pad_value=-100, dtype=torch.long).to(device)
    padded_vision_mask = _pad_sequences(vision_mask_sequences, pad_value=0, dtype=torch.long).to(device)
    attention_mask = (padded_input_ids != tokenizer.pad_token_id).long().to(device)
    padded_vision_feats = _pad_vision_sequences(vision_sequences).to(device)

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "vision_xattn_mask": padded_vision_mask,
        "vision_feats": {
            "feats": padded_vision_feats,
            "spatial_res": spatial_res or [5, 7],
        },
        "samples": samples,
    }


def prepare_generation_inputs(
    sample: Stage2Sample,
    model,
    vision_encoder,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Prepare one sample for smoke-test generation."""
    device = torch.device(device or model.device)
    tokenizer = model.tokenizer
    special_token_ids = model.special_token_ids
    special_token_strings = model.special_token_strings

    encoded_video = vision_encoder.encode_clip(
        video_path=sample.video_path,
        rotate_90_cw=sample.rotate_90_cw,
    )
    model_name_or_path = getattr(model, "base_llm_model_name_or_path", None)
    if model_name_or_path and "instruct" in model_name_or_path.lower():
        user_content = (
            f"{build_repeated_special_token_string(special_token_strings['vision'], encoded_video['feats'].shape[0])}"
            f"\n{get_request_text(sample).strip()}\n"
        )
        input_ids = build_chat_aligned_prefix(
            tokenizer=tokenizer,
            system_prompt=sample.system_prompt,
            user_content=user_content,
            model_name_or_path=model_name_or_path,
            add_generation_prompt=True,
        ) + [special_token_ids["answer_begin"]]
        vision_mask = token_mask_from_ids(input_ids, special_token_ids["vision"])
    else:
        system_ids = _encode_text(tokenizer, f"{sample.system_prompt.strip()}\n")
        request_ids = _encode_text(tokenizer, f"\n{get_request_text(sample).strip()}\n")
        bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        vision_ids = [special_token_ids["vision"]] * encoded_video["feats"].shape[0]
        input_ids = bos_ids + system_ids + vision_ids + request_ids + [special_token_ids["answer_begin"]]
        vision_mask = (
            ([0] * (len(bos_ids) + len(system_ids)))
            + ([2] * len(vision_ids))
            + ([0] * (len(request_ids) + 1))
        )

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask_t = torch.ones_like(input_ids_t)
    vision_mask_t = torch.tensor([vision_mask], dtype=torch.long, device=device)
    vision_feats_t = encoded_video["feats"].unsqueeze(0).to(device)

    return {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "vision_xattn_mask": vision_mask_t,
        "vision_feats": {
            "feats": vision_feats_t,
            "spatial_res": encoded_video["spatial_res"],
        },
    }


def build_optimizer(model, config: Stage2TrainingConfig) -> AdamW:
    """Create the xattn-only optimizer."""
    trainable_parameters = model.trainable_parameters()
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found; xattn patching likely failed")

    return AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )


def assert_only_xattn_has_gradients(model) -> None:
    """Fail if a frozen parameter unexpectedly receives gradients."""
    invalid_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and "xattn_layer" not in name
    ]
    if invalid_names:
        invalid_str = ", ".join(invalid_names)
        raise RuntimeError(f"Frozen parameters received gradients: {invalid_str}")


def _evaluate_validation_loss(
    model,
    dataloader,
    vision_encoder,
    runtime,
) -> float:
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            prepared = prepare_training_batch(batch, model=model, vision_encoder=vision_encoder)
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
                    labels=prepared["labels"],
                )
            _raise_if_nonfinite("validation.outputs.logits", outputs.logits)
            _raise_if_nonfinite("validation.outputs.loss", outputs.loss)
            total_loss += float(outputs.loss.item())
            batch_count += 1

    return total_loss / max(1, batch_count)


def _generate_probe_text(
    model,
    sample: Stage2Sample,
    vision_encoder,
    max_new_tokens: int,
) -> str:
    generation_inputs = prepare_generation_inputs(
        sample,
        model=model,
        vision_encoder=vision_encoder,
    )
    generated_ids = model.generate_greedy(
        input_ids=generation_inputs["input_ids"],
        attention_mask=generation_inputs["attention_mask"],
        vision_feats=generation_inputs["vision_feats"],
        vision_xattn_mask=generation_inputs["vision_xattn_mask"],
        max_new_tokens=max_new_tokens,
    )
    return model.tokenizer.decode(generated_ids[0], skip_special_tokens=False)


def _save_eval_history(output_dir: Path, eval_history: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))


def train_stage2(
    model,
    dataloader,
    vision_encoder,
    config: Stage2TrainingConfig,
    output_dir: str | Path | None = None,
    validation_dataloader=None,
    probe_sample: Stage2Sample | None = None,
) -> list[dict[str, float]]:
    """Train xattn-only stage-2 alignment with gradient accumulation."""
    runtime = resolve_runtime(
        preferred_device=model.device.type,
        llm_dtype=next(model.model.parameters()).dtype,
    )
    model.train()
    optimizer = build_optimizer(model, config)
    grad_accum_steps = config.gradient_accumulation_steps
    use_autocast = runtime.use_autocast
    scaler_dtype = runtime.llm_dtype
    history: list[dict[str, float]] = []
    eval_history: list[dict[str, Any]] = []
    global_step = 0
    output_dir_path = Path(output_dir) if output_dir is not None else None

    for epoch_index in range(config.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(dataloader, desc=f"epoch {epoch_index + 1}/{config.epochs}")

        for step_index, batch in enumerate(progress_bar, start=1):
            prepared = prepare_training_batch(batch, model=model, vision_encoder=vision_encoder)

            with torch.autocast(
                device_type=model.device.type,
                dtype=scaler_dtype,
                enabled=use_autocast,
            ):
                outputs = model(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    vision_feats=prepared["vision_feats"],
                    vision_xattn_mask=prepared["vision_xattn_mask"],
                    labels=prepared["labels"],
                )
                _raise_if_nonfinite("outputs.logits", outputs.logits)
                _raise_if_nonfinite("outputs.loss", outputs.loss)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            _raise_if_nonfinite_trainable_gradients(model)
            running_loss += float(loss.item()) * grad_accum_steps

            should_step = (step_index % grad_accum_steps == 0) or (step_index == len(dataloader))
            if should_step:
                clip_grad_norm_(model.trainable_parameters(), config.grad_clip_norm)
                assert_only_xattn_has_gradients(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step_index % config.log_every == 0 or should_step:
                progress_bar.set_postfix(loss=f"{running_loss / step_index:.4f}")

            should_eval = (
                config.eval_every_steps > 0
                and (step_index % config.eval_every_steps == 0)
                and (validation_dataloader is not None or probe_sample is not None)
            )
            if should_eval:
                model.eval()
                eval_record: dict[str, Any] = {
                    "step": int((epoch_index * len(dataloader)) + step_index),
                    "epoch": float(epoch_index + (step_index / max(1, len(dataloader)))),
                    "running_train_loss": running_loss / step_index,
                }
                if validation_dataloader is not None:
                    validation_loss = _evaluate_validation_loss(
                        model=model,
                        dataloader=validation_dataloader,
                        vision_encoder=vision_encoder,
                        runtime=runtime,
                    )
                    eval_record["validation_loss"] = validation_loss
                if probe_sample is not None:
                    probe_text = _generate_probe_text(
                        model=model,
                        sample=probe_sample,
                        vision_encoder=vision_encoder,
                        max_new_tokens=config.max_new_tokens_eval,
                    )
                    eval_record["probe_sample_id"] = probe_sample.sample_id
                    eval_record["probe_generation"] = probe_text

                eval_history.append(eval_record)
                if output_dir_path is not None:
                    _save_eval_history(output_dir_path, eval_history)

                summary = (
                    f"[eval] step={eval_record['step']} "
                    f"train_loss={eval_record['running_train_loss']:.4f}"
                )
                if "validation_loss" in eval_record:
                    summary += f" val_loss={eval_record['validation_loss']:.4f}"
                tqdm.write(summary)
                if "probe_generation" in eval_record:
                    tqdm.write("[eval] probe generation:")
                    tqdm.write(eval_record["probe_generation"])
                model.train()

        epoch_record = {
            "epoch": float(epoch_index + 1),
            "avg_loss": running_loss / max(1, len(dataloader)),
            "optimizer_steps": float(global_step),
        }
        history.append(epoch_record)

        if output_dir is not None:
            model.save_stage2_checkpoint(
                Path(output_dir) / f"epoch_{epoch_index + 1}",
                extra_config={
                    "training_config": asdict(config),
                    "epoch_record": epoch_record,
                },
            )

    if output_dir_path is not None and eval_history:
        _save_eval_history(output_dir_path, eval_history)

    return history
