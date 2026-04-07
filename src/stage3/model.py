"""Stage-3 LoRA model builder on top of the stage-2 xattn checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.utils import GatedRepoError
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.stage2.model import (
    XAttnConfig,
    XAttnOnlyStreamVLM,
    _resize_embeddings_if_needed,
    resolve_semantic_tokens,
)
from src.stage2.runtime import resolve_runtime


@dataclass(frozen=True)
class Stage3LoraConfig:
    """LoRA and optional QLoRA configuration for stage-3 fine-tuning."""

    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_qlora: bool = False


class Stage3LoRAStreamVLM(XAttnOnlyStreamVLM):
    """Stage-3 StreamVLM wrapper with frozen stage-2 vision/xattn and trainable LoRA LLM."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage2_checkpoint_dir: str | None = None
        self.lora_config: Stage3LoraConfig | None = None
        self.base_llm_model_name_or_path: str | None = None

    @property
    def timeline_token_ids(self) -> dict[str, int]:
        return {
            "next": self.semantic_tokens.vision_token_id,
            "feedback_begin": self.semantic_tokens.answer_begin_token_id,
            "feedback_end": self.semantic_tokens.answer_end_token_id,
        }

    @property
    def timeline_token_strings(self) -> dict[str, str]:
        return {
            "next": self.semantic_tokens.vision_token,
            "feedback_begin": self.semantic_tokens.answer_begin_token,
            "feedback_end": self.semantic_tokens.answer_end_token,
        }

    def enable_lora(self, lora_config: Stage3LoraConfig) -> None:
        """Attach LoRA adapters to the LLM and freeze everything else."""
        self.lora_config = lora_config

        # Freeze the stage-2 base before inserting LoRA modules.
        for _, parameter in self.named_parameters():
            parameter.requires_grad = False

        if lora_config.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            target_modules=list(lora_config.target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)

        for name, parameter in self.named_parameters():
            parameter.requires_grad = "lora_" in name

    def named_trainable_parameters(self):
        """Yield only trainable LoRA parameters."""
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_trainable_parameters()]

    def assert_only_lora_is_trainable(self) -> None:
        invalid_names = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if invalid_names:
            invalid_str = ", ".join(invalid_names)
            raise RuntimeError(f"Non-LoRA parameters left trainable in stage-3: {invalid_str}")

    def save_lora_adapter(
        self,
        output_dir: str | Path,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        """Save PEFT adapter weights and stage-3 metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Stage-3 model does not have an attached PEFT adapter")

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        metadata = {
            "stage2_checkpoint_dir": self.stage2_checkpoint_dir,
            "base_llm_model_name_or_path": self.base_llm_model_name_or_path,
            "lora_config": asdict(self.lora_config) if self.lora_config is not None else None,
            "timeline_token_ids": self.timeline_token_ids,
            "timeline_token_strings": self.timeline_token_strings,
        }
        if extra_config is not None:
            metadata["extra_config"] = extra_config
        (output_path / "stage3_adapter_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_lora_adapter(self, adapter_dir: str | Path, is_trainable: bool = False) -> None:
        """Load saved LoRA adapter weights into the already-built stage-3 base."""
        self.model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=is_trainable)
        if is_trainable:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = "lora_" in name
        else:
            for _, parameter in self.named_parameters():
                parameter.requires_grad = False


def load_stage2_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load stage-2 checkpoint metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    return json.loads((checkpoint_dir / "stage2_config.json").read_text())


def _build_quantization_kwargs(use_qlora: bool, runtime_device: str) -> dict[str, Any]:
    if not use_qlora:
        return {}
    if runtime_device != "cuda":
        raise RuntimeError("QLoRA is only supported on CUDA profiles")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA requested but BitsAndBytesConfig is unavailable. Install bitsandbytes on CUDA."
        ) from exc

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        "device_map": {"": 0},
    }


def build_stage3_lora_streamvlm(
    stage2_checkpoint_dir: str | Path,
    device: str | None = None,
    llm_dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    hf_token: str | None = None,
    local_files_only: bool = False,
    lora_config: Stage3LoraConfig | None = None,
) -> Stage3LoRAStreamVLM:
    """Build the stage-3 LoRA model from a validated stage-2 checkpoint."""
    stage2_checkpoint_dir = Path(stage2_checkpoint_dir)
    metadata = load_stage2_checkpoint_metadata(stage2_checkpoint_dir)
    runtime = resolve_runtime(preferred_device=device, llm_dtype=llm_dtype)

    resolved_hf_token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if resolved_hf_token:
        load_kwargs["token"] = resolved_hf_token

    tokenizer = AutoTokenizer.from_pretrained(
        stage2_checkpoint_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    semantic_tokens = resolve_semantic_tokens(tokenizer, allow_tokenizer_resize=False)

    llm_model_name_or_path = metadata["extra_config"]["llm_model_name_or_path"]
    xattn_config = XAttnConfig(**metadata["xattn_config"])

    model_load_kwargs = _build_quantization_kwargs(
        use_qlora=(lora_config.use_qlora if lora_config is not None else False),
        runtime_device=runtime.device,
    )
    if not model_load_kwargs:
        model_load_kwargs["torch_dtype"] = runtime.llm_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            **load_kwargs,
            **model_load_kwargs,
        )
    except (GatedRepoError, OSError) as exc:
        message = str(exc)
        if "gated repo" not in message.lower() and "401" not in message:
            raise
        raise RuntimeError(
            "Failed to load base model weights for stage-3. Authenticate with Hugging Face "
            f"or provide a local checkpoint directory for `{llm_model_name_or_path}`."
        ) from exc

    semantic_tokens = _resize_embeddings_if_needed(model, tokenizer, semantic_tokens)
    if runtime.device != "cuda" or not (lora_config and lora_config.use_qlora):
        model.to(runtime.device)

    wrapper = Stage3LoRAStreamVLM(
        model=model,
        tokenizer=tokenizer,
        semantic_tokens=semantic_tokens,
        xattn_config=xattn_config,
        vision_feat_dim=1280,
        device=runtime.device,
    )
    wrapper.stage2_checkpoint_dir = str(stage2_checkpoint_dir)
    wrapper.base_llm_model_name_or_path = llm_model_name_or_path
    wrapper.load_stage2_checkpoint(stage2_checkpoint_dir)

    if runtime.device == "cuda" and lora_config is not None and lora_config.use_qlora:
        wrapper.adapter.to(runtime.device, dtype=torch.float32)
        for layer_index in wrapper.xattn_config.adapter_insert_layers:
            wrapper.decoder.layers[layer_index].xattn_layer.to(runtime.device, dtype=torch.float32)
    else:
        wrapper.to(device=runtime.device, dtype=runtime.llm_dtype)

    wrapper.enable_lora(lora_config or Stage3LoraConfig())
    wrapper.assert_only_lora_is_trainable()
    return wrapper
