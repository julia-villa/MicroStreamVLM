"""Llama 3.2 xattn-only StreamVLM model builder."""

from __future__ import annotations

import json
import os
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.utils import GatedRepoError
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.chat_format import get_instruct_role_guard_token_ids
from src.vision_modules.adapter import XAttnAdapter
from src.vision_modules.cross_attention import CustomDotProdXAttnModule
from src.stage2.runtime import resolve_runtime


@dataclass(frozen=True)
class SemanticTokenConfig:
    """Resolved semantic special tokens."""

    vision_token: str
    answer_begin_token: str
    answer_end_token: str
    vision_token_id: int
    answer_begin_token_id: int
    answer_end_token_id: int
    tokenizer_was_resized: bool = False


def _resolve_existing_token_id(
    tokenizer: PreTrainedTokenizerBase,
    candidates: list[str],
) -> tuple[str, int] | None:
    vocab = tokenizer.get_vocab()
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for candidate in candidates:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if token_id is None:
            continue
        if candidate in vocab or token_id != unk_token_id:
            return candidate, int(token_id)
    return None


def resolve_semantic_tokens(
    tokenizer: PreTrainedTokenizerBase,
    allow_tokenizer_resize: bool = False,
) -> SemanticTokenConfig:
    """Map semantic marker names onto tokenizer-native special/reserved ids when possible."""
    candidate_map = {
        "vision": ["<vision>", "<|reserved_special_token_0|>"],
        "answer_begin": ["<answer>", "<|reserved_special_token_1|>"],
        "answer_end": ["<answer/>", "<|reserved_special_token_2|>", tokenizer.eos_token],
    }

    resolved: dict[str, tuple[str, int]] = {}
    missing: dict[str, list[str]] = {}
    for semantic_name, candidates in candidate_map.items():
        filtered_candidates = [candidate for candidate in candidates if candidate]
        match = _resolve_existing_token_id(tokenizer, filtered_candidates)
        if match is None:
            missing[semantic_name] = filtered_candidates
        else:
            resolved[semantic_name] = match

    tokenizer_was_resized = False
    if missing and allow_tokenizer_resize:
        tokens_to_add = []
        for semantic_name in ["vision", "answer_begin", "answer_end"]:
            if semantic_name in missing:
                tokens_to_add.append(candidate_map[semantic_name][0])
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        resized_tokens = resolve_semantic_tokens(tokenizer, allow_tokenizer_resize=False)
        return _replace_dataclass_field(
            resized_tokens,
            tokenizer_was_resized=True,
        )

    if missing:
        missing_str = "; ".join(
            f"{semantic_name} candidates={candidates}" for semantic_name, candidates in missing.items()
        )
        raise ValueError(
            "Could not resolve semantic special tokens without resizing the tokenizer. "
            f"Missing: {missing_str}"
        )

    return SemanticTokenConfig(
        vision_token=resolved["vision"][0],
        answer_begin_token=resolved["answer_begin"][0],
        answer_end_token=resolved["answer_end"][0],
        vision_token_id=resolved["vision"][1],
        answer_begin_token_id=resolved["answer_begin"][1],
        answer_end_token_id=resolved["answer_end"][1],
        tokenizer_was_resized=tokenizer_was_resized,
    )


@dataclass(frozen=True)
class XAttnConfig:
    """Cross-attention insertion configuration."""

    adapter_insert_layers: tuple[int, ...] = (6, 8, 10, 12, 14, 16, 18, 20, 22)
    xattn_block_size: int = 1
    num_of_xattn_heads: int = 1
    attn_dim: int | None = None


def _replace_dataclass_field(config: SemanticTokenConfig, **kwargs: Any) -> SemanticTokenConfig:
    values = asdict(config)
    values.update(kwargs)
    return SemanticTokenConfig(**values)


class XAttnOnlyStreamVLM(nn.Module):
    """Frozen-backbone StreamVLM wrapper for Llama 3.2 + trainable xattn."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        semantic_tokens: SemanticTokenConfig,
        xattn_config: XAttnConfig,
        vision_feat_dim: int = 1280,
        device: str | None = None,
    ) -> None:
        super().__init__()
        runtime = resolve_runtime(preferred_device=device)
        self.model = model
        self.tokenizer = tokenizer
        self.semantic_tokens = semantic_tokens
        self.xattn_config = xattn_config
        self.vision_feat_dim = vision_feat_dim
        self.device_name = runtime.device
        self.base_llm_model_name_or_path: str | None = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.decoder = self._get_decoder()
        embed_tokens = self.model.get_input_embeddings()
        self.adapter = XAttnAdapter(embed_tokens, list(self.xattn_config.adapter_insert_layers))
        self._uses_native_xattn = False
        self._patch_decoder_layers()
        self._freeze_backbones()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {
            "vision": self.semantic_tokens.vision_token_id,
            "answer_begin": self.semantic_tokens.answer_begin_token_id,
            "answer_end": self.semantic_tokens.answer_end_token_id,
        }

    @property
    def special_token_strings(self) -> dict[str, str]:
        return {
            "vision": self.semantic_tokens.vision_token,
            "answer_begin": self.semantic_tokens.answer_begin_token,
            "answer_end": self.semantic_tokens.answer_end_token,
        }

    def _get_decoder(self) -> nn.Module:
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model
        raise AttributeError("Could not locate Llama decoder layers on the loaded model")

    def _patch_decoder_layers(self) -> None:
        hidden_dim = int(self.model.config.hidden_size)
        attn_dim = int(self.xattn_config.attn_dim or hidden_dim)
        for layer_index in self.xattn_config.adapter_insert_layers:
            decoder_layer = self.decoder.layers[layer_index]
            if getattr(decoder_layer, "_streamvlm_xattn_patched", False):
                continue

            decoder_layer.xattn_layer = CustomDotProdXAttnModule(
                hidden_dim=hidden_dim,
                vision_feat_dim=self.vision_feat_dim,
                attn_dim=attn_dim,
                num_xattn_layers=self.xattn_config.xattn_block_size,
                num_xattn_heads=self.xattn_config.num_of_xattn_heads,
            )
            original_forward = decoder_layer.forward

            def _make_patched_forward(bound_original_forward):
                def forward_with_xattn(this_layer: nn.Module, *args, **kwargs):
                    outputs = bound_original_forward(*args, **kwargs)
                    context = getattr(this_layer, "_streamvlm_ctx", None)
                    if context is None or getattr(this_layer, "xattn_layer", None) is None:
                        return outputs

                    multimodal_embedding = context["multimodal_embedding"]
                    vision_mask = multimodal_embedding["vision_xattn_mask"]
                    if not torch.any(vision_mask):
                        return outputs

                    layer_id = str(context["layer_index"])
                    hidden_states, output_kind = self._extract_hidden_states_from_layer_output(outputs)
                    vision_feats = multimodal_embedding[layer_id]["vision"].to(hidden_states.device)
                    hidden_states, needs_transpose_back = self._align_hidden_states_to_mask(
                        hidden_states,
                        vision_mask,
                    )
                    xattn_values = this_layer.xattn_layer(hidden_states, vision_feats, vision_mask)
                    hidden_states = hidden_states.clone()
                    for batch_index in range(hidden_states.shape[0]):
                        valid_positions = torch.where(vision_mask[batch_index])[0]
                        if valid_positions.numel() == 0:
                            continue
                        hidden_states[batch_index].index_add_(
                            0,
                            valid_positions,
                            xattn_values[batch_index].to(hidden_states),
                        )
                    if needs_transpose_back:
                        hidden_states = hidden_states.transpose(0, 1).contiguous()

                    return self._replace_hidden_states_in_layer_output(
                        outputs,
                        hidden_states,
                        output_kind,
                    )

                return forward_with_xattn

            decoder_layer.forward = types.MethodType(_make_patched_forward(original_forward), decoder_layer)
            decoder_layer._streamvlm_original_forward = original_forward
            decoder_layer._streamvlm_xattn_patched = True

    @staticmethod
    def _align_hidden_states_to_mask(
        hidden_states: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Normalize hidden-state layout to [batch, seq, hidden] for xattn."""
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"Expected 3D hidden states for xattn, got shape {tuple(hidden_states.shape)}"
            )
        if vision_mask.ndim != 2:
            raise RuntimeError(
                f"Expected 2D vision_xattn_mask, got shape {tuple(vision_mask.shape)}"
            )

        if hidden_states.shape[:2] == vision_mask.shape:
            return hidden_states, False

        if hidden_states.shape[0] == vision_mask.shape[1] and hidden_states.shape[1] == vision_mask.shape[0]:
            return hidden_states.transpose(0, 1).contiguous(), True

        raise RuntimeError(
            "Hidden-state shape does not align with vision_xattn_mask. "
            f"hidden_states={tuple(hidden_states.shape)}, vision_xattn_mask={tuple(vision_mask.shape)}"
        )

    @staticmethod
    def _extract_hidden_states_from_layer_output(
        outputs: Any,
    ) -> tuple[torch.Tensor, str]:
        """Normalize decoder outputs so xattn always operates on a 3D hidden-state tensor."""
        if isinstance(outputs, torch.Tensor):
            return outputs, "tensor"
        if isinstance(outputs, tuple):
            if not outputs:
                raise RuntimeError("Decoder layer returned an empty tuple")
            return outputs[0], "tuple"
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state, "object"
        raise RuntimeError(f"Unsupported decoder layer output type for xattn: {type(outputs)}")

    @staticmethod
    def _replace_hidden_states_in_layer_output(
        outputs: Any,
        hidden_states: torch.Tensor,
        output_kind: str,
    ) -> Any:
        """Rebuild the decoder output in the same structural form returned by the original layer."""
        if output_kind == "tensor":
            return hidden_states
        if output_kind == "tuple":
            return (hidden_states,) + outputs[1:]
        if output_kind == "object":
            outputs.last_hidden_state = hidden_states
            return outputs
        raise RuntimeError(f"Unsupported decoder output kind for xattn replacement: {output_kind}")

    def _freeze_backbones(self) -> None:
        for parameter_name, parameter in self.model.named_parameters():
            parameter.requires_grad = "xattn_layer" in parameter_name

    def named_trainable_parameters(self):
        """Yield only trainable xattn parameters."""
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return trainable xattn parameters only."""
        return [parameter for _, parameter in self.named_trainable_parameters()]

    def assert_only_xattn_is_trainable(self) -> None:
        """Validate that only xattn weights remain trainable."""
        invalid_names = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad and "xattn_layer" not in name
        ]
        if invalid_names:
            invalid_str = ", ".join(invalid_names)
            raise RuntimeError(f"Non-xattn parameters left trainable: {invalid_str}")

    def _set_layer_context(self, multimodal_embedding: dict[str, Any]) -> None:
        for layer_index in self.xattn_config.adapter_insert_layers:
            self.decoder.layers[layer_index]._streamvlm_ctx = {
                "layer_index": layer_index,
                "multimodal_embedding": multimodal_embedding,
            }

    def _clear_layer_context(self) -> None:
        for layer_index in self.xattn_config.adapter_insert_layers:
            layer = self.decoder.layers[layer_index]
            if hasattr(layer, "_streamvlm_ctx"):
                delattr(layer, "_streamvlm_ctx")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_feats: dict[str, torch.Tensor | list[int]],
        vision_xattn_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run the frozen LLM with xattn inserted at selected decoder layers."""
        multimodal_embedding = self.adapter(vision_feats, input_ids, vision_xattn_mask)

        self._set_layer_context(multimodal_embedding)
        try:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=kwargs.get("use_cache", False),
                output_attentions=kwargs.get("output_attentions", False),
                output_hidden_states=kwargs.get("output_hidden_states", False),
                return_dict=kwargs.get("return_dict", True),
            )
        finally:
            self._clear_layer_context()

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_feats: dict[str, torch.Tensor | list[int]],
        vision_xattn_mask: torch.Tensor,
        max_new_tokens: int = 128,
        stop_token_ids: list[int] | None = None,
        forbidden_token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Simple greedy generation that keeps xattn active without relying on HF generate()."""
        stop_token_ids = stop_token_ids or [self.semantic_tokens.answer_end_token_id]
        forbidden_token_ids = (
            forbidden_token_ids
            if forbidden_token_ids is not None
            else get_instruct_role_guard_token_ids(self.tokenizer, self.base_llm_model_name_or_path)
        )
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        current_vision_mask = vision_xattn_mask.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=current_attention_mask,
                vision_feats=vision_feats,
                vision_xattn_mask=current_vision_mask,
                labels=None,
                use_cache=False,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            if forbidden_token_ids and bool(
                torch.isin(
                    next_token[0],
                    torch.tensor(forbidden_token_ids, device=next_token.device),
                )
            ):
                break
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones_like(next_token)], dim=1
            )
            current_vision_mask = torch.cat(
                [current_vision_mask, torch.zeros_like(next_token)], dim=1
            )
            if bool(torch.isin(next_token[0], torch.tensor(stop_token_ids, device=next_token.device))):
                break

        return generated_ids

    def save_stage2_checkpoint(
        self,
        output_dir: str | Path,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        """Save tokenizer, model config, and xattn-only weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)

        xattn_state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in self.state_dict().items()
            if "xattn_layer" in name
        }
        torch.save(xattn_state_dict, output_path / "xattn_state_dict.pt")

        metadata = {
            "semantic_tokens": asdict(self.semantic_tokens),
            "xattn_config": asdict(self.xattn_config),
        }
        if extra_config is not None:
            metadata["extra_config"] = extra_config
        with (output_path / "stage2_config.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def load_stage2_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load saved xattn weights into an already-built model."""
        checkpoint_path = Path(checkpoint_dir) / "xattn_state_dict.pt"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)


def _resize_embeddings_if_needed(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    semantic_tokens: SemanticTokenConfig,
) -> SemanticTokenConfig:
    if not semantic_tokens.tokenizer_was_resized:
        return semantic_tokens

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return _replace_dataclass_field(semantic_tokens, tokenizer_was_resized=True)


def build_xattn_only_streamvlm(
    llm_model_name_or_path: str,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    allow_tokenizer_resize: bool = False,
    xattn_config: XAttnConfig | None = None,
    hf_token: str | None = None,
    local_files_only: bool = False,
) -> XAttnOnlyStreamVLM:
    """Load Llama 3.2, resolve semantic tokens, and insert trainable xattn modules."""
    runtime = resolve_runtime(preferred_device=device, llm_dtype=torch_dtype)
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name_or_path,
            **load_kwargs,
        )
    except (GatedRepoError, OSError) as exc:
        message = str(exc)
        if "gated repo" not in message.lower() and "401" not in message:
            raise
        raise RuntimeError(
            "Failed to load the base LLM from Hugging Face. "
            f"`{llm_model_name_or_path}` is gated or requires authentication. "
            "Use one of these options: "
            "1) run `huggingface-cli login`, "
            "2) pass `hf_token=...` to `build_xattn_only_streamvlm(...)`, "
            "3) export `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`, or "
            "4) set `llm_model_name_or_path` to a local directory containing the model files."
        ) from exc
    semantic_tokens = resolve_semantic_tokens(
        tokenizer,
        allow_tokenizer_resize=allow_tokenizer_resize,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            torch_dtype=runtime.llm_dtype,
            **load_kwargs,
        )
    except (GatedRepoError, OSError) as exc:
        message = str(exc)
        if "gated repo" not in message.lower() and "401" not in message:
            raise
        raise RuntimeError(
            "Failed to load model weights for the base LLM. "
            f"`{llm_model_name_or_path}` requires Hugging Face access. "
            "Authenticate with `huggingface-cli login`, pass `hf_token=...`, export "
            "`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`, or point `llm_model_name_or_path` to a local checkpoint."
        ) from exc
    semantic_tokens = _resize_embeddings_if_needed(model, tokenizer, semantic_tokens)
    model.to(runtime.device)

    wrapper = XAttnOnlyStreamVLM(
        model=model,
        tokenizer=tokenizer,
        semantic_tokens=semantic_tokens,
        xattn_config=xattn_config or XAttnConfig(),
        vision_feat_dim=1280,
        device=runtime.device,
    )
    wrapper.base_llm_model_name_or_path = llm_model_name_or_path
    wrapper.to(device=runtime.device, dtype=runtime.llm_dtype)
    wrapper.assert_only_xattn_is_trainable()
    return wrapper
