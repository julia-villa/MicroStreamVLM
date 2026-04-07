"""Shared prompt-format helpers for chat/instruct checkpoints."""

from __future__ import annotations

from typing import Sequence

ROLE_LEAK_SPECIAL_TOKENS = (
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|python_tag|>",
)
ROLE_LEAK_TEXT_MARKERS = (
    "assistant",
    "user",
    "system",
)


def is_instruct_model_name(model_name_or_path: str | None) -> bool:
    return "instruct" in (model_name_or_path or "").lower()


def should_use_chat_template(tokenizer, model_name_or_path: str | None) -> bool:
    if not is_instruct_model_name(model_name_or_path):
        return False
    return bool(getattr(tokenizer, "apply_chat_template", None) and getattr(tokenizer, "chat_template", None))


def _normalize_chat_system_prompt(system_prompt: str) -> str:
    prompt = system_prompt.strip()
    if prompt.startswith("<system>") and prompt.endswith("</system>"):
        prompt = prompt[len("<system>") : -len("</system>")].strip()
    return prompt


def build_chat_aligned_prefix(
    tokenizer,
    system_prompt: str,
    user_content: str,
    model_name_or_path: str | None,
    add_generation_prompt: bool,
) -> list[int]:
    if should_use_chat_template(tokenizer, model_name_or_path):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _normalize_chat_system_prompt(system_prompt)},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer(rendered, add_special_tokens=False).input_ids

    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    return bos_ids + tokenizer(
        f"{system_prompt.strip()}\n{user_content}",
        add_special_tokens=False,
    ).input_ids


def get_instruct_role_guard_token_ids(tokenizer, model_name_or_path: str | None) -> list[int]:
    if not should_use_chat_template(tokenizer, model_name_or_path):
        return []

    token_ids: set[int] = set()
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    for token in ROLE_LEAK_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_token_id:
            token_ids.add(int(token_id))

    for marker in ROLE_LEAK_TEXT_MARKERS:
        encoded = tokenizer(marker, add_special_tokens=False).input_ids
        if len(encoded) == 1:
            token_ids.add(int(encoded[0]))

    return sorted(token_ids)


def trim_role_leakage_text(text: str) -> str:
    trimmed = text
    for special_token in ROLE_LEAK_SPECIAL_TOKENS:
        if special_token in trimmed:
            trimmed = trimmed.split(special_token, 1)[0]

    lowered = trimmed.lower()
    cut_index = None
    for marker in ROLE_LEAK_TEXT_MARKERS:
        for needle in (f"\n{marker}", f" {marker}", f">{marker}", f".{marker}", f":{marker}"):
            idx = lowered.find(needle)
            if idx != -1:
                if cut_index is None or idx < cut_index:
                    cut_index = idx
    if cut_index is not None:
        trimmed = trimmed[:cut_index]

    return trimmed.strip()


def contains_role_leakage(text: str) -> bool:
    if any(token in text for token in ROLE_LEAK_SPECIAL_TOKENS):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in ROLE_LEAK_TEXT_MARKERS)


def build_repeated_special_token_string(token: str, count: int) -> str:
    return "".join(token for _ in range(max(0, int(count))))


def token_mask_from_ids(token_ids: Sequence[int], active_token_id: int) -> list[int]:
    return [2 if int(token_id) == int(active_token_id) else 0 for token_id in token_ids]
