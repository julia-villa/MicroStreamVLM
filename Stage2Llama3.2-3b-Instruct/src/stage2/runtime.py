"""Backend and dtype helpers for stage-2 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    """Resolved runtime backend and dtype policy."""

    device: str
    llm_dtype: torch.dtype
    vision_dtype: torch.dtype
    use_autocast: bool


def resolve_device(preferred_device: str | None = None) -> str:
    """Resolve the best available device."""
    if preferred_device is not None:
        return preferred_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_runtime(
    preferred_device: str | None = None,
    llm_dtype: torch.dtype | None = None,
    vision_dtype: torch.dtype | None = None,
    use_autocast: bool | None = None,
) -> Stage2RuntimeConfig:
    """Resolve backend-safe runtime defaults for cluster or Apple Silicon execution."""
    device = resolve_device(preferred_device)

    if llm_dtype is None:
        if device == "cuda":
            llm_dtype = torch.bfloat16
        elif device == "mps":
            llm_dtype = torch.float32
        else:
            llm_dtype = torch.float32

    if vision_dtype is None:
        if device == "cuda":
            vision_dtype = torch.float32
        elif device == "mps":
            vision_dtype = torch.float32
        else:
            vision_dtype = torch.float32

    if use_autocast is None:
        use_autocast = device == "cuda"

    return Stage2RuntimeConfig(
        device=device,
        llm_dtype=llm_dtype,
        vision_dtype=vision_dtype,
        use_autocast=use_autocast,
    )
