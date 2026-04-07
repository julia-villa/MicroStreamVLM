"""Stage-2 xattn-only training utilities."""

from src.stage2.dataset import (
    DEFAULT_SYSTEM_PROMPT,
    FEEDBACK_REQUEST,
    Stage2QEVDFit300KDataset,
    Stage2Sample,
    get_request_text,
    stage2_collate,
)
from src.stage2.model import (
    SemanticTokenConfig,
    XAttnConfig,
    XAttnOnlyStreamVLM,
    build_xattn_only_streamvlm,
    resolve_semantic_tokens,
)
from src.stage2.runtime import Stage2RuntimeConfig, resolve_device, resolve_runtime
from src.stage2.training import (
    Stage2TrainingConfig,
    assert_only_xattn_has_gradients,
    build_optimizer,
    prepare_generation_inputs,
    prepare_training_batch,
    train_stage2,
)
from src.stage2.vision_encoder import FrozenEfficientNetStreamEncoder

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "FEEDBACK_REQUEST",
    "FrozenEfficientNetStreamEncoder",
    "SemanticTokenConfig",
    "Stage2QEVDFit300KDataset",
    "Stage2Sample",
    "Stage2TrainingConfig",
    "Stage2RuntimeConfig",
    "XAttnConfig",
    "XAttnOnlyStreamVLM",
    "assert_only_xattn_has_gradients",
    "build_optimizer",
    "build_xattn_only_streamvlm",
    "get_request_text",
    "prepare_generation_inputs",
    "prepare_training_batch",
    "resolve_device",
    "resolve_semantic_tokens",
    "resolve_runtime",
    "stage2_collate",
    "train_stage2",
]
