"""Stage-3 utilities with lazy exports to avoid unnecessary model dependencies."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "LEGACY_STAGE3_SYSTEM_PROMPT": "src.stage3.manifest",
    "STAGE3_SYSTEM_PROMPT": "src.stage3.manifest",
    "STAGE3_STYLE_PROMPT": "src.stage3.manifest",
    "Stage3SegmentRecord": "src.stage3.manifest",
    "build_stage3_text_prefix": "src.stage3.manifest",
    "extract_exercise_name": "src.stage3.manifest",
    "get_feedback_spans": "src.stage3.manifest",
    "limit_records": "src.stage3.manifest",
    "load_long_range_segments": "src.stage3.manifest",
    "load_segment_manifest": "src.stage3.manifest",
    "load_video_timestamps": "src.stage3.manifest",
    "resolve_stage3_system_prompt": "src.stage3.manifest",
    "save_segment_manifest": "src.stage3.manifest",
    "segment_long_range_record": "src.stage3.manifest",
    "split_train_validation_segments": "src.stage3.manifest",
    "Stage3SegmentDataset": "src.stage3.dataset",
    "stage3_collate": "src.stage3.dataset",
    "evaluate_predictions": "src.stage3.evaluation",
    "load_predictions": "src.stage3.predictions",
    "save_predictions": "src.stage3.predictions",
    "build_segment_feature_cache": "src.stage3.cache",
    "get_segment_cache_path": "src.stage3.cache",
    "load_cached_segment": "src.stage3.cache",
    "generate_benchmark_predictions": "src.stage3.generation",
    "generate_prediction_from_features": "src.stage3.generation",
    "generate_segment_prediction": "src.stage3.generation",
    "trace_generation_step_choices": "src.stage3.generation",
    "run_live_webcam_inference": "src.stage3.live",
    "Stage3LoRAStreamVLM": "src.stage3.model",
    "Stage3LoraConfig": "src.stage3.model",
    "build_stage3_lora_streamvlm": "src.stage3.model",
    "Stage3TrainingConfig": "src.stage3.training",
    "compute_stage3_action_statistics": "src.stage3.training",
    "compute_stage3_loss_metrics": "src.stage3.training",
    "prepare_stage3_batch": "src.stage3.training",
    "train_stage3": "src.stage3.training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
