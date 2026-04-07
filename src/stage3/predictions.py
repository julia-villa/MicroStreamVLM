"""Standalone stage-3 prediction file helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def save_predictions(predictions: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    """Persist stage-3 predictions to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(predictions), indent=2), encoding="utf-8")


def load_predictions(predictions: str | Path | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load stage-3 predictions from a JSON file or an in-memory sequence."""
    if isinstance(predictions, (str, Path)):
        return json.loads(Path(predictions).read_text())
    return list(predictions)


__all__ = ["load_predictions", "save_predictions"]
