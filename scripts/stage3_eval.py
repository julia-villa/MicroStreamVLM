#!/usr/bin/env python3
"""Evaluate stage-3 prediction JSON against a segment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to the prediction JSON file")
    parser.add_argument("--references", required=True, help="Path to the stage-3 segment manifest JSON")
    parser.add_argument("--output-path", required=True, help="Where to write the metrics JSON")
    parser.add_argument("--tolerance", type=float, default=3.0, help="Temporal matching tolerance in seconds")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.stage3.evaluation import evaluate_predictions

    metrics = evaluate_predictions(
        predictions=args.predictions,
        references=args.references,
        output_path=args.output_path,
        tolerance=args.tolerance,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
