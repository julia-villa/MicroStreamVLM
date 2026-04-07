#!/usr/bin/env python3
"""Create a stage-3 segment manifest from long-range metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", required=True, help="Path to feedbacks_long_range_*.json")
    parser.add_argument("--video-dir", required=True, help="Directory containing long-range videos and timestamp files")
    parser.add_argument("--split", required=True, help="Split label to embed into segment_id, e.g. benchmark/train")
    parser.add_argument("--output-path", required=True, help="Where to write the segment manifest JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from src.stage3.manifest import load_long_range_segments, save_segment_manifest

    records = load_long_range_segments(
        metadata_path=args.metadata_path,
        video_dir=args.video_dir,
        split=args.split,
    )
    save_segment_manifest(records, args.output_path)
    print(f"saved {len(records)} segment records to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
