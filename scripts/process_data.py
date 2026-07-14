"""Process raw data into train/eval CSVs."""

from __future__ import annotations

import argparse

import yaml

from vnlegal_rag_v2.data.pipeline import DataPreparationPipeline
from vnlegal_rag_v2.utils.text import SegmentationMethod


def _seg(value: str | None) -> SegmentationMethod:
    if value is None or value in ("none", "None", "null", ""):
        return None
    if value in ("pyvi", "underthesea"):
        return value
    raise ValueError(f"Unsupported segmentation: {value!r}")


def _parse_segmentation(value) -> list[SegmentationMethod]:
    """Parse segmentation config — single value or list."""
    if value is None:
        return [None]
    if isinstance(value, list):
        return [_seg(v) for v in value]
    return [_seg(value)]


def main():
    parser = argparse.ArgumentParser(description="Process raw data into train/eval splits")
    parser.add_argument("--config", type=str, required=True, help="Path to data config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = DataPreparationPipeline(
        raw_path=config["raw_path"],
        processed_path=config["processed_path"],
        eval_size=config.get("eval_size", 0.1),
        random_state=config.get("random_state", 36),
        segmentation_methods=_parse_segmentation(config.get("segmentation")),
        overwrite=config.get("overwrite", False),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
