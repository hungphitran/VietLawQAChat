from __future__ import annotations

from typing import Literal, TypeAlias, cast

from pyvi import ViTokenizer
from underthesea import word_tokenize

SegmentationMethod: TypeAlias = Literal["pyvi", "underthesea"] | None


def segment_text(text: str, method: SegmentationMethod = "underthesea") -> str:
    if method is None:
        return text
    if method == "pyvi":
        return ViTokenizer.tokenize(text)
    if method == "underthesea":
        return cast(str, word_tokenize(text, format="text"))
    raise ValueError(f"Unsupported segmentation method: {method}")
