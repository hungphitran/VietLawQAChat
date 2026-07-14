from __future__ import annotations

from collections import defaultdict


def rrf(
    rankings: list[list[int]],
    weights: list[float],
    **kwargs,
) -> list[int]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into one by summing each cid's
    weighted reciprocal rank `weight / (k + rank)`. Higher `k` smooths top ranks. Returns
    cids sorted by fused score (descending)."""
    k = kwargs.get("k", 60)
    scores: dict[int, float] = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += weight / (k + rank)

    return [
        cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
