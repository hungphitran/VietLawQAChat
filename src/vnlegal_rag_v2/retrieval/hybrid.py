from __future__ import annotations

from pathlib import Path
from typing import Callable

from vnlegal_rag_v2.retrieval.fusion import rrf


class HybridRetriever:
    def __init__(
        self,
        retrievers: list,
        weights: list[float] | None = None,
        fusion: Callable = rrf,
        fusion_kwargs: dict | None = None,
    ):
        """Combine multiple retrievers via score fusion (default: Reciprocal Rank Fusion).

        `weights` scales each retriever's contribution in fusion; a 0-weight retriever is
        skipped entirely (never indexed, never queried).
        """
        assert len(retrievers) > 0
        if weights is not None:
            assert len(weights) == len(retrievers)
        self.retrievers = retrievers
        self.weights = weights
        self._fusion = fusion
        self._fusion_kwargs = fusion_kwargs or {}

    def _active_indices(self) -> list[int]:
        """Indices of retrievers with nonzero weight (all of them if weights unset).

        A retriever weighted 0 contributes nothing to fusion, so it's skipped
        entirely — no indexing, no per-query retrieve. Set w_bm25=0 (or w_dense=0)
        to drop that leg and save its compute.
        """
        if self.weights is None:
            return list(range(len(self.retrievers)))
        return [i for i, w in enumerate(self.weights) if w]

    def index(self, documents: list[str], cids: list[int]) -> None:
        for i in self._active_indices():
            self.retrievers[i].index(documents, cids)

    def retrieve(self, queries: list[str], top_k: int = 100) -> list[list[int]]:
        """Fuse each retriever's per-query ranking. Single-active-retriever is a fast path
        that returns its ranking directly (no fusion overhead)."""
        active = self._active_indices()
        assert active, "HybridRetriever: all weights are zero"

        # fast path: a single active retriever → its ranking directly, skip fusion
        if len(active) == 1:
            sole = self.retrievers[active[0]]
            return [sole.retrieve([q], top_k)[0] for q in queries]

        weights = self.weights or [1.0 / len(self.retrievers)] * len(self.retrievers)
        active_weights = [weights[i] for i in active]

        results = []
        for query in queries:
            rankings = [self.retrievers[i].retrieve([query], top_k)[0] for i in active]
            fused = self._fusion(rankings, active_weights, **self._fusion_kwargs)
            results.append(fused[:top_k])

        return results

    # --- index persistence: forward to active children only ---
    def index_signature(self) -> str:
        parts = [
            f"hybrid|weights={self.weights}|fusion={getattr(self._fusion, '__name__', 'fn')}"
            f"|fk={sorted(self._fusion_kwargs.items())}",
        ]
        for i in self._active_indices():
            parts.append(self.retrievers[i].index_signature())
        return "|".join(parts)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        for i in self._active_indices():
            self.retrievers[i].save(p / f"retriever_{i}")

    def load(self, path: str | Path) -> None:
        p = Path(path)
        for i in self._active_indices():
            self.retrievers[i].load(p / f"retriever_{i}")
