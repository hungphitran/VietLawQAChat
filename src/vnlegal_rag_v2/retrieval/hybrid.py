from __future__ import annotations

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
        assert len(retrievers) > 0
        if weights is not None:
            assert len(weights) == len(retrievers)
        self.retrievers = retrievers
        self.weights = weights
        self._fusion = fusion
        self._fusion_kwargs = fusion_kwargs or {}

    def index(self, documents: list[str], cids: list[int]) -> None:
        for retriever in self.retrievers:
            retriever.index(documents, cids)

    def retrieve(self, queries: list[str], top_k: int = 100) -> list[list[int]]:
        n = len(self.retrievers)
        weights = self.weights or [1.0 / n] * n

        results = []
        for i, query in enumerate(queries):
            rankings = [retriever.retrieve([query], top_k)[0] for retriever in self.retrievers]
            fused = self._fusion(rankings, weights, **self._fusion_kwargs)
            results.append(fused[:top_k])

        return results
