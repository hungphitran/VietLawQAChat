from __future__ import annotations

from vnlegal_rag_v2.evaluation.evaluator import Evaluator
from vnlegal_rag_v2.evaluation.metrics import MetricFn


class RAGPipeline:
    def __init__(
        self,
        retriever,
        rerankers: list | None = None,
        top_k_retrieval: int = 100,
        top_k_rerank: int = 10,
    ):
        self.retriever = retriever
        self.rerankers: list = rerankers or []
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self._documents: list[str] | None = None
        self._cids: list[int] | None = None

    def index(self, documents: list[str], cids: list[int]) -> None:
        self._documents = list(documents)
        self._cids = list(cids)
        self.retriever.index(documents, cids)

    # --- individual stages ---

    def retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[list[int]]:
        assert self._cids is not None, "Call index() first"
        return self.retriever.retrieve(queries, top_k or self.top_k_retrieval)

    def rerank(
        self,
        queries: list[str],
        candidates: list[list[int]],
        top_k: int | None = None,
    ) -> list[list[int]]:
        """Rerank retrieval candidates through each reranker in sequence."""
        assert self._documents is not None, "Call index() first"
        assert self.rerankers, "No rerankers configured"

        result = candidates
        k = top_k or self.top_k_rerank
        for reranker in self.rerankers:
            result = reranker.rerank(
                queries, result, self._documents, self._cids, k
            )
        return result

    # --- combined ---

    def query(
        self,
        queries: list[str],
        top_k_retrieval: int | None = None,
        top_k_rerank: int | None = None,
    ) -> list[list[int]]:
        candidates = self.retrieve(queries, top_k_retrieval)
        if self.rerankers:
            candidates = self.rerank(queries, candidates, top_k_rerank)
        return candidates

    # --- evaluation ---

    def evaluate(
        self,
        queries: list[str],
        relevant_cids: list[list[int]],
        metrics: list[tuple[str, MetricFn, int]] | None = None,
        top_k_retrieval: int | None = None,
        top_k_rerank: int | None = None,
    ) -> dict[str, float]:
        predictions = self.query(queries, top_k_retrieval, top_k_rerank)
        return Evaluator(predictions, relevant_cids).evaluate(metrics)
