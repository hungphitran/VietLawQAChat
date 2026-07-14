from __future__ import annotations

import numpy as np
from tqdm import tqdm

from vnlegal_rag_v2.models.cross_encoder import CrossEncoderModel


class CrossEncoderReranker:
    def __init__(self, model: CrossEncoderModel, batch_size: int = 16):
        self.model = model
        self._batch_size = batch_size

    def rerank(
        self,
        queries: list[str],
        candidate_cids: list[list[int]],
        documents: list[str],
        cids: list[int],
        top_k: int = 100,
        **kwargs,
    ) -> list[list[int]]:
        """Score every (query, candidate-doc) pair with the cross-encoder, then return the
        top_k docs per query sorted by score.

        All pairs are flattened into one batched `predict()` call (GPU-efficient), then split
        back per query using the cumulative-boundary offsets.
        """
        assert len(queries) == len(candidate_cids)

        cid_to_idx = {cid: i for i, cid in enumerate(cids)}

        # Flatten all (query, doc) pairs
        all_pairs: list[list[str]] = []
        boundaries: list[int] = [0]
        for query, cand_cids in zip(queries, candidate_cids):
            for cid in cand_cids:
                all_pairs.append([query, documents[cid_to_idx[cid]]])
            boundaries.append(boundaries[-1] + len(cand_cids))

        # Batch predict
        all_scores = np.empty(len(all_pairs), dtype=np.float32)
        for start in tqdm(
            range(0, len(all_pairs), self._batch_size),
            desc="Reranking (Cross-Encoder)",
            total=(len(all_pairs) + self._batch_size - 1) // self._batch_size,
        ):
            end = min(start + self._batch_size, len(all_pairs))
            kwargs.setdefault("show_progress_bar", False)
            all_scores[start:end] = self.model.predict(
                all_pairs[start:end], **kwargs,
            )

        # Split scores back per query and rank
        results: list[list[int]] = []
        for i, cand_cids in enumerate(candidate_cids):
            scores = all_scores[boundaries[i]:boundaries[i + 1]]
            sorted_pairs = sorted(
                zip(cand_cids, scores), key=lambda x: x[1], reverse=True
            )
            results.append([cid for cid, _ in sorted_pairs[:top_k]])

        return results
