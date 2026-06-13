from __future__ import annotations

import random

import pandas as pd


class NegativeMiner:
    """Mine negative examples using a retriever to find negatives across difficulty spectrum."""

    def __init__(
        self,
        retriever,
        documents: list[str],
        cids: list[int],
    ):
        self.retriever = retriever
        self._documents = list(documents)
        self._cids = list(cids)
        self._cid_to_idx = {cid: i for i, cid in enumerate(cids)}
        self.retriever.index(documents, cids)

    def mine(
        self,
        queries: list[str],
        relevant_cids: list[list[int]],
        strategy: str = "hard",
        num_negatives: int = 3,
        top_k: int = 100,
        seed: int = 28,
        **kwargs,
    ) -> list[dict]:
        assert len(queries) == len(relevant_cids)
        assert strategy in ("easy", "moderate", "semi_hard", "hard")

        if strategy == "easy":
            neg_cids = self.random_negatives(
                self._cids, relevant_cids, num_negatives, seed,
            )
        elif strategy == "semi_hard":
            predictions = self.retriever.retrieve(queries, top_k)
            neg_cids = self.semi_hard_negatives(
                predictions, relevant_cids, num_negatives,
                margin=kwargs.get("margin", 10),
            )
        else:
            predictions = self.retriever.retrieve(queries, top_k)
            neg_cids = self.from_predictions(
                predictions, relevant_cids, num_negatives, strategy, seed,
            )

        results = []
        for query, neg_cid_list in zip(queries, neg_cids):
            for neg_cid in neg_cid_list:
                results.append({
                    "query": query,
                    "negative_text": self._documents[self._cid_to_idx[neg_cid]],
                    "negative_cid": neg_cid,
                })

        return results

    def mine_to_csv(
        self,
        queries: list[str],
        relevant_cids: list[list[int]],
        output_path: str,
        strategy: str = "hard",
        num_negatives: int = 3,
        top_k: int = 100,
        seed: int = 28,
        **kwargs,
    ) -> pd.DataFrame:
        results = self.mine(
            queries, relevant_cids, strategy, num_negatives, top_k, seed,
            **kwargs,
        )
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        return df

    @staticmethod
    def random_negatives(
        corpus_cids: list[int],
        relevant_cids: list[list[int]],
        num_negatives: int,
        seed: int = 28,
    ) -> list[list[int]]:
        rng = random.Random(seed)
        cid_set = set(corpus_cids)

        results = []
        for relevant in relevant_cids:
            candidates = [cid for cid in cid_set if cid not in relevant]
            results.append(rng.sample(candidates, min(num_negatives, len(candidates))))

        return results

    @staticmethod
    def from_predictions(
        pred_cids: list[list[int]],
        relevant_cids: list[list[int]],
        num_negatives: int,
        strategy: str = "hard",
        seed: int = 28,
    ) -> list[list[int]]:
        assert len(pred_cids) == len(relevant_cids)
        assert strategy in ("moderate", "hard")

        rng = random.Random(seed)

        results = []
        for preds, relevant in zip(pred_cids, relevant_cids):
            rel_set = set(relevant)
            non_relevant = [cid for cid in preds if cid not in rel_set]

            if strategy == "hard":
                results.append(non_relevant[:num_negatives])
            else:
                results.append(
                    rng.sample(non_relevant, min(num_negatives, len(non_relevant)))
                )

        return results

    @staticmethod
    def semi_hard_negatives(
        pred_cids: list[list[int]],
        relevant_cids: list[list[int]],
        num_negatives: int,
        margin: int = 10,
    ) -> list[list[int]]:
        """Negatives ranked just after the last relevant document in predictions."""
        assert len(pred_cids) == len(relevant_cids)

        results = []
        for preds, relevant in zip(pred_cids, relevant_cids):
            rel_set = set(relevant)

            last_relevant_rank = -1
            for rank, cid in enumerate(preds):
                if cid in rel_set:
                    last_relevant_rank = rank

            if last_relevant_rank == -1:
                # No relevant found in predictions — fall back to hard
                non_relevant = [cid for cid in preds if cid not in rel_set]
                results.append(non_relevant[:num_negatives])
            else:
                window_start = last_relevant_rank + 1
                window_end = min(window_start + margin, len(preds))
                window = [
                    preds[i] for i in range(window_start, window_end)
                    if preds[i] not in rel_set
                ]
                results.append(window[:num_negatives])

        return results
