import ast
import json
import os

import pandas as pd

from vnlegal_rag_v2.evaluation.metrics import (
    MetricFn,
    mrr_at_k,
    success_at_k,
)


class Evaluator:
    def __init__(
        self,
        predictions: list[list[int]],
        relevant_cids: list[list[int]],
    ):
        assert len(predictions) == len(relevant_cids)
        self.predictions = predictions
        self.relevant_cids = relevant_cids

    def evaluate(
        self,
        metrics: list[tuple[str, MetricFn, int]] | None = None,
    ) -> dict[str, float]:
        if metrics is None:
            metrics = [
                ("mrr@10", mrr_at_k, 10),
                ("success@10", success_at_k, 10),
            ]

        return {
            name: fn(self.predictions, self.relevant_cids, k) for name, fn, k in metrics
        }

    @staticmethod
    def save_results(
        results: dict[str, float],
        output_path: str,
    ) -> None:
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(results)
            results = existing

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_files(
        cls,
        pred_path: str,
        data_path: str,
        cid_column: str = "relevant_cids",
    ) -> "Evaluator":
        predictions = _load_predictions(pred_path)
        relevant_cids = _load_relevant_cids(data_path, cid_column)
        return cls(predictions, relevant_cids)


def _load_predictions(path: str) -> list[list[int]]:
    predictions = []
    with open(path, "r") as f:
        for line in f:
            predictions.append([int(x) for x in line.strip().split()])
    return predictions


def _load_relevant_cids(
    path: str,
    column: str = "relevant_cids",
) -> list[list[int]]:
    df = pd.read_csv(path, encoding="utf-8")
    return df[column].apply(ast.literal_eval).tolist()
