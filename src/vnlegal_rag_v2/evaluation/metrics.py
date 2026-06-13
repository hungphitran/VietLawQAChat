import math
from typing import Callable, TypeAlias

MetricFn: TypeAlias = Callable[[list[list[int]], list[list[int]], int], float]


def _hit_count(pred: list[int], relevant: set[int]) -> int:
    return sum(1 for cid in pred if cid in relevant)


def success_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0
    for pred, rel in zip(predictions, relevant_cids):
        rel_set = set(rel)
        total += any(cid in rel_set for cid in pred[:k])

    return total / len(predictions)


def mrr_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0.0
    for pred, rel in zip(predictions, relevant_cids):
        rel_set = set(rel)
        for i, cid in enumerate(pred[:k]):
            if cid in rel_set:
                total += 1.0 / (i + 1)
                break

    return total / len(predictions)


def recall_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0.0
    for pred, rel in zip(predictions, relevant_cids):
        if not rel:
            continue
        hits = _hit_count(pred[:k], set(rel))
        total += hits / len(rel)

    return total / len(predictions)


def precision_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0.0
    for pred, rel in zip(predictions, relevant_cids):
        hits = _hit_count(pred[:k], set(rel))
        total += hits / k

    return total / len(predictions)


def f1_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0.0
    for pred, rel in zip(predictions, relevant_cids):
        if not rel:
            continue
        hits = _hit_count(pred[:k], set(rel))
        p = hits / k
        r = hits / len(rel)
        if p + r == 0:
            continue
        total += 2 * p * r / (p + r)

    return total / len(predictions)


def ndcg_at_k(
    predictions: list[list[int]],
    relevant_cids: list[list[int]],
    k: int = 10,
) -> float:
    assert len(predictions) == len(relevant_cids)

    total = 0.0
    for pred, rel in zip(predictions, relevant_cids):
        if not rel:
            continue
        rel_set = set(rel)
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, cid in enumerate(pred[:k])
            if cid in rel_set
        )
        idcg = sum(
            1.0 / math.log2(i + 2)
            for i in range(min(len(rel), k))
        )
        total += dcg / idcg if idcg > 0 else 0.0

    return total / len(predictions)
