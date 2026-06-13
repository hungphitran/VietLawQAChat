"""Run full pipeline: load data → index → retrieve → (rerank) → evaluate → log results."""

import argparse
import json
import os
import sys
from itertools import product

import pandas as pd
import yaml
from tqdm import tqdm

from vnlegal_rag_v2.data.loaders import extract_corpus, extract_queries, load_processed
from vnlegal_rag_v2.evaluation.metrics import (
    MetricFn,
    f1_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    success_at_k,
)
from vnlegal_rag_v2.factory import build_pipeline

METRIC_REGISTRY = {
    "mrr_at_k": mrr_at_k,
    "success_at_k": success_at_k,
    "recall_at_k": recall_at_k,
    "precision_at_k": precision_at_k,
    "f1_at_k": f1_at_k,
}

ALL_METRICS = [
    {"name": "mrr@1", "fn": "mrr_at_k", "k": 1},
    {"name": "mrr@5", "fn": "mrr_at_k", "k": 5},
    {"name": "mrr@10", "fn": "mrr_at_k", "k": 10},
    {"name": "mrr@100", "fn": "mrr_at_k", "k": 100},
    {"name": "success@1", "fn": "success_at_k", "k": 1},
    {"name": "success@5", "fn": "success_at_k", "k": 5},
    {"name": "success@10", "fn": "success_at_k", "k": 10},
    {"name": "success@100", "fn": "success_at_k", "k": 100},
    {"name": "recall@10", "fn": "recall_at_k", "k": 10},
    {"name": "recall@50", "fn": "recall_at_k", "k": 50},
    {"name": "recall@100", "fn": "recall_at_k", "k": 100},
    {"name": "precision@10", "fn": "precision_at_k", "k": 10},
    {"name": "precision@100", "fn": "precision_at_k", "k": 100},
    {"name": "f1@10", "fn": "f1_at_k", "k": 10},
    {"name": "f1@100", "fn": "f1_at_k", "k": 100},
]

DEFAULT_METRICS = [
    {"name": "mrr@10", "fn": "mrr_at_k", "k": 10},
    {"name": "success@10", "fn": "success_at_k", "k": 10},
    {"name": "recall@100", "fn": "recall_at_k", "k": 100},
]


def _resolve_corpus_path(data_path: str, segmentation) -> tuple[str | None, bool]:
    """Check for pre-segmented corpus file.
    Returns (path, is_presegmented)."""
    if segmentation:
        seg_corpus = os.path.join(data_path, f"corpus_{segmentation}.csv")
        if os.path.exists(seg_corpus):
            return seg_corpus, True
    base_corpus = os.path.join(data_path, "corpus.csv")
    if os.path.exists(base_corpus):
        return base_corpus, False
    return None, False


def _load_corpus(data_config: dict, segmentation=None) -> tuple[list[str], list[int]]:
    data_path = data_config.get("data_path", "")
    corpus_path, _ = _resolve_corpus_path(data_path, segmentation)

    if not corpus_path:
        corpus_path = data_config.get("corpus_path")

    if corpus_path:
        df = pd.read_csv(corpus_path, encoding="utf-8")
        if "text" in df.columns:
            return df["text"].tolist(), df["cid"].tolist()
        return extract_corpus(df)

    data_path = data_config["data_path"]
    corpus_from = data_config.get("corpus_from", "eval")

    if corpus_from == "eval":
        df = load_processed(data_path, data_config.get("eval_file", "eval.csv"))
    elif corpus_from == "train":
        df = load_processed(data_path, "train.csv")
    else:
        df = pd.read_csv(corpus_from, encoding="utf-8")

    return extract_corpus(df)


def _parse_metrics(metrics_config: list[dict] | str | None) -> list[tuple[str, MetricFn, int]]:
    if metrics_config is None:
        metrics_config = DEFAULT_METRICS
    elif metrics_config == "all":
        metrics_config = ALL_METRICS

    return [
        (m["name"], METRIC_REGISTRY[m["fn"]], m["k"])  # type: ignore[index]
        for m in metrics_config
    ]


def _save_predictions(predictions: list[list[int]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for cids in predictions:
            f.write(" ".join(map(str, cids)) + "\n")


def _accumulate_scores(name: str, scores: dict[str, float], path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            all_scores = json.load(f)
    else:
        all_scores = {}

    all_scores[name] = scores

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)


def _expand_grid(config: dict) -> list[tuple[str, dict]]:
    """Expand a config with list-valued retrieval params into individual configs."""
    params = config.get("retrieval", {}).get("params", {})
    list_keys = [k for k, v in params.items() if isinstance(v, (list, tuple, set, range))]

    if not list_keys:
        return [("", config)]

    # Normalize iterables to lists
    list_values = [list(params[k]) for k in list_keys]
    combos = list(product(*list_values))
    results = []
    for combo in combos:
        # Deep copy with normalized params
        p = {k: list(v) if isinstance(v, (list, tuple, set, range)) else v for k, v in params.items()}
        for k, v in zip(list_keys, combo):
            p[k] = v
        cfg = json.loads(json.dumps({**config, "retrieval": {**config["retrieval"], "params": p}}))
        for k, v in zip(list_keys, combo):
            p[k] = v
        cfg["retrieval"]["params"] = p
        suffix = "_" + "_".join(f"{k}={v}" for k, v in zip(list_keys, combo))
        results.append((suffix, cfg))
    return results


def run_single(config_path: str, results_path: str | None = None) -> dict[str, dict[str, float]]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_name = config.get("name", os.path.splitext(os.path.basename(config_path))[0])
    grid = _expand_grid(config)
    is_grid = len(grid) > 1

    if is_grid:
        print(f"\nGrid search: {len(grid)} combinations")

    # Load eval data once
    data_config = config["data"]
    data_path = data_config["data_path"]
    eval_seg = config.get("retrieval", {}).get("params", {}).get("segmentation")
    eval_suffix = f"_{eval_seg}" if eval_seg else ""
    eval_file = data_config.get("eval_file", f"eval{eval_suffix}.csv")

    eval_df = load_processed(data_path, eval_file)
    queries, relevant_cids = extract_queries(eval_df)
    metrics = _parse_metrics(config.get("evaluation", {}).get("metrics"))
    primary_metric = metrics[0][0] if metrics else None

    all_scores: dict[str, dict[str, float]] = {}
    best_score = -1.0
    best_name: str | None = None

    # Cache corpus across grid combos (same data, only BM25 params change)
    cached_docs: tuple[list[str], list[int]] | None = None
    cached_seg: str | None = ""  # sentinel so first combo always loads

    iterator = tqdm(grid, desc=base_name, unit="cfg") if is_grid else grid
    for suffix, cfg in iterator:
        name = base_name + suffix
        seg = cfg.get("retrieval", {}).get("params", {}).get("segmentation")

        # Auto-detect pre-segmented corpus → skip retriever segmentation
        _, is_preseg = _resolve_corpus_path(data_path, seg)
        if is_preseg:
            cfg = json.loads(json.dumps(cfg))  # copy
            cfg["retrieval"]["params"]["segmentation"] = None

        if seg != cached_seg:
            documents, cids = _load_corpus(data_config, seg)
            cached_docs = (documents, cids)
            cached_seg = seg

        assert cached_docs is not None
        documents, cids = cached_docs

        pipeline = build_pipeline(cfg)
        pipeline.index(documents, cids)
        scores = pipeline.evaluate(queries, relevant_cids, metrics=metrics)

        if is_grid:
            iterator.set_postfix({primary_metric: f"{scores.get(primary_metric, 0):.4f}"})  # type: ignore[union-attr]
        else:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            print(f"  Queries: {len(queries)}  |  Corpus: {len(documents)}")
            for m, v in scores.items():
                print(f"  {m}: {v:.4f}")

        output_config = cfg.get("output", {})
        pred_path = output_config.get("predictions")
        if pred_path and not is_grid:
            predictions = pipeline.query(queries)
            _save_predictions(predictions, pred_path)
            print(f"  Predictions → {pred_path}")

        scores_path = output_config.get("scores")
        if scores_path:
            _accumulate_scores(name, scores, scores_path)
            if not is_grid:
                print(f"  Scores → {scores_path}")

        if results_path:
            _accumulate_scores(name, scores, results_path)

        all_scores[name] = scores

        if primary_metric and scores.get(primary_metric, 0) > best_score:
            best_score = scores[primary_metric]
            best_name = name

    if is_grid and primary_metric and best_name:
        print(f"\n{'=' * 60}")
        print(f"Best {primary_metric}: {best_score:.4f} → {best_name}")
        print(f"{'=' * 60}")

    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Run pipeline from YAML config(s)")
    parser.add_argument("configs", nargs="+", help="Path(s) to pipeline YAML config(s)")
    parser.add_argument(
        "--results",
        type=str,
        default="results/scores.json",
        help="Shared results file for all experiments (default: results/scores.json)",
    )
    args = parser.parse_args()

    all_scores: dict[str, dict[str, float]] = {}
    for config_path in args.configs:
        scores = run_single(config_path, args.results)
        all_scores |= scores

    print(f"\n{'=' * 60}")
    print(f"All results accumulated → {args.results}")
    print(f"{'=' * 60}")

    # Print comparison table
    if len(all_scores) > 1:
        metric_names = list(next(iter(all_scores.values())).keys())
        header = f"{'experiment':<55}" + "".join(f"{m:<15}" for m in metric_names)
        print(header)
        print("-" * len(header))
        for name, scores in all_scores.items():
            row = f"{name:<55}" + "".join(f"{scores[m]:<15.4f}" for m in metric_names)
            print(row)

    return all_scores


if __name__ == "__main__":
    main()
