"""Tune hybrid retrieval (BM25 + Dense -> RRF -> Cross-Encoder rerank).

Config-driven sweep over all components, with two cost-saving properties:
  - Index ONCE: each dense model + each BM25(k1,b) is built & cached once;
    the fusion grid re-fuses cached rankings (instant).
  - Auto batch size: encode/rerank batch sizes come from the VRAM-aware
    `vnlegal_rag_v2.utils.batch_size` (same source as update_batch_size.py).

Phase 1 (retrieval) logs all fusion combos. Phase 2 (rerank) reranks the
top-N retrieval configs with each cross model. All combos + full params
are logged to scores.json. Re-runs load everything from cache.

Usage:
    python scripts/tune_hybrid.py configs/hybrid-tuning/hybrid_sweep.yaml
    python scripts/tune_hybrid.py configs/hybrid-tuning/hybrid_sweep.yaml --vram 16
"""

import argparse
import itertools
import json
import logging
import os
import platform
import warnings

# Quiet the noise (FutureWarnings from torch/pandas/bm25s, transformers logs,
# HF download bars). Must run BEFORE importing sentence-transformers/torch.
# On macOS, cap OpenMP threads to avoid a flaky segfault when torch (MPS) and
# scikit-learn share libomp in one process. Linux/CUDA keeps full threading.
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
import transformers
import yaml

# Must run AFTER importing transformers — the env var alone doesn't catch the
# per-batch "overflowing tokens ... longest_first truncation" reranker warning
# that otherwise floods the log thousands of times. Same fix as training code.
transformers.logging.set_verbosity_error()

from vnlegal_rag_v2.data.loaders import extract_queries
from vnlegal_rag_v2.evaluation.evaluator import Evaluator
from vnlegal_rag_v2.evaluation.metrics import (
    f1_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    success_at_k,
)
from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
from vnlegal_rag_v2.models.cross_encoder import CrossEncoderModel
from vnlegal_rag_v2.reranking.cross_encoder import CrossEncoderReranker
from vnlegal_rag_v2.retrieval.dense import DenseRetriever
from vnlegal_rag_v2.retrieval.fusion import rrf
from vnlegal_rag_v2.retrieval.sparse import BM25Retriever
from vnlegal_rag_v2.utils.batch_size import compute_batch_sizes, detect_vram, resolve_model

METRIC_REGISTRY = {
    "mrr_at_k": mrr_at_k,
    "success_at_k": success_at_k,
    "recall_at_k": recall_at_k,
    "ndcg_at_k": ndcg_at_k,
    "precision_at_k": precision_at_k,
    "f1_at_k": f1_at_k,
}


def _metrics(cfg: list[dict]) -> list[tuple[str, object, int]]:
    return [(m["name"], METRIC_REGISTRY[m["fn"]], m["k"]) for m in cfg]


def _grid(v) -> list:
    """Scalar → [v]; list/tuple of non-dicts → grid axis."""
    if isinstance(v, (list, tuple)) and (len(v) == 0 or not isinstance(v[0], dict)):
        return list(v)
    return [v]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="Path to hybrid tuning YAML config")
    ap.add_argument("--vram", type=str, default="auto", help="VRAM in GB, or 'auto' (default)")
    ap.add_argument("--no-cache-rerank", action="store_true",
                    help="Always recompute Phase 2 rerank (ignore stale rerank caches)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── batch sizes (auto from VRAM, VRAM-aware) ─────────────────────
    vram = detect_vram() if args.vram == "auto" else int(args.vram)
    sizes = compute_batch_sizes(vram)
    dense_models = cfg["models"]["dense"]
    cross_models = cfg["models"]["cross"]
    dense_bs = sizes[resolve_model(next(iter(dense_models.values())))]["encode"]
    cross_bs = sizes[resolve_model(next(iter(cross_models.values())))]["rerank"]
    print(f"VRAM={vram}GB → encode bs={dense_bs}, rerank bs={cross_bs}\n")

    top_k_r = cfg["retrieval"]["top_k"]
    out = cfg["output"]
    cache_dir, scores_path = out["cache_dir"], out["scores"]
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)

    # ── data (pre-segmented → segmentation_method=None) ──────────────
    data_cfg = cfg["data"]
    corpus_df = pd.read_csv(data_cfg["corpus_path"])
    documents, cids = corpus_df["text"].tolist(), corpus_df["cid"].tolist()
    eval_path = os.path.join(data_cfg["data_path"], data_cfg.get("eval_file", "eval.csv"))
    queries, relevant_cids = extract_queries(pd.read_csv(eval_path))
    n = len(queries)
    print(f"corpus: {len(documents):,} docs | queries: {n:,}\n")

    metrics_r = _metrics(cfg["evaluation"]["metrics"])
    primary = metrics_r[0][0]
    # selection metric: which Phase-1 config(s) get reranked (default = primary)
    sel_cfg = cfg["evaluation"].get("selection_metric")
    if sel_cfg:
        sel_metric = (sel_cfg["name"], METRIC_REGISTRY[sel_cfg["fn"]], sel_cfg["k"])
        sel_name = sel_cfg["name"]
    else:
        sel_metric, sel_name = metrics_r[0], primary
    results: dict[str, dict] = {}

    def log(name, scores, stage, params):
        results[name] = {**scores, "stage": stage, "params": params}
        json.dump(results, open(scores_path, "w"), ensure_ascii=False, indent=2)

    # ── Phase 1: build indexes once, cache rankings ──────────────────
    dense_keys = _grid(cfg["retrieval"]["dense"])
    dense_rankings: dict[str, list[list[int]]] = {}
    for dkey in dense_keys:
        cache = os.path.join(cache_dir, f"dense_{dkey}_top{top_k_r}.json")
        if os.path.exists(cache):
            dense_rankings[dkey] = json.load(open(cache))
            print(f"dense[{dkey}]: cached")
        else:
            print(f"dense[{dkey}]: encoding corpus…")
            m = BiEncoderModel(model_name_or_path=dense_models[dkey])
            dr = DenseRetriever(model=m, segmentation_method=None)
            dr.index(documents, cids)
            dense_rankings[dkey] = dr.retrieve(queries, top_k_r)
            json.dump(dense_rankings[dkey], open(cache, "w"))
            print(f"dense[{dkey}]: built + cached")

    bm = cfg["retrieval"]["bm25"]
    bm_variant = bm.get("variant", "bm25")
    bm_delta = bm.get("delta")  # only meaningful for the bm25+ variant
    bm25_rankings: dict[tuple, list[list[int]]] = {}
    for k1, b in itertools.product(_grid(bm["k1"]), _grid(bm["b"])):
        tag = f"{bm_variant}_k1={k1}_b={b}" + (f"_d={bm_delta}" if bm_delta is not None else "")
        cache = os.path.join(cache_dir, f"bm25_{tag}_top{top_k_r}.json")
        if os.path.exists(cache):
            bm25_rankings[(k1, b)] = json.load(open(cache))
            print(f"bm25 {tag}: cached")
        else:
            bm25_kwargs = {"k1": k1, "b": b}
            if bm_delta is not None:
                bm25_kwargs["delta"] = bm_delta
            r = BM25Retriever(segmentation_method=None, variant=bm_variant, **bm25_kwargs)
            r.index(documents, cids)
            bm25_rankings[(k1, b)] = r.retrieve(queries, top_k_r)
            json.dump(bm25_rankings[(k1, b)], open(cache, "w"))
            print(f"bm25 {tag}: built + cached")

    # ── Phase 1: fusion sweep (instant — re-fuses cached rankings) ────
    weights = _grid(cfg["retrieval"]["w_bm25"])
    fusion_ks = _grid(cfg["retrieval"]["fusion_k"])
    for dkey, k1, b, w, fk in itertools.product(dense_keys, _grid(bm["k1"]), _grid(bm["b"]), weights, fusion_ks):
        dn, bm_r = dense_rankings[dkey], bm25_rankings[(k1, b)]
        fused = [rrf([bm_r[i], dn[i]], [w, round(1 - w, 2)], k=fk)[:top_k_r] for i in range(n)]
        sc = Evaluator(fused, relevant_cids).evaluate(metrics_r + [sel_metric])
        log(f"retr_{dkey}_k1={k1}_b={b}_w={w}_fk={fk}", sc, "retrieval",
            {"dense": dkey, "k1": k1, "b": b, "w_bm25": w, "w_dense": round(1 - w, 2),
             "fusion_k": fk, "top_k": top_k_r})
    retr_n = sum(1 for v in results.values() if v["stage"] == "retrieval")
    print(f"\nPhase 1: {retr_n} retrieval combos logged\n")

    # ── Phase 2: rerank sweep ────────────────────────────────────────
    rer = cfg.get("rerank")
    if not rer:
        _print_summary(results, primary, sel_name)
        return

    top_k_x = rer["top_k"]
    cross_keys = _grid(rer["cross"])

    # Candidate sets: auto top-N retrieval configs (ranked by selection metric) + overrides
    retr_results = sorted(
        [(v["params"], results[k][sel_name]) for k, v in results.items() if v["stage"] == "retrieval"],
        key=lambda kv: kv[1], reverse=True,
    )
    cands = []
    for params, _ in retr_results[: rer.get("auto_top_n", 3)]:
        cands.append({**params, "label": f"auto-{params['dense']}-w={params['w_bm25']}-fk={params['fusion_k']}"})
    for cs in rer.get("candidate_sets", []) or []:
        cands.append(cs)
    print(f"\nPhase 2: reranking top-{len(cands)} config(s) by {sel_name}:")
    for c in cands:
        print(f"  → {c['label']}")

    # Precompute candidate ranking lists (dedup by params)
    cand_cache: dict[str, list[list[int]]] = {}

    def get_cands(cs):
        key = f"{cs['dense']}_{cs['k1']}_{cs['b']}_{cs['w_bm25']}_{cs['fusion_k']}"
        if key in cand_cache:
            return cand_cache[key]
        dkey, k1, b, w, fk = cs["dense"], cs["k1"], cs["b"], cs["w_bm25"], cs["fusion_k"]
        if dkey is None or w == 1.0:
            out = bm25_rankings[(k1, b)]
        elif w == 0.0:
            out = dense_rankings[dkey]
        else:
            bm_r, dn = bm25_rankings[(k1, b)], dense_rankings[dkey]
            out = [rrf([bm_r[i], dn[i]], [w, round(1 - w, 2)], k=fk)[:top_k_r] for i in range(n)]
        cand_cache[key] = out
        return out

    metrics_x = metrics_r  # rerank reuses the same metric set
    for cs in cands:
        cands_list = get_cands(cs)
        for ckey in cross_keys:
            cache = os.path.join(cache_dir, f"rerank_{cs['label']}_{ckey}.json")
            if os.path.exists(cache) and not args.no_cache_rerank:
                reranked = json.load(open(cache))
                print(f"rerank {cs['label']} × {ckey}: cached")
            else:
                print(f"rerank {cs['label']} × {ckey}: running…")
                cross = CrossEncoderModel(model_name_or_path=cross_models[ckey], max_length=256)
                rr = CrossEncoderReranker(model=cross, batch_size=cross_bs)
                reranked = rr.rerank(queries, cands_list, documents, cids, top_k=top_k_x)
                json.dump(reranked, open(cache, "w"))
                print(f"rerank {cs['label']} × {ckey}: done + cached")
            sc = Evaluator(reranked, relevant_cids).evaluate(metrics_x)
            log(f"rerank_{cs['label']}_{ckey}", sc, "rerank",
                {"cand": cs["label"], "cross": ckey, **{k: cs[k] for k in ("dense", "k1", "b", "w_bm25", "fusion_k")},
                 "top_k_rerank": top_k_x})

    _print_summary(results, primary, sel_name)


def _print_summary(results: dict, primary: str, sel_name: str | None = None) -> None:
    sk = sel_name or primary
    retr = sorted([(k, v) for k, v in results.items() if v["stage"] == "retrieval"],
                  key=lambda kv: kv[1].get(sk, kv[1][primary]), reverse=True)
    rer = sorted([(k, v) for k, v in results.items() if v["stage"] == "rerank"],
                 key=lambda kv: kv[1][primary], reverse=True)

    cols = [sk] + ([primary] if primary != sk else [])
    print(f"\n=== Retrieval — top 15 by {sk} ===")
    print(f"{'combo':<46} " + " ".join(f"{m:>10}" for m in cols))
    print("-" * (48 + 11 * len(cols)))
    for k, v in retr[:15]:
        print(f"{k:<46} " + " ".join(f"{v[c]:>10.4f}" for c in cols))

    if rer:
        print(f"\n=== Rerank — by {primary} ===")
        print(f"{'combo':<40} {primary:>10}")
        print("-" * 52)
        for k, v in rer:
            print(f"{k:<40} {v[primary]:>10.4f}")
        bk, bv = rer[0]
        print(f"\nBEST END-TO-END: {bk}  ({primary}={bv[primary]:.4f})")
        print(f"  {json.dumps(bv['params'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
