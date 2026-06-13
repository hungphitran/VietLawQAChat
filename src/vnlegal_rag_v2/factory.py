from __future__ import annotations

from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
from vnlegal_rag_v2.models.cross_encoder import CrossEncoderModel
from vnlegal_rag_v2.pipeline import RAGPipeline
from vnlegal_rag_v2.reranking.cross_encoder import CrossEncoderReranker
from vnlegal_rag_v2.retrieval.dense import DenseRetriever
from vnlegal_rag_v2.retrieval.fusion import rrf
from vnlegal_rag_v2.retrieval.hybrid import HybridRetriever
from vnlegal_rag_v2.retrieval.sparse import BM25Retriever, TFIDFRetriever
from vnlegal_rag_v2.utils.text import SegmentationMethod

FUSION_REGISTRY = {
    "rrf": rrf,
}


def _seg(value: str | None) -> SegmentationMethod:
    if value is None or value in ("none", "None", "null", ""):
        return None
    if value in ("pyvi", "underthesea"):
        return value
    raise ValueError(f"Unsupported segmentation: {value!r}")


def _build_retriever(config: dict):
    method = config["method"]
    params = config.get("params", {})

    if method == "bm25":
        params_copy = {k: v for k, v in params.items() if k not in ("segmentation", "variant")}
        return BM25Retriever(
            segmentation_method=_seg(params.get("segmentation")),
            variant=params.get("variant", "bm25"),
            **params_copy,
        )

    if method == "tfidf":
        params_copy = {k: v for k, v in params.items() if k != "segmentation"}
        return TFIDFRetriever(
            segmentation_method=_seg(params.get("segmentation")),
            **params_copy,
        )

    if method == "dense":
        model = BiEncoderModel(
            model_name_or_path=params["model_name"],
            device=params.get("device"),
            max_length=params.get("max_length"),
            trust_remote_code=params.get("trust_remote_code", True),
        )
        return DenseRetriever(
            model=model,
            segmentation_method=_seg(params.get("segmentation")),
            encode_kwargs=params.get("encode_kwargs"),
        )

    if method == "hybrid":
        sub_retrievers = [_build_retriever(sub) for sub in params["retrievers"]]
        fusion_name = params.get("fusion", "rrf")
        return HybridRetriever(
            retrievers=sub_retrievers,
            weights=params.get("weights"),
            fusion=FUSION_REGISTRY[fusion_name],
            fusion_kwargs=params.get("fusion_kwargs"),
        )

    raise ValueError(f"Unknown retrieval method: {method}")


def _build_rerankers(config: dict | None) -> list | None:
    if config is None:
        return None

    method = config["method"]
    params = config.get("params", {})

    if method == "cross_encoder":
        model = CrossEncoderModel(
            model_name_or_path=params["model_name"],
            max_length=params.get("max_length", 256),
            device=params.get("device"),
        )
        return [CrossEncoderReranker(model=model, batch_size=params.get("batch_size", 16))]

    raise ValueError(f"Unknown reranking method: {method}")


def build_pipeline(config: dict) -> RAGPipeline:
    """Build a RAGPipeline from a pipeline config dict."""
    retrieval_config = config["retrieval"]
    reranking_config = config.get("reranking")

    retriever = _build_retriever(retrieval_config)
    rerankers = _build_rerankers(reranking_config)

    return RAGPipeline(
        retriever=retriever,
        rerankers=rerankers,
        top_k_retrieval=retrieval_config.get("top_k", 100),
        top_k_rerank=reranking_config.get("top_k", 10) if reranking_config else 10,
    )
