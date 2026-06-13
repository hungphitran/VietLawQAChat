# Research Objectives

The first objective is to build a reproducible retrieval benchmark for Vietnamese legal documents. The benchmark compares TF-IDF, BM25, BM25+, and dense bi-encoder retrieval on the same corpus and evaluation split.

The second objective is to identify a strong baseline and a strong dense retriever for the later RAG system. The current evidence shows that BM25+ with Pyvi is the best sparse baseline, while `embeddinggemma-300m` is the best zero-shot dense retriever among the tested models.

The third objective is to connect retrieval experiments with a full legal QA system. The codebase already supports modular retrieval, hybrid fusion, reranking, training loops, and negative mining. GraphRAG, custom loss design, and LLM answer generation are included in the project scope, but their final quantitative evaluation should be reported only after integration is complete.
