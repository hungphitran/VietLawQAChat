# Re-ranking

Re-ranking is used when the first-stage retriever can find a useful candidate set but cannot always place the best document at the top. A cross-encoder reads the query and document together, so it can model detailed token-level interaction that a bi-encoder loses when it compresses each text into one vector. This usually improves top-rank precision but is much more expensive than first-stage retrieval.

This cost-quality trade-off is why re-ranking is used as a second stage rather than as the full search engine. Related neural ranking work such as ColBERT also studies this balance between expressive interaction and retrieval efficiency [Khattab2020]. In this project, the reranker is designed to operate on the top candidates returned by BM25, dense retrieval, or hybrid retrieval.
