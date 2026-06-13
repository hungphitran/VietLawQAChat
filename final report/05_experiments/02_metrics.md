# Evaluation Metrics

The main evaluation metrics are MRR@10, Recall@100, and Success@k. MRR@10 measures how highly the first relevant document appears in the ranking, so it reflects ranking quality near the top. Recall@100 measures whether the system can keep at least one relevant document in the candidate set, which is important for later reranking or generation. Success@k provides a simpler view of whether a relevant result appears within the top-k list.

These metrics fit the project well because the system is not only a retrieval benchmark, but also a multi-stage QA pipeline. A good retriever must recover relevant evidence early enough for reranking, while the reranker must then move the best evidence toward the top of the list.
