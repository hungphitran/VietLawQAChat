# Contributions

This project makes three completed contributions and one system contribution in progress. First, it builds a clean Vietnamese legal retrieval framework with sparse, dense, hybrid, reranking, training, and negative-mining modules. All retrievers follow the same `index(documents, cids)` and `retrieve(queries, top_k)` interface, which keeps the system easy to extend.

Second, it provides a real BM25 indexing baseline on a 262K-passage legal corpus. The sparse experiments compare TF-IDF, BM25, and BM25+ with segmentation choices and hyperparameter tuning. The best sparse configuration reaches 0.3981 MRR@10 for ranking and 0.8526 Success@100 for recall.

Third, it evaluates zero-shot dense retrievers on the same full corpus and evaluation split. The best dense model, `embeddinggemma-300m`, reaches 0.4730 MRR@10 and 0.9169 Success@100. Fourth, the project connects this retrieval core to a full legal QA application with reranking, GraphRAG, and LLM response generation as active system components.
