# Problem Motivation

Classical search engines remain a strong starting point for legal retrieval because they can index large document collections and match exact legal terms efficiently. In this project, BM25 indexing is therefore treated as the main baseline, following the standard pipeline of document indexing, query processing, search, and ranked output.

However, BM25 alone cannot solve all legal retrieval cases. A user may ask in everyday language while the relevant legal passage uses formal terminology. The available experiments support this limitation: the best sparse configuration reaches 0.3981 MRR@10 and 0.8526 Success@100, while the best zero-shot dense retriever reaches 0.4730 MRR@10 and 0.9169 Success@100.

These results motivate a modular system rather than a single retriever. BM25 gives a strong and interpretable baseline, dense retrieval improves semantic recall, reranking can improve the top-ranked results, and the final RAG application can use the selected evidence to generate grounded answers.
