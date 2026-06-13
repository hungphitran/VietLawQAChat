# Abstract

Vietnamese legal question answering requires a retrieval system that can find reliable legal evidence before generating an answer. This project studies legal document retrieval on a Vietnamese corpus of 262,168 passages and builds a modular pipeline for sparse retrieval, dense retrieval, reranking, and later RAG-based answer generation. The main challenge is that legal queries are often short, while legal passages are long, formal, and sensitive to Vietnamese word segmentation.

We first build a classical BM25 indexing baseline that follows the standard information retrieval pipeline: document indexing, query processing, search, and ranked output. We then compare TF-IDF, BM25, BM25+, and zero-shot dense retrievers under the same evaluation setting. On 13,364 evaluation queries, the best sparse ranking baseline is BM25+ with Pyvi segmentation, reaching 0.3981 MRR@10, while the best sparse recall setting reaches 0.8526 Success@100. The best zero-shot dense model, `embeddinggemma-300m`, improves the result to 0.4730 MRR@10 and 0.9169 Success@100.

These results show that BM25 indexing is a strong and interpretable baseline, but dense retrieval gives better semantic recall on Vietnamese legal text. The project also provides an extensible system design for hybrid retrieval, cross-encoder reranking, GraphRAG, and LLM-based answer generation. The current report focuses on the completed retrieval experiments and documents the full application modules as part of the ongoing system integration.

