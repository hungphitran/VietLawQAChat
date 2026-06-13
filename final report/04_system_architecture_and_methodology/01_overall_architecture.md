# Overall Architecture

The system is organized as a modular RAG pipeline with retrieval as the core component. A user query is first processed and sent to one or more retrievers. The sparse retriever uses BM25-style inverted indexing, while the dense retriever uses bi-encoder embeddings and FAISS search. The pipeline can then combine rankings through reciprocal rank fusion and send the top candidates to a cross-encoder reranker.

The full application extends this retrieval pipeline into legal QA. After reranking, the system can expand evidence through GraphRAG and pass selected passages to an LLM for grounded answer generation. This structure follows the standard RAG idea of separating indexing, retrieval, augmentation, and generation, while keeping each module testable inside the codebase.

The current report separates completed retrieval experiments from system components that are still being integrated. Sparse and dense retrieval results are reported quantitatively. Hybrid retrieval, reranking, GraphRAG, and LLM answer generation are described as implemented or in-scope system modules, but they should only receive final result tables after their experiments are complete.
