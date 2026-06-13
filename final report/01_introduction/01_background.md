# Background

Vietnamese legal question answering depends on reliable document retrieval. Given a user question, the system must find legal passages that can support an answer, not only passages that share surface keywords with the question. This makes retrieval the first and most important stage of the full QA pipeline.

The task is difficult because Vietnamese legal data has both linguistic and structural challenges. Many user questions are short, while legal documents are long, formal, and full of domain-specific terms. The corpus used in this project has 262,168 legal passages, with a median length of 175 tokens but a maximum length of 55,368 tokens. This large length gap makes both sparse indexing and dense representation quality important.

This project studies retrieval as the foundation of a legal RAG system. The implemented experiments start from a BM25 inverted-index baseline, compare sparse retrieval variants, and then evaluate zero-shot dense retrievers on the same corpus. The full application extends this retrieval core toward reranking, GraphRAG, and grounded answer generation.
