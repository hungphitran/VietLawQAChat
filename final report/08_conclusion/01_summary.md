# Summary

This project studies Vietnamese legal document retrieval as the foundation of a broader legal QA system. It builds a modular pipeline that covers sparse retrieval, dense retrieval, hybrid fusion, reranking, and training support, while also preparing the path toward GraphRAG and grounded answer generation.

The main message of the project is that retrieval quality is the key to the whole system. A strong lexical baseline is useful, dense retrieval improves semantic matching, reranking improves the final order, and the application layer can then use the selected evidence to support user-facing answers.
