# Scalability Notes

The system is designed with scalability in mind because legal corpora can grow quickly. Sparse retrieval is cheap to query, dense retrieval requires vector storage and encoding cost, reranking adds extra compute on top of the candidate set, and GraphRAG may increase complexity through graph traversal. A scalable design therefore needs careful control of retrieval depth and module boundaries.

These notes are important even if the current project is still small enough to run locally. They show that the architecture is not tied to one toy dataset, and they also explain why the system is split into independent modules rather than one large end-to-end model.
