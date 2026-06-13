# Discussion

The results suggest a clear trade-off between simplicity and quality. Sparse retrieval is fast and interpretable, dense retrieval is stronger on semantic matching, hybrid retrieval aims to combine the strengths of both, and reranking improves the top of the list at a higher computational cost. This makes the architecture easy to justify from a system design point of view.

The discussion should also connect the retrieval results to the final application. A better retriever is not only a better benchmark score; it is also a better evidence source for GraphRAG and LLM response generation. This is the main reason the project treats retrieval as the foundation of the whole system rather than as an isolated benchmark task.
