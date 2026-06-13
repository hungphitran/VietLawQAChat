# Hybrid Retrieval

Hybrid retrieval combines sparse and dense rankings before the reranking stage. The main reason for this design is that sparse retrieval and dense retrieval fail in different ways: sparse models are strong on exact matching, while dense models are strong on semantic matching. Fusion allows the system to keep candidates from both sides.

In this project, the hybrid retriever is implemented as a modular wrapper over multiple retrievers, followed by rank fusion. This makes the design easy to extend and easy to compare against single-retriever baselines. In practice, the hybrid stage is most useful when the query is ambiguous or when the relevant passage uses a different surface form from the query.
