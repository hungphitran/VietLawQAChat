# Hybrid and Re-ranking Experiments

Hybrid retrieval and reranking are the most important system-level experiments because they test whether the components actually complement each other. Sparse retrieval gives high lexical coverage, dense retrieval adds semantic matching, and reranking should improve the final order of the top candidates.

The experiment design should compare three settings at minimum: single retriever, hybrid retriever, and retriever plus reranker. This makes it possible to see whether each stage contributes measurable value, instead of assuming that a more complex pipeline is automatically better.
