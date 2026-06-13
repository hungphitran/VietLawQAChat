# Negative Mining

Negative mining is the process of selecting non-relevant examples for training a retriever or reranker. Easy negatives are sampled at random, hard negatives are taken from top-ranked but non-relevant results, and moderate negatives sit between these two extremes. Semi-hard negatives are a practical middle ground: they are difficult enough to teach the model, but not so difficult that training becomes unstable.

This idea is especially relevant for retrieval models because the choice of negatives shapes what the model learns to distinguish. If the negatives are too easy, the model learns little. If they are too hard or noisy, it may overfit to misleading examples. Semi-hard negative mining is therefore a useful strategy for legal retrieval, where many documents are topically similar but only a subset is truly relevant.
