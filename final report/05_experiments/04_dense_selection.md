# Dense Model Selection

The dense selection experiment evaluates zero-shot bi-encoder retrieval on the full corpus. Documents are encoded into dense vectors, normalized, and searched with FAISS using inner-product similarity. The experiment uses `top_k=100`, the same evaluation split of 13,364 queries, and no task-specific fine-tuning.

| Model | MRR@10 | Success@100 |
|---|---:|---:|
| embeddinggemma-300m | 0.4730 | 0.9169 |
| vietnamese-bi-encoder | 0.4615 | 0.8806 |
| multilingual-e5-base | 0.4115 | 0.8760 |
| granite-embedding-97m | 0.4076 | 0.8383 |

The best zero-shot dense model is `embeddinggemma-300m`, with MRR@10 of 0.4730 and Success@100 of 0.9169. Compared with the best sparse ranking baseline, BM25+ with Pyvi, this is an 18.8% relative gain in MRR@10 and an 8.3% relative gain in Success@100. This result gives real evidence that semantic retrieval is useful for the Vietnamese legal corpus.
