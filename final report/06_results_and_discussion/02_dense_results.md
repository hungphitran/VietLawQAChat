# Dense Retrieval Results

The dense results show that zero-shot semantic retrieval already improves over the BM25 indexing baseline. The best dense model, `embeddinggemma-300m`, reaches 0.4730 MRR@10 and 0.9169 Success@100. This is higher than the best sparse MRR score of 0.3981 and the best sparse Success@100 score of 0.8526.

The model ranking also gives useful engineering guidance. `vietnamese-bi-encoder` is close to the best model in MRR@10, with 0.4615, but its Success@100 is lower than `embeddinggemma-300m`. `multilingual-e5-base` is stable but weaker, while `granite-embedding-97m` is lightweight but not competitive on this dataset. The result suggests that model selection must be measured on the target legal corpus rather than assumed from model size or language label.

The remaining gap is top-rank quality. Even the best dense model has Success@100 above 0.91 but MRR@10 below 0.50. This means the relevant document is often inside the top 100 but not high enough in the list. This is exactly the condition where cross-encoder reranking should be useful.
