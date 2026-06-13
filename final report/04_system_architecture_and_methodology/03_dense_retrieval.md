# Dense Retrieval

Dense retrieval uses a bi-encoder model to encode documents and queries into the same vector space. The implementation computes document embeddings, normalizes them with L2 normalization, and stores them in a FAISS `IndexFlatIP` index. At query time, the query embedding is normalized and searched against the same index to return the top document IDs.

The dense module is evaluated first in zero-shot mode, before task-specific fine-tuning. This gives a clean measurement of how much pretrained embedding models already know about Vietnamese legal retrieval. The strongest tested model is `embeddinggemma-300m`, which outperforms the best sparse baseline on both MRR@10 and Success@100.
