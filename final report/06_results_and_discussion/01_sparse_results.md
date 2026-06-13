# Sparse Retrieval Results

The sparse results show that BM25 indexing is the correct baseline for this project. TF-IDF with Pyvi reaches only 0.2602 MRR@10 and 0.7621 Success@100, while BM25 with Pyvi reaches 0.3928 MRR@10 and 0.8470 Success@100. This is a large improvement from adding BM25 term saturation and document-length normalization.

The best sparse ranking score comes from BM25+ with Pyvi at `k1=1.0, b=0.9`, which reaches 0.3981 MRR@10. The best sparse recall score comes from BM25+ with Pyvi at `k1=1.8, b=0.9`, which reaches 0.8526 Success@100. This split is useful: lower `k1` favors top-rank quality, while higher `k1` improves recall at depth 100.

The strongest practical finding is that Pyvi segmentation helps sparse retrieval. For BM25, Success@100 improves from 0.8108 without segmentation to 0.8494 with Pyvi. This means Vietnamese tokenization should be treated as part of the retrieval method, not only as preprocessing.
