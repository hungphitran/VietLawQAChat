# Baseline Experiments

The main baseline follows the classical information retrieval pipeline shown in Figure 1: documents are indexed into an inverted index, the query is processed into a query representation, and search returns ranked documents. In this project, BM25 indexing is the central baseline because it is strong, fast, and suitable for exact legal terms. TF-IDF and BM25+ are included to measure how much term saturation and length normalization matter.

The sparse grid search evaluates TF-IDF, BM25, and BM25+ with non-segmented and Pyvi-segmented text. BM25 and BM25+ are tuned over `k1` and `b`, while BM25+ also includes `delta`. The evaluation uses MRR@10 for top-rank quality and Success@100 for candidate recall.

| Method | Best configuration | MRR@10 | Success@100 |
|---|---|---:|---:|
| TF-IDF | Pyvi | 0.2602 | 0.7621 |
| BM25 | Pyvi, k1=1.0, b=0.9 | 0.3928 | 0.8470 |
| BM25+ | Pyvi, k1=1.0, b=0.9 | 0.3981 | 0.8467 |
| BM25+ | Pyvi, k1=1.8, b=0.9 | 0.3922 | 0.8526 |

The results show that BM25 indexing is a much stronger baseline than TF-IDF. Pyvi segmentation also improves the sparse baseline: for BM25, Success@100 increases from 0.8108 without segmentation to 0.8494 with Pyvi. This confirms that Vietnamese word segmentation is not a small preprocessing detail; it changes retrieval quality.
