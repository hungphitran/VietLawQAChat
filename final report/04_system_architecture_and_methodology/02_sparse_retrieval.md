# Sparse Retrieval

Sparse retrieval is the baseline module of the project. The corpus is tokenized, indexed, and searched through TF-IDF, BM25, or BM25+. For BM25-based retrieval, the implementation uses `bm25s` and supports both the Robertson BM25 variant and BM25+. For TF-IDF, the implementation uses `TfidfVectorizer` and cosine similarity.

The sparse module also tests Vietnamese segmentation as part of the retrieval method. The same retrieval model can run with raw text, Pyvi segmentation, or Underthesea segmentation. In the reported baseline experiments, Pyvi gives the strongest sparse results, which shows that tokenization choices are central to Vietnamese legal search.
