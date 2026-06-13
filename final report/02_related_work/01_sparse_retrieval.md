# Sparse Retrieval

Sparse retrieval is the classical search setting where documents are indexed before query time and each query is matched against an inverted index. This project uses this setting as the baseline because it is efficient, transparent, and easy to reproduce. Following the standard information retrieval pipeline, the document collection is first tokenized and indexed, while the user query is processed into a comparable representation and searched against the index to produce ranked documents.

BM25 is the main sparse baseline in this project because it remains one of the strongest lexical ranking functions for text retrieval. BM25 improves over plain TF-IDF by using term-frequency saturation and document-length normalization, which are important for legal text because legal passages vary strongly in length [Robertson2009]. BM25+ is also tested because it modifies the BM25 score to reduce over-penalization of long documents.

Sparse retrieval is still valuable in legal search because many legal questions contain exact terms, article names, document identifiers, and formal legal phrases. These lexical signals are often precise and interpretable. However, sparse retrieval is limited when the user query uses general wording while the relevant legal passage uses formal wording. This limitation motivates dense retrieval and reranking as later stages.
