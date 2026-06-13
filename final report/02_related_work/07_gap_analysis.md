# Gap Analysis

Prior work already shows that sparse retrieval is a strong baseline and that dense retrieval and reranking can improve ranking quality. However, the legal domain still has three open issues. First, Vietnamese legal text is sensitive to segmentation and vocabulary choices. Second, legal questions often need both exact matching and semantic matching. Third, a retrieval-only model is not enough for a full QA workflow that must return grounded answers.

This project addresses these gaps with a modular system that compares sparse, dense, hybrid, and reranking methods on the same dataset, while also preparing the path toward GraphRAG and a full answer generation layer. In this sense, the project is not only a reproduction of a retrieval paper, but also a broader system study that connects retrieval quality to the final user-facing application.
