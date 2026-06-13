# Re-ranking

Reranking is performed by a cross-encoder that scores a query-document pair jointly. The retriever first produces a candidate set, and the reranker then reorders only the top candidates. This keeps the cost manageable while allowing the model to use richer interaction between the query and the text.

The main advantage of reranking is that it can improve the top of the list, which is the part that matters most for the user. In the legal setting, this is especially important because the best answer may be hidden among several closely related passages. The current implementation supports a two-stage retrieval pipeline, and the same design can be reused for both sparse and dense candidate sets.
