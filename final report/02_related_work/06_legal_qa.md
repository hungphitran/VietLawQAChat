# Legal QA and Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a language model with an external retrieval system. Lewis et al. introduced RAG as a way to combine parametric memory in a language model with non-parametric memory from a retrieved document index [Lewis2020]. Later surveys describe modern RAG systems as modular pipelines that include indexing, retrieval, augmentation, and generation [Gao2023].

This project follows that pipeline view. The current implemented core focuses on indexing, retrieval, reranking, and evaluation, while the full application extends these modules into answer generation. This design is especially suitable for legal QA because a legal answer should be grounded in retrieved evidence rather than produced only from the language model's internal knowledge.

The main risk in legal RAG is unsupported generation. A fluent answer is not enough if the answer cannot be traced to a legal passage. For that reason, this report treats retrieval quality as the foundation of the full QA system: weak retrieval creates weak evidence, and weak evidence makes grounded generation unreliable.
