# Dense Retrieval

Dense retrieval represents queries and documents as vectors and ranks documents by vector similarity. This design became important in open-domain QA because dual-encoder retrievers can precompute document embeddings and retrieve relevant passages efficiently at query time [Karpukhin2020]. Sentence embedding models such as SBERT also showed that semantic similarity search can be made practical by encoding sentences into fixed-size vectors [Reimers2019].

This project uses dense retrieval as the second retrieval family after BM25 indexing. The dense retriever encodes the legal corpus with a bi-encoder, normalizes embeddings, and searches them using inner-product similarity through FAISS. This makes dense retrieval suitable for semantic matching, especially when the query and the legal passage do not share many exact words.

Dense retrieval does not replace sparse retrieval in this project. It complements it. Legal retrieval still benefits from exact terms and references, while dense models help with paraphrase and semantic similarity. This is why the project evaluates both single dense retrievers and later hybrid retrieval.
