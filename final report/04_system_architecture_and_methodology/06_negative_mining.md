# Negative Mining

Negative mining is used to build harder training examples for the retriever and reranker. The project supports easy negatives, hard negatives, moderate negatives, and semi-hard negatives. Easy negatives are sampled randomly, hard negatives are taken from high-ranked non-relevant results, and semi-hard negatives are selected from the region just after the last relevant prediction.

This design is useful because retrieval training depends strongly on the difficulty of the negative set. If negatives are too easy, the model learns too little. If they are too hard or noisy, training can become unstable. Semi-hard mining provides a practical balance, which is why it is a central idea in the reference paper and also a meaningful part of this project.
