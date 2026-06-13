# GraphRAG

GraphRAG extends retrieval by representing knowledge as a graph rather than as isolated passages. In a legal setting, this is a natural idea because laws are connected through citations, amendments, shared concepts, and hierarchical structure. A graph can help the system move from one relevant document to related documents that are not easy to find through lexical matching alone.

The main value of GraphRAG is controlled expansion. Instead of trusting a single retrieved passage, the system can collect related nodes and then reason over a small legal subgraph. The main risk is graph noise: if the graph is built too loosely, expansion can add irrelevant nodes and weaken precision. For that reason, graph-based retrieval should be combined with a strong retriever and a clear ranking policy.
