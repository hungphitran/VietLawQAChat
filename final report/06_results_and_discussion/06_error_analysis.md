# Error Analysis

Error analysis is important because retrieval systems often fail in systematic ways rather than random ways. In this project, likely failure cases include ambiguous short queries, long documents that are truncated, passages with many near-duplicates, and legal references that require multi-hop reasoning. These cases are worth discussing even if the main metrics look strong.

The goal of this subsection is not to list every bug, but to identify patterns that explain where the system is weak. That makes the report more credible and also helps motivate the GraphRAG and QA extensions, since those modules are meant to address some of the harder retrieval and reasoning cases.
