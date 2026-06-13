# Figures and Tables to Include

## Figure 1: BM25 indexing baseline

Use the provided lecture image or redraw it in the same structure:

Document collection -> Indexing -> Inverted index -> Search -> Ranked documents.

Query -> Query processing -> Query vector -> Search.

Caption suggestion:

`Figure 1. Classical sparse retrieval pipeline used as the baseline in this project. Documents are indexed into an inverted index, while each user query is processed into a query representation and matched against the index to return ranked documents.`

## Figure 2: Proposed retrieval and QA pipeline

Recommended structure:

User query -> preprocessing -> BM25/BM25+ retriever and dense retriever -> rank fusion -> cross-encoder reranker -> optional GraphRAG expansion -> LLM answer generation -> answer with cited evidence.

Caption suggestion:

`Figure 2. Modular legal QA pipeline. The implemented core covers sparse retrieval, dense retrieval, hybrid fusion, reranking, and evaluation. GraphRAG and answer generation are included in the full application scope and are integrated as downstream modules.`

## Table 1: Dataset summary

Use these values from `data/local-report.md`:

| Item | Value |
|---|---:|
| Corpus rows | 262,168 |
| Raw train rows | 119,456 |
| Eval queries | 13,364 |
| Duplicate corpus CIDs | 465 |
| Duplicate train questions | 3,988 |
| Median document length | 175 tokens |
| Mean document length | 240.3 tokens |
| Max document length | 55,368 tokens |

## Table 2: Sparse baseline results

Use these values from `results/baseline-selection/*scores.json`:

| Method | Best configuration | MRR@10 | Success@100 |
|---|---|---:|---:|
| TF-IDF | Pyvi | 0.2602 | 0.7621 |
| BM25 | Pyvi, k1=1.0, b=0.9 | 0.3928 | 0.8470 |
| BM25+ | Pyvi, k1=1.0, b=0.9 | 0.3981 | 0.8467 |
| BM25+ | Pyvi, k1=1.8, b=0.9 | 0.3922 | 0.8526 |

## Table 3: Dense zero-shot results

Use these values from `results/dense-selection/*scores.json`:

| Model | MRR@10 | Success@100 |
|---|---:|---:|
| embeddinggemma-300m | 0.4730 | 0.9169 |
| vietnamese-bi-encoder | 0.4615 | 0.8806 |
| multilingual-e5-base | 0.4115 | 0.8760 |
| granite-embedding-97m | 0.4076 | 0.8383 |

