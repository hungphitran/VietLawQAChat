### 3.1 Dataset Overview

The dataset was provided by the organizers of the **SoICT Hackathon 2024 — Legal Document Retrieval** track. The competition task is passage-level retrieval: given a Vietnamese legal question, rank corpus passages by relevance, measured by MRR@10.

The organizers distributed the data as split 7z archives (`dataset_part_aa` through `dataset_part_ae`) with a README containing only extraction instructions. The only documentation about the data itself is a short `readme.txt` inside the `Train/` directory that describes four fields across three files (see Section 3.2). No data card, data paper, collection methodology, or corpus provenance was provided.

The dataset contains two components:

| Component | Rows | Source file |
|---|---:|---|
| Corpus | 262,168 passages | `updated_corpus.csv` |
| Training set | 119,456 rows | `Train/train.csv` |

The corpus contains Vietnamese legal passages pre-segmented by the organizers. Based on content inspection, the passages appear to originate from Vietnamese legal documents (Luật, Nghị định, Thông tư, etc.), but the organizers did not disclose the specific source database or segmentation method.

Two observable properties shape the method design. First, questions are short (typically one sentence) while corpus passages vary widely in length — from 1 token to over 55,000 tokens (see Figure 3.2). Second, most questions have one relevant passage, but some require multiple (see Section 3.4). These properties make both top-rank precision and recall at larger depth important.
