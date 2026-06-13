### 3.6 Data Processing Pipeline

Our data processing pipeline transforms the raw competition files into the processed datasets used for training and evaluation. The pipeline has three steps:

#### 3.6.1 Step 1: QA Pair Expansion

The raw `train.csv` stores each question with a list of relevant CIDs (e.g., `[58264 58688]`). We expand each row into one training sample per CID by looking up the passage text in the corpus. Each resulting sample contains:

| Field | Source |
|---|---|
| `question` | `train.question` |
| `positive_text` | `corpus.text[cid]` |
| `positive_cid` | One element from `train.cid` |
| `relevant_cids` | Full list from `train.cid` (copied to all expanded rows) |

This means a question with 2 relevant CIDs produces 2 training rows, both carrying the full `relevant_cids` list. This preserves the multi-relevance information for evaluation while providing one positive example per row for contrastive training.

#### 3.6.2 Step 2: Train/Eval Split

The expanded QA pairs are split at the question level as described in Section 3.5.

#### 3.6.3 Step 3: Text Segmentation

Vietnamese is an isolating language without explicit word boundaries. Word segmentation can improve retrieval quality, particularly for sparse methods. We apply three segmentation settings and save each as a separate file variant:

| Setting | Tool | Output suffix |
|---|---|---|
| No segmentation (raw text) | — | (default) |
| Word segmentation | Pyvi | `_pyvi` |
| Word segmentation | UnderTheSea | `_underthesea` |

The same train/eval split is used for all segmentation variants. Segmentation is applied after splitting to avoid any data-dependent processing from affecting the split.
