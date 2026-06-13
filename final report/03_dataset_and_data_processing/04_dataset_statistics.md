### 3.4 Dataset Statistics

All statistics below are measured by us from the raw data files. No official statistics were provided by the organizers.

#### 3.4.1 Corpus Length Distribution

We tokenize passages by whitespace split (approximate token count) following the same method used in `visualize.ipynb`.

Table 3.2: Corpus passage length statistics (whitespace-tokenized)

| Statistic | Tokens |
|---|---:|
| n | 262,168 |
| min | 1 |
| Q1 | 96 |
| median | 175 |
| Q3 | 307 |
| max | 55,368 |
| mean | 240.3 |
| IQR | 211 |
| outliers (1.5×IQR) | 13,458 |

The distribution is strongly right-skewed with a long tail of documents far exceeding typical encoder context windows (512 tokens). Figure 3.1 shows the histogram and boxplot.

**[Figure 3.1: Placeholder — `fig_corpus_length_histogram_and_boxplot`]**
*Histogram and boxplot of corpus passage token counts. Source: `data/visualize.ipynb`.*

Figure 3.2 shows the binned distribution.

**[Figure 3.2: Placeholder — `fig_corpus_length_binned`]**
*Corpus passage count by token bins (0–128, 128–256, 256–512, 512–1024, 1024+). Source: `data/visualize.ipynb`.*

#### 3.4.2 Training Set Statistics

We measure question length and the number of relevant passages per question from the raw `train.csv`.

**[Figure 3.3: Placeholder — `fig_train_question_length_and_cid_count`]**
*Left: histogram of question token count. Right: histogram of relevant CID count per question. Source: `data/visualize.ipynb`.*

#### 3.4.3 Processed Data Statistics

After our data processing pipeline (Section 3.6), the data is expanded and split as follows:

Table 3.3: Processed dataset sizes

| File | Rows | Description |
|---|---:|---|
| `train.csv` | 120,204 | Question–positive pairs (expanded from multi-CID rows) |
| `eval.csv` | 13,364 | Question–positive pairs (held-out split) |
| `corpus.csv` | 262,168 | Corpus passages (unchanged) |
| `train_pyvi.csv` | 120,204 | Pyvi-segmented training set |
| `eval_pyvi.csv` | 13,364 | Pyvi-segmented evaluation set |
| `corpus_pyvi.csv` | 262,168 | Pyvi-segmented corpus |
| `train_underthesea.csv` | — | Underthesea-segmented training set |
| `eval_underthesea.csv` | — | Underthesea-segmented evaluation set |
| `corpus_underthesea.csv` | — | Underthesea-segmented corpus |

The training set is larger than the raw file (120,204 vs 119,456) because multi-CID rows are expanded into one row per CID. For example, a raw row with `cid: [58264 58688]` produces two processed rows, one for each CID, each sharing the same `relevant_cids` list.

The split is performed at the **question level** (see Section 3.6), so the same question text never appears in both train and eval.
