### 3.3 Data Quality

We conducted quality checks on the raw data. Both the corpus and the training set contain **zero null rows**. However, both contain duplicates.

#### 3.3.1 Corpus Duplicates

The corpus contains **465 rows** that share a `cid` with at least one other row (i.e., the same cid maps to different text). These appear in the updated portion of the corpus (rows with index > 261,000), suggesting they were introduced during the corpus update.

#### 3.3.2 Training Set Duplicates

The training set contains **3,988 rows** that share the same `question` text with at least one other row. These duplicates always share the same `context` and `cid` values but have different `qid` values, indicating that the same question was assigned multiple query IDs. The most frequently duplicated question appears 25 times.

Table 3.1: Data quality summary (raw files)

| Issue | Corpus | Training |
|---|---:|---:|
| Total rows | 262,168 | 119,456 |
| Null rows | 0 | 0 |
| Duplicate cid rows | 465 | — |
| Duplicate question rows | — | 3,988 |

#### 3.3.3 Long Documents

The corpus has a strongly right-skewed length distribution. Figure 3.2 (Section 3.4) shows the full distribution. The longest passage contains 55,368 tokens (approximately an entire amended legal article), while some passages contain as few as 1 token (degenerate cases like `"..."`). This motivates careful handling of truncation in dense models and explains why BM25 length normalization matters.
