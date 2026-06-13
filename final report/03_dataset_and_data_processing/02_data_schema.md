### 3.2 Data Schema

#### 3.2.1 Author-Documented Schema

The following schema is reproduced verbatim from the organizer's `readme.txt`:

**Corpus (`corpus.csv`)**

| Field | Type | Description (from readme.txt) |
|---|---|---|
| `text` | string | "Một đoạn văn bản pháp luật bất kỳ" |
| `cid` | int | "Id của đoạn văn bản đó trong corpus" |

**Training set (`train.csv`)**

| Field | Type | Description (from readme.txt) |
|---|---|---|
| `question` | string | "Dạng văn bản của câu hỏi" |
| `qid` | string | "Mã id của câu hỏi (viết tắt của question_id)" |
| `context` | list | "Các đoạn văn bản luật pháp liên quan" |
| `cid` | list | "Mã id của các đoạn văn bản pháp luật trong corpus có liên quan tới câu hỏi (viết tắt của context_id)" |

#### 3.2.2 Observed Data Characteristics

Beyond the readme, we observe the following from the actual data:

- The `cid` field in `train.csv` is serialized as a Python-style list (e.g., `[62492]` or `[58264 58688]`), where each integer maps to a passage in the corpus.
- The `context` field is similarly a serialized list of strings, where each string is the full text of a relevant corpus passage.
- The `updated_corpus.csv` file contains 571 more passages than the original `Train/corpus.csv` (262,168 vs 261,597), suggesting a post-release corpus update by the organizers.
- No document-level metadata is provided. Passages do not carry fields for parent document name, article number, effective date, or amendment status.
