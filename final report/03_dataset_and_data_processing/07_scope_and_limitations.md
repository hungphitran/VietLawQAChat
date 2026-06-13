### 3.7 Dataset Scope and Limitations

The organizers did not document the dataset's scope, collection criteria, or intended limitations. Based on the available readme and our inspection of the data, we note the following:

**What the organizers documented:**
- The corpus contains "một đoạn văn bản pháp luật bất kỳ" (any legal passage).
- Each passage has a unique cid.
- Training questions are linked to relevant passages via cid lists.

**What the organizers did not document:**
- The source database or collection methodology.
- The segmentation strategy used to split documents into passages.
- The time period covered by the corpus.
- Whether the corpus is a snapshot or includes version history.
- The criteria for selecting which legal documents to include.
- The annotation process for question–relevance pairs.

**Observed limitations:**
- The corpus is a **flat snapshot** with no temporal metadata. Passages do not carry effective dates or amendment status.
- No document-level grouping is provided. The relationship between passages from the same legal document is not encoded in the schema.
- Some corpus passages are degenerate (1 token, e.g., `"..."`), and some are extremely long (55K+ tokens), suggesting the pre-segmentation is not uniform.
- The training set contains duplicate questions with different qids (3,988 rows), the purpose of which is not explained.

These limitations do not prevent retrieval evaluation, but they constrain the types of methods that can benefit from the data. In particular, the lack of structured metadata means the task is purely text-to-text matching, without leveraging document hierarchy, temporal validity, or amendment chains.
