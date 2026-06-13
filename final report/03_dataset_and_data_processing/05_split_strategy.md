### 3.5 Split Strategy

The raw dataset does not include a pre-defined evaluation split. We create the train/eval split ourselves using a **question-level strategy**:

1. Expand raw rows into question–positive pairs (one row per CID in `relevant_cids`).
2. Extract all unique question texts.
3. Split the unique questions (not rows) into 90% train / 10% eval using `sklearn.model_selection.train_test_split` with `random_state=36`.
4. Assign all rows belonging to train questions to the training set, and all rows belonging to eval questions to the evaluation set.

This ensures that the same question text never appears in both sets. The fixed random state makes the split reproducible across experiments.

The question-level split is important because the training set contains 3,988 duplicate question rows. A naive row-level split would place identical questions in both sets, inflating evaluation scores.
