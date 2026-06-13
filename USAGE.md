# Scripts & Commands

---

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate vnlegal-rag-v2
```
```bash
conda env update -f environment.yml --prune
```

```bash
conda remove -n vnlegal-rag --all
conda env create -f environment.yml
```

### Pip

```bash
pip install -e .
pip install -e ".[dev]"
```

## 1. Data Processing

```bash
# Default: eval_size=0.1, segmentation=none
python scripts/process_data.py --config configs/data/default.yaml
```

Create multiple splits by copying and editing the config:

```bash
cp configs/data/default.yaml configs/data/small_eval.yaml
# edit eval_size, random_state, segmentation as needed
python scripts/process_data.py --config configs/data/small_eval.yaml
```

---

## 2. Training

```bash
# Bi-Encoder
python scripts/train_bi.py --config configs/train/bi_encoder.yaml

# Cross-Encoder (run after bi-encoder training + negative mining)
python scripts/train_cross.py --config configs/train/cross_encoder.yaml
```

---

## 3. Run Pipeline (Evaluation)

### Single experiment

```bash
python scripts/run_pipeline.py configs/pipeline/bm25_only.yaml
```

### Batch — all pipeline experiments

```bash
python scripts/run_pipeline.py configs/pipeline/*.yaml --results results/scores.json
```

### Batch — all model-selection (zero-shot)

```bash
python scripts/run_pipeline.py configs/model-selection/*.yaml --results results/model-selection/scores.json
```

### Single model-selection

```bash
python scripts/run_pipeline.py configs/model-selection/bge-m3.yaml --results results/model-selection/scores.json
```

### Run everything at once

```bash
python scripts/run_pipeline.py configs/pipeline/*.yaml configs/model-selection/*.yaml --results results/scores.json
```

---

## 4. Results

Results accumulate into a single JSON keyed by experiment name:

```bash
cat results/scores.json
```

Example output:

```json
{
  "bm25-only": { "mrr@10": 0.45, "success@10": 0.62, "recall@100": 0.78 },
  "zero-shot-bge-m3": { "mrr@10": 0.71, "success@10": 0.85, "recall@100": 0.93 }
}
```

---

## Full Workflow (end-to-end)

```bash
# Step 1: Process data
python scripts/process_data.py --config configs/data/default.yaml

# Step 2: Train Bi-Encoder
python scripts/train_bi.py --config configs/train/bi_encoder.yaml

# Step 3: Train Cross-Encoder
python scripts/train_cross.py --config configs/train/cross_encoder.yaml

# Step 4: Zero-shot model selection
python scripts/run_pipeline.py configs/model-selection/*.yaml --results results/model-selection/scores.json

# Step 5: Pipeline experiments (sparse, dense, hybrid, + reranking)
python scripts/run_pipeline.py configs/pipeline/*.yaml --results results/pipeline/scores.json
```
