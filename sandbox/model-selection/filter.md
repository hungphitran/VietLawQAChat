# [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

MTEB is used as **Phase 1: soft model selection** only.  
The selected models are then checked in **Phase 2: zero-shot evaluation** on our Vietnamese legal retrieval dataset.

MTEB paper: https://aclanthology.org/2023.eacl-main.148/

---

## PHASE 1: MTEB soft selection

### Task Type

- Retrieval
- Reranking
- PairClassification

### Domain

- Legal
- Government
- Written
- Encyclopaedic
- Web
- Non-fiction
- Academic

### Language

- vie
- eng
- fra
- deu
- ita
- ell
- cmn
- jpn

### Tasks

- AILAStatutes
- LegalBenchCorporateLobbying
- MIRACLRetrievalHardNegatives
- MLQARetrieval
- WikipediaRetrievalMultilingual
- HagridRetrieval
- StatcanDialogueDatasetRetrieval
- ArguAna
- T2Reranking
- VoyageMMarcoReranking
- WikipediaRerankingMultilingual
- XNLI
- RTE3
- CTKFactsNLI
- PawsXPairClassification

### Model filter

- `active params <= 0.3B`
- `total params != null`

### Manual post-filtering

- Remove non-commercial models, e.g. `CC-BY-NC`.
- Remove models without public architecture / paper / technical report / detailed model card.
- Remove older or weaker models when a newer model in the same family is available.
- Remove models that are unstable in our environment, e.g. frequent CUDA/runtime errors.
- Keep open-weight models if they have public architecture and terms allowing research or commercial use.
- Treat `total params != null` only as a heuristic, not as proof of open-source status.

### Phase 1 candidate set

- `embeddinggemma-300m`
- `F2LLM-v2-330M`
- `granite-embedding-311m-multilingual-r2`
- `harrier-oss-v1-270m`
- `F2LLM-v2-160M`
- `granite-embedding-97m-multilingual-r2`
- `multilingual-e5-base`

---

## PHASE 2: Local zero-shot screening

### Setup

- Sample ratio: `0.2%` of the full dataset
- Queries: `26,714`
- Corpus documents: `19,962`
- Evaluation type: zero-shot dense retrieval
- Main screening metrics: `Recall@10`, `Recall@100`, `Success@100`
- Supporting metric: `MRR@10`

### Additional Vietnamese baselines

- `vietnamese-bi-encoder`
- `phobert-base`
- `phobert-base-v2`
- `phobert-large`

### Selection notes

- `embeddinggemma-300m`: selected as the top-tier candidate because it gives the best `Recall@10` and `Recall@100` among the strong ~300M models.
- `granite-embedding-97m-multilingual-r2`: selected as the ultra-lightweight candidate because it has only `0.028B` active parameters while keeping competitive recall.
- `multilingual-e5-base`: selected as a standard multilingual baseline.
- `vietnamese-bi-encoder`: selected as the Vietnamese-specific baseline because it clearly outperforms the PhoBERT variants.

### Final models after Phase 2

- `embeddinggemma-300m`
- `granite-embedding-97m-multilingual-r2`
- `multilingual-e5-base`
- `vietnamese-bi-encoder`

---

## Note

Phase 1 and Phase 2 are both screening steps.  
The final conclusion should be based on a larger held-out evaluation or fine-tuning experiments.
