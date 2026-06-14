from __future__ import annotations

import os

from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import (
    CrossEncoderTrainer as _CrossEncoderHFTrainer,
)
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from vnlegal_rag_v2.data.loaders import extract_queries, load_processed
from vnlegal_rag_v2.utils.device import get_device


def _get_loss_fn(loss_type: str, model):
    """Return loss module for cross-encoder training."""
    if loss_type == "ranknet":
        from sentence_transformers.cross_encoder.losses import RankNetLoss
        return RankNetLoss(model=model)
    else:  # default: bce
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        return BinaryCrossEntropyLoss(model=model)


class CrossEncoderTrainer:
    def __init__(
        self,
        model_name_or_path: str = "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length: int = 512,
        trust_remote_code: bool = True,
        device: str | None = None,
    ):
        self.device = device or get_device()
        self.model = CrossEncoder(
            model_name_or_path,
            max_length=max_length,
            device=self.device,
            trust_remote_code=trust_remote_code,
        )

        # Use fp16 in training args for proper mixed precision, not .half()

    def train(
        self,
        data_path: str = "data/processed",
        neg_path: str | None = None,
        output_dir: str = "models/cross-encoder",
        num_epochs: int = 2,
        batch_size: int = 64,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        segmentation: str | None = None,
        num_negatives: int | None = None,
        loss_type: str = "bce",
        save_strategy: str = "epoch",
        save_steps: int = 500,
        save_total_limit: int = 3,
        eval_strategy: str = "epoch",
        metric_for_best_model: str = "eval_loss",
        seed: int = 42,
        **kwargs,
    ) -> None:
        suffix = f"_{segmentation}" if segmentation else ""
        pos_df = load_processed(data_path, f"train{suffix}.csv")
        eval_df = load_processed(data_path, f"eval{suffix}.csv")

        train_dataset = self._build_dataset(pos_df, neg_path, seed, num_negatives, loss_type)

        loss = _get_loss_fn(loss_type, self.model)

        # RankNet uses listwise format — incompatible with pointwise eval, skip eval
        eval_dataset = None
        if loss_type != "ranknet":
            eval_dataset = Dataset.from_dict({
                "sentence_0": eval_df["question"].tolist(),
                "sentence_1": eval_df["positive_text"].tolist(),
                "label": [1.0] * len(eval_df),
            })

        defaults = dict(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            fp16=self.device not in ("cpu", "mps"),
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="epoch" if loss_type != "ranknet" else "no",
            eval_steps=1,
            save_strategy="epoch" if loss_type != "ranknet" else "no",
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=loss_type != "ranknet",
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False,
            logging_strategy="epoch",
            logging_steps=1,
            report_to="none",
            log_level="error",
        )
        defaults.update(kwargs)

        train_args = CrossEncoderTrainingArguments(**defaults)

        trainer = _CrossEncoderHFTrainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        trainer.train()
        self.model.save(os.path.join(output_dir, "best"))

    def evaluate(
        self,
        model_path: str,
        data_path: str = "data/processed",
        segmentation: str | None = None,
        retriever_model: str | None = None,
        retrieve_top_k: int = 100,
        rerank_top_k: int = 100,
        metrics: list[dict] | None = None,
        encode_kwargs: dict | None = None,
    ) -> dict[str, float]:
        from vnlegal_rag_v2.evaluation.evaluator import Evaluator
        from vnlegal_rag_v2.evaluation.metrics import (
            mrr_at_k,
            ndcg_at_k,
            recall_at_k,
            success_at_k,
        )
        from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
        from vnlegal_rag_v2.models.cross_encoder import CrossEncoderModel
        from vnlegal_rag_v2.reranking import CrossEncoderReranker
        from vnlegal_rag_v2.retrieval.dense import DenseRetriever

        if metrics is None:
            metrics = [
                {"name": "mrr@10", "fn": "mrr_at_k", "k": 10},
                {"name": "success@100", "fn": "success_at_k", "k": 100},
            ]

        fn_map = {
            "mrr_at_k": mrr_at_k,
            "success_at_k": success_at_k,
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k,
        }
        metric_defs = [(m["name"], fn_map[m["fn"]], m["k"]) for m in metrics]

        # Load data
        suffix = f"_{segmentation}" if segmentation else ""
        eval_df = load_processed(data_path, f"eval{suffix}.csv")
        corpus_df = load_processed(data_path, f"corpus{suffix}.csv")
        queries, relevant_cids = extract_queries(eval_df)
        documents = corpus_df["text"].tolist()
        cids = corpus_df["cid"].tolist()

        # Stage 1: Dense retrieval
        bi_model = BiEncoderModel(
            retriever_model or "phatvucoder/vietnamese-bi-encoder",
            trust_remote_code=True,
        )
        retriever = DenseRetriever(bi_model, encode_kwargs=encode_kwargs)
        retriever.index(documents, cids)
        predictions = retriever.retrieve(queries, top_k=retrieve_top_k)

        # Stage 2: Cross-encoder reranking
        import transformers
        transformers.logging.set_verbosity_error()

        ce_model = CrossEncoderModel(model_path)
        reranker = CrossEncoderReranker(ce_model, batch_size=256)
        predictions = reranker.rerank(
            queries, predictions, documents, cids, top_k=rerank_top_k,
            show_progress_bar=False,
        )

        # Compute metrics
        results = Evaluator(predictions, relevant_cids).evaluate(metric_defs)
        for name, score in results.items():
            print(f"  {name}: {score:.4f}")
        return results

    @staticmethod
    def _build_dataset(
        pos_df,
        neg_path: str | None,
        seed: int,
        num_negatives: int | None = None,
        loss_type: str = "bce",
    ) -> Dataset:
        """Build train dataset from positive examples + pre-mined negatives CSV.

        BCEWithLogitsLoss  → pointwise: (sentence_0, sentence_1, label)
        RankNetLoss        → listwise:  (query, docs, labels) grouped per query
        """
        import pandas as pd

        # Collect negatives per query
        neg_by_query: dict[str, list[str]] = {}
        if neg_path:
            neg_df = pd.read_csv(neg_path, encoding="utf-8")
            if num_negatives is not None:
                q_col = "question" if "question" in neg_df.columns else "query"
                neg_df = neg_df.groupby(q_col).head(num_negatives).reset_index(drop=True)
            q_col = "question" if "question" in neg_df.columns else "query"
            a_col = "answer" if "answer" in neg_df.columns else "negative_text"
            for _, row in neg_df.iterrows():
                neg_by_query.setdefault(row[q_col], []).append(row[a_col])

        if loss_type == "ranknet":
            return CrossEncoderTrainer._build_listwise_dataset(pos_df, neg_by_query)
        return CrossEncoderTrainer._build_pointwise_dataset(pos_df, neg_by_query)

    @staticmethod
    def _build_pointwise_dataset(pos_df, neg_by_query: dict[str, list[str]]) -> Dataset:
        s0, s1, labels = [], [], []
        for _, row in pos_df.iterrows():
            s0.append(row["question"])
            s1.append(row["positive_text"])
            labels.append(1.0)
            for neg in neg_by_query.get(row["question"], []):
                s0.append(row["question"])
                s1.append(neg)
                labels.append(0.0)
        return Dataset.from_dict({"sentence_0": s0, "sentence_1": s1, "label": labels})

    @staticmethod
    def _build_listwise_dataset(pos_df, neg_by_query: dict[str, list[str]]) -> Dataset:
        queries, docs, lbls = [], [], []
        for _, row in pos_df.iterrows():
            q = row["question"]
            negs = neg_by_query.get(q, [])
            queries.append(q)
            docs.append([row["positive_text"]] + negs)
            lbls.append([1.0] + [0.0] * len(negs))
        return Dataset.from_dict({"query": queries, "docs": docs, "label": lbls})
