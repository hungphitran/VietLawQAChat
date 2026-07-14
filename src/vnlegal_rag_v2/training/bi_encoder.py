from __future__ import annotations

import os

import torch

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from vnlegal_rag_v2.data.loaders import load_processed
from vnlegal_rag_v2.training.losses import GroupedBatchSamplerFactory, MultiPositiveContrastiveLoss
from vnlegal_rag_v2.utils.device import get_device
from vnlegal_rag_v2.utils.text import SegmentationMethod


class BiEncoderTrainer:
    def __init__(
        self,
        model_name_or_path: str = "bkai-foundation-models/vietnamese-bi-encoder",
        trust_remote_code: bool = True,
        device: str | None = None,
    ):
        self.device = device or get_device()
        self.model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": torch.float32},
        )

    def train(
        self,
        data_path: str,
        output_dir: str = "models/bi-encoder",
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int | None = None,
        segmentation: SegmentationMethod = None,
        loss_type: str = "mnr",
        save_strategy: str = "epoch",
        save_steps: int = 500,
        save_total_limit: int = 3,
        eval_strategy: str = "epoch",
        metric_for_best_model: str = "eval_loss",
        seed: int = 28,
        **kwargs,
    ) -> None:
        """Fine-tune the bi-encoder with contrastive learning, saving the best checkpoint.

        Loads pre-segmented CSVs when `segmentation` is set (suffix convention), else raw.
        `loss_type="mnr"` uses MultipleNegativesRankingLoss; `"multi_positive"` groups
        multiple positives per query (see `_setup_multi_positive`). Best model → `output_dir/best`.
        """
        if max_length is not None:
            self.model.max_seq_length = max_length

        # Load pre-segmented files if segmentation specified, else raw
        suffix = f"_{segmentation}" if segmentation else ""
        train_df = load_processed(data_path, f"train{suffix}.csv")
        eval_df = load_processed(data_path, f"eval{suffix}.csv")

        if loss_type == "multi_positive":
            train_dataset, loss, batch_sampler_fn, train_bs = self._setup_multi_positive(
                train_df, batch_size, seed,
            )
        else:
            train_dataset = Dataset.from_dict(
                {"query": train_df["question"].tolist(), "positive": train_df["positive_text"].tolist()}
            ).shuffle(seed=seed)
            loss = losses.MultipleNegativesRankingLoss(self.model)
            batch_sampler_fn = BatchSamplers.NO_DUPLICATES
            train_bs = batch_size

        eval_dataset = Dataset.from_dict(
            {"query": eval_df["question"].tolist(), "positive": eval_df["positive_text"].tolist()}
        )

        defaults = dict(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            bf16=self.device not in ("cpu", "mps"),
            batch_sampler=batch_sampler_fn,
            eval_strategy=eval_strategy,
            eval_steps=1,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False,
            logging_strategy="epoch",
            logging_steps=1,
            weight_decay=weight_decay,
            report_to="none",
            log_level="error",
        )
        defaults.update(kwargs)

        train_args = SentenceTransformerTrainingArguments(**defaults)

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        trainer.train()
        best_path = os.path.join(output_dir, "best")
        self.model.save(best_path)
        print(f"Best model saved to {best_path}")

    def _setup_multi_positive(
        self,
        train_df,
        batch_size: int,
        seed: int,
    ) -> tuple[Dataset, MultiPositiveContrastiveLoss, callable, int]:
        """Setup for multi-positive contrastive loss.

        Groups rows by (question, positive_cid) pairs as unique samples.
        Each unique question gets a group_id so all its positives land in the same batch.
        """
        # Deduplicate: one row per (question, positive_cid)
        dedup = train_df.drop_duplicates(subset=["question", "positive_cid"]).reset_index(drop=True)

        # Assign group_id per unique question
        questions = dedup["question"].tolist()
        q_to_gid: dict[str, int] = {}
        group_ids: list[int] = []
        for q in questions:
            if q not in q_to_gid:
                q_to_gid[q] = len(q_to_gid)
            group_ids.append(q_to_gid[q])

        from collections import Counter
        group_counts = Counter(group_ids)
        multi = sum(1 for gid, cnt in group_counts.items() if cnt > 1)
        print(f"Multi-positive: {len(dedup)} pairs, {len(q_to_gid)} unique queries")
        print(f"  Multi-positive queries: {multi}/{len(q_to_gid)}")

        train_dataset = Dataset.from_dict({
            "query": dedup["question"].tolist(),
            "positive": dedup["positive_text"].tolist(),
            "label": group_ids,
        })

        loss = MultiPositiveContrastiveLoss(self.model)

        # Effective batch_size must fit largest group
        max_group = max(group_counts.values()) if group_counts else 1
        effective_bs = max(batch_size, max_group)

        batch_sampler_fn = GroupedBatchSamplerFactory(group_ids, effective_bs, seed)

        return train_dataset, loss, batch_sampler_fn, effective_bs

    def evaluate(
        self,
        model_path: str,
        data_path: str = "data/processed",
        segmentation: SegmentationMethod = None,
        top_k: int = 100,
        metrics: list[dict] | None = None,
        encode_kwargs: dict | None = None,
    ) -> dict[str, float]:
        """Zero-shot retrieval eval: index `data_path` corpus with `model_path`, retrieve, score.

        No segmentation here — the corpus file is already segmented when pre-segmented.
        """
        from vnlegal_rag_v2.data.loaders import extract_queries
        from vnlegal_rag_v2.evaluation.evaluator import Evaluator
        from vnlegal_rag_v2.evaluation.metrics import (
            mrr_at_k,
            ndcg_at_k,
            recall_at_k,
            success_at_k,
        )
        from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
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

        model = BiEncoderModel(model_path, trust_remote_code=True)
        # Text already segmented when loading pre-segmented files — no re-segmentation needed
        retriever = DenseRetriever(model, segmentation_method=None, encode_kwargs=encode_kwargs)

        suffix = f"_{segmentation}" if segmentation else ""
        eval_df = load_processed(data_path, f"eval{suffix}.csv")
        corpus_df = load_processed(data_path, f"corpus{suffix}.csv")

        queries, relevant_cids = extract_queries(eval_df)
        documents = corpus_df["text"].tolist()
        cids = corpus_df["cid"].tolist()

        retriever.index(documents, cids)
        predictions = retriever.retrieve(queries, top_k)

        results = Evaluator(predictions, relevant_cids).evaluate(metric_defs)

        for name, score in results.items():
            print(f"  {name}: {score:.4f}")

        return results
