"""Train Cross-Encoder model."""

import argparse
import os

import yaml

from vnlegal_rag_v2.evaluation.evaluator import Evaluator
from vnlegal_rag_v2.training.cross_encoder import CrossEncoderTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Encoder")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    # CLI overrides for experiment loops
    parser.add_argument("--neg-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--loss-type", type=str, default=None, choices=["bce", "ranknet"])
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output-scores", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for key, val in vars(args).items():
        if key != "config" and val is not None:
            config[key] = val

    trainer = CrossEncoderTrainer(
        model_name_or_path=config.get("model_name", "Alibaba-NLP/gte-multilingual-reranker-base"),
        max_length=config.get("max_length", 512),
        trust_remote_code=config.get("trust_remote_code", True),
        device=config.get("device"),
    )
    seg_value = config.get("segmentation")
    segmentation = None if seg_value in (None, "null", "none", "") else seg_value

    trainer.train(
        data_path=config.get("data_path", "data/processed"),
        neg_path=config.get("neg_path"),
        output_dir=config.get("output_dir", "models/cross-encoder"),
        num_epochs=config.get("num_epochs", 2),
        batch_size=config.get("batch_size", 64),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_steps=config.get("warmup_steps", 0),
        segmentation=segmentation,
        num_negatives=config.get("num_negatives"),
        loss_type=config.get("loss_type", "bce"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        seed=config.get("seed", 42),
    )

    # Evaluate best model on eval set
    best_model_path = os.path.join(config.get("output_dir", "models/cross-encoder"), "best")
    print(f"\nEvaluating {best_model_path}...")

    metrics_config = config.get("eval_metrics", [
        {"name": "mrr@10", "fn": "mrr_at_k", "k": 10},
        {"name": "success@100", "fn": "success_at_k", "k": 100},
    ])

    scores = trainer.evaluate(
        model_path=best_model_path,
        data_path=config.get("data_path", "data/processed"),
        segmentation=segmentation,
        retriever_model=config.get("retriever_model"),
        retrieve_top_k=config.get("retrieve_top_k", 100),
        rerank_top_k=config.get("rerank_top_k", 100),
        metrics=metrics_config,
        encode_kwargs={"batch_size": config.get("batch_size", 64), "show_progress_bar": True},
    )

    # Save scores
    output_scores = config.get("output_scores")
    experiment_name = config.get("name", "cross-unknown")
    if output_scores:
        Evaluator.save_results({experiment_name: scores}, output_scores)

    # Also update master scores.json
    Evaluator.save_results({experiment_name: scores}, config.get("scores_json", "results/scores.json"))


if __name__ == "__main__":
    main()
