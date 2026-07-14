"""Train Bi-Encoder model."""

from __future__ import annotations

import argparse
import os

import yaml

from vnlegal_rag_v2.evaluation.evaluator import Evaluator
from vnlegal_rag_v2.training.bi_encoder import BiEncoderTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Bi-Encoder")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--results", type=str, default="results/scores.json",
                        help="Overall scores file (default: results/scores.json)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = BiEncoderTrainer(
        model_name_or_path=config.get("model_name", "bkai-foundation-models/vietnamese-bi-encoder"),
        trust_remote_code=config.get("trust_remote_code", True),
        device=config.get("device"),
    )
    # segmentation: null for multilingual, pyvi for vietnamese-bi-encoder
    seg_value = config.get("segmentation")
    segmentation = None if seg_value in (None, "null", "none", "") else seg_value

    trainer.train(
        data_path=config.get("data_path", "data/processed"),
        output_dir=config.get("output_dir", "models/BiEncoder/model1"),
        num_epochs=config.get("num_epochs", 3),
        batch_size=config.get("batch_size", 32),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        max_length=config.get("max_length"),
        segmentation=segmentation,
        loss_type=config.get("loss_type", "mnr"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        seed=config.get("seed", 28),
    )

    best_model_path = os.path.join(config.get("output_dir", "models/BiEncoder/model1"), "best")
    print(f"\nEvaluating {best_model_path}...")

    metrics_config = config.get("eval_metrics", [
        {"name": "mrr@10", "fn": "mrr_at_k", "k": 10},
        {"name": "success@100", "fn": "success_at_k", "k": 100},
    ])

    scores = trainer.evaluate(
        model_path=best_model_path,
        data_path=config.get("data_path", "data/processed"),
        segmentation=segmentation,
        top_k=config.get("eval_top_k", 100),
        metrics=metrics_config,
        encode_kwargs={"batch_size": config.get("batch_size", 32), "show_progress_bar": True},
    )

    output_scores = config.get("output_scores")
    experiment_name = config.get("name", "train-unknown")
    if output_scores:
        Evaluator.save_results({experiment_name: scores}, output_scores)

    Evaluator.save_results({experiment_name: scores}, args.results)


if __name__ == "__main__":
    main()
