"""Mine negative examples for cross-encoder training using a trained bi-encoder."""

from __future__ import annotations

import argparse
import ast

import yaml

from vnlegal_rag_v2.data.loaders import extract_queries, load_processed
from vnlegal_rag_v2.mining import NegativeMiner
from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
from vnlegal_rag_v2.retrieval.dense import DenseRetriever
from vnlegal_rag_v2.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Mine negatives from bi-encoder predictions")
    parser.add_argument("--config", type=str, default="configs/data/negative_mining.yaml")
    parser.add_argument("--num-negatives", type=int, default=None, help="Override num_negatives from config")
    parser.add_argument("--output", type=str, default=None, help="Override output_path from config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.num_negatives is not None:
        cfg["num_negatives"] = args.num_negatives
    if args.output is not None:
        cfg["output_path"] = args.output

    device = get_device()
    model = BiEncoderModel(cfg["model_path"], trust_remote_code=True)
    retriever = DenseRetriever(model, encode_kwargs={"batch_size": cfg.get("batch_size", 128), "show_progress_bar": True})

    # Load corpus and train data — use segmented files when segmentation specified
    seg = cfg.get("segmentation")
    suffix = f"_{seg}" if seg else ""
    corpus_df = load_processed(cfg["data_path"], f"corpus{suffix}.csv")
    documents = corpus_df["text"].tolist()
    cids = corpus_df["cid"].tolist()
    train_df = load_processed(cfg["data_path"], f"train{suffix}.csv")
    queries = train_df["question"].tolist()
    relevant_cids = [
        ast.literal_eval(c) if isinstance(c, str) else c
        for c in train_df["relevant_cids"]
    ]

    miner = NegativeMiner(retriever, documents, cids)
    df = miner.mine_to_csv(
        queries,
        relevant_cids,
        output_path=cfg["output_path"],
        strategy=cfg.get("strategy", "moderate"),
        num_negatives=cfg.get("num_negatives", 3),
        top_k=cfg.get("top_k", 100),
        seed=cfg.get("seed", 28),
    )

    print(f"Mined {len(df)} negative pairs → {cfg['output_path']}")


if __name__ == "__main__":
    main()
