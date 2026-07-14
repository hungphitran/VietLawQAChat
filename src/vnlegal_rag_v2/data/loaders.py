from __future__ import annotations

import ast
import os

import pandas as pd


def extract_corpus(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    dedup = df.drop_duplicates(subset=["positive_cid"])
    documents = dedup["positive_text"].tolist()
    cids = dedup["positive_cid"].tolist()
    return documents, cids


def extract_queries(df: pd.DataFrame) -> tuple[list[str], list[list[int]]]:
    questions = df["question"].tolist()
    relevant = df["relevant_cids"].apply(ast.literal_eval).tolist()
    return questions, relevant


def load_processed(
    processed_path: str,
    filename: str = "eval.csv",
) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(processed_path, filename), encoding="utf-8"
    )


class DataLoader:
    @staticmethod
    def load_raw_data(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Recursively find `train.csv` + `updated_corpus.csv` under `raw_path` and load both."""
        train_path = None
        corpus_path = None

        for dirpath, _, filenames in os.walk(raw_path):
            if "train.csv" in filenames:
                train_path = os.path.join(dirpath, "train.csv")

            if "updated_corpus.csv" in filenames:
                corpus_path = os.path.join(dirpath, "updated_corpus.csv")

        if train_path is None or corpus_path is None:
            raise FileNotFoundError(
                "Could not find 'train.csv' or 'updated_corpus.csv'."
            )

        train_df = pd.read_csv(train_path, encoding="utf-8")
        corpus_df = pd.read_csv(corpus_path, encoding="utf-8")

        return train_df, corpus_df
