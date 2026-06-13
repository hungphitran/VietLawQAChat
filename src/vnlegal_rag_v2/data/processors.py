import ast
import os
from typing import TypeAlias

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from vnlegal_rag_v2.utils.io import check_existing_files
from vnlegal_rag_v2.utils.text import SegmentationMethod, segment_text

TextSegmentationMethod: TypeAlias = SegmentationMethod


class DataProcessor:
    @staticmethod
    def clean_context(context: str) -> str:
        parsed = ast.literal_eval(context)

        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed).__name__}: {context}")

        if not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"Expected list[str], got: {parsed}")

        return "\n".join(parsed)

    @staticmethod
    def parse_cids(raw_cids: str) -> list[int]:
        raw_cids = raw_cids.strip("[]")

        if not raw_cids:
            return []

        return [int(cid) for cid in raw_cids.split()]

    @classmethod
    def build_qa_pairs(
        cls,
        train_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
    ) -> list[dict]:
        cid_to_text = dict(zip(corpus_df["cid"], corpus_df["text"]))
        samples = []

        for _, row in train_df.iterrows():
            cids = cls.parse_cids(row["cid"])

            for cid in cids:
                answer = cid_to_text[cid]
                samples.append(
                    {
                        "question": row["question"],
                        "positive_text": answer,
                        "positive_cid": cid,
                        "relevant_cids": cids,
                    }
                )

        return samples

    @staticmethod
    def segment_text(
        text: str,
        method: TextSegmentationMethod = "underthesea",
    ) -> str:
        return segment_text(text, method)

    @classmethod
    def segment_corpus(
        cls,
        corpus_df: pd.DataFrame,
        method: TextSegmentationMethod,
    ) -> pd.DataFrame:
        df = corpus_df.copy()
        df["text"] = [cls.segment_text(t, method) for t in tqdm(df["text"], desc=f"Segmenting corpus ({method})")]
        return df

    @classmethod
    def segment_qa_pairs(
        cls,
        samples: list[dict],
        method: TextSegmentationMethod = "underthesea",
    ) -> list[dict]:
        segmented_samples = []

        for sample in tqdm(samples, desc="Segmenting QA pairs"):
            segmented_samples.append(
                {
                    "question": cls.segment_text(sample["question"], method),
                    "positive_text": cls.segment_text(sample["positive_text"], method),
                    "positive_cid": sample["positive_cid"],
                    "relevant_cids": sample["relevant_cids"],
                }
            )

        return segmented_samples

    @staticmethod
    def split_train_eval(
        samples: list[dict],
        eval_size: float = 0.1,
        random_state: int = 36,
    ) -> tuple[list[dict], list[dict]]:
        df = pd.DataFrame(samples)
        unique_questions = pd.Series(df["question"].unique())
        train_q, eval_q = train_test_split(
            unique_questions,
            test_size=eval_size,
            random_state=random_state,
        )
        train_df = df[df["question"].isin(set(train_q))]
        eval_df = df[df["question"].isin(set(eval_q))]
        return train_df.to_dict("records"), eval_df.to_dict("records")

    @staticmethod
    def save_processed_data(
        train_data: list[dict],
        eval_data: list[dict],
        output_path: str,
        train_filename: str = "train.csv",
        eval_filename: str = "eval.csv",
        overwrite: bool = False,
    ) -> None:
        os.makedirs(output_path, exist_ok=True)

        output_files = [
            train_filename,
            eval_filename,
        ]

        if overwrite:
            for filename in output_files:
                filepath = os.path.join(output_path, filename)

                if os.path.exists(filepath):
                    os.remove(filepath)

        elif check_existing_files(output_path, output_files):
            raise FileExistsError(
                f"Output files already exist in: {output_path}"
            )

        columns = ["question", "positive_text", "positive_cid", "relevant_cids"]

        train_df = pd.DataFrame(train_data, columns=columns)
        eval_df = pd.DataFrame(eval_data, columns=columns)

        train_df.to_csv(
            os.path.join(output_path, train_filename),
            index=False,
        )

        eval_df.to_csv(
            os.path.join(output_path, eval_filename),
            index=False,
        )
