import os

from vnlegal_rag_v2.data.loaders import DataLoader
from vnlegal_rag_v2.data.processors import DataProcessor
from vnlegal_rag_v2.utils.io import check_existing_files
from vnlegal_rag_v2.utils.text import SegmentationMethod


class DataPreparationPipeline:
    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        eval_size: float = 0.1,
        random_state: int = 36,
        segmentation_methods: list[SegmentationMethod] | None = None,
        overwrite: bool = False,
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.eval_size = eval_size
        self.random_state = random_state
        self.overwrite = overwrite
        self.segmentation_methods: list[SegmentationMethod] = segmentation_methods or [None]

    def run(self) -> None:
        train_df, corpus_df = DataLoader.load_raw_data(self.raw_path)

        # Build QA pairs and split once (same split for all segmentations)
        samples = DataProcessor.build_qa_pairs(train_df, corpus_df)
        train_data, eval_data = DataProcessor.split_train_eval(
            samples,
            eval_size=self.eval_size,
            random_state=self.random_state,
        )

        for method in self.segmentation_methods:
            suffix = f"_{method}" if method else ""
            train_file = f"train{suffix}.csv"
            eval_file = f"eval{suffix}.csv"
            corpus_file = f"corpus{suffix}.csv"

            if method:
                train_seg = DataProcessor.segment_qa_pairs(train_data, method=method)
                eval_seg = DataProcessor.segment_qa_pairs(eval_data, method=method)
                corpus_seg = DataProcessor.segment_corpus(corpus_df, method)
            else:
                train_seg = train_data
                eval_seg = eval_data
                corpus_seg = corpus_df

            DataProcessor.save_processed_data(
                train_data=train_seg,
                eval_data=eval_seg,
                output_path=self.processed_path,
                train_filename=train_file,
                eval_filename=eval_file,
                overwrite=self.overwrite,
            )
            corpus_seg.to_csv(
                os.path.join(self.processed_path, corpus_file),
                index=False,
            )
            print(f"  Saved: {train_file}, {eval_file}, {corpus_file}")

        print(f"Data processing completed. Saved to {self.processed_path}")
