from tqdm import tqdm

import faiss

from vnlegal_rag_v2.models.bi_encoder import BiEncoderModel
from vnlegal_rag_v2.utils.text import SegmentationMethod, segment_text


class DenseRetriever:
    def __init__(
        self,
        model: BiEncoderModel,
        segmentation_method: SegmentationMethod = None,
        encode_kwargs: dict | None = None,
    ):
        self.model = model
        self._seg_method: SegmentationMethod = segmentation_method
        self._encode_kwargs = encode_kwargs or {}
        self._index: faiss.IndexFlatIP | None = None
        self.cids: list[int] = []

    def _segment(self, texts: list[str]) -> list[str]:
        if self._seg_method is None:
            return texts
        return [segment_text(t, self._seg_method) for t in tqdm(texts, desc=f"Segmenting ({self._seg_method})")]

    def index(self, documents: list[str], cids: list[int]) -> None:
        assert len(documents) == len(cids)
        self.cids = list(cids)

        embeddings = self.model.encode(self._segment(documents), **self._encode_kwargs).astype('float32')
        faiss.normalize_L2(embeddings)  # type: ignore[arg-type]

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)  # type: ignore[arg-type]

    def retrieve(self, queries: list[str], top_k: int = 100) -> list[list[int]]:
        assert self._index is not None, "Call index() before retrieve()"

        query_embeddings = self.model.encode(self._segment(queries), **self._encode_kwargs).astype('float32')
        faiss.normalize_L2(query_embeddings)  # type: ignore[arg-type]

        _, indices = self._index.search(query_embeddings, min(top_k, len(self.cids)))  # type: ignore[arg-type]

        return [
            [self.cids[i] for i in row if i != -1]
            for row in indices
        ]
