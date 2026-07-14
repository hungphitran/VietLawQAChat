from __future__ import annotations

from pathlib import Path

import numpy as np
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
        model_name: str | None = None,
    ):
        self.model = model
        self._model_name = model_name
        self._seg_method: SegmentationMethod = segmentation_method
        self._encode_kwargs = encode_kwargs or {}
        self._index: faiss.IndexFlatIP | None = None
        self.cids: list[int] = []

    def _segment(self, texts: list[str]) -> list[str]:
        if self._seg_method is None:
            return texts
        return [segment_text(t, self._seg_method) for t in tqdm(texts, desc=f"Segmenting ({self._seg_method})")]

    def index(self, documents: list[str], cids: list[int]) -> None:
        """Encode documents with the bi-encoder and add to a FAISS inner-product index.

        Vectors are L2-normalized so inner product == cosine similarity (IndexFlatIP).
        """
        assert len(documents) == len(cids)
        self.cids = list(cids)

        embeddings = self.model.encode(self._segment(documents), **self._encode_kwargs).astype('float32')
        faiss.normalize_L2(embeddings)  # type: ignore[arg-type]

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)  # type: ignore[arg-type]

    def retrieve(self, queries: list[str], top_k: int = 100) -> list[list[int]]:
        """Return the top_k cids per query by cosine similarity. -1 padding from FAISS is filtered."""
        assert self._index is not None, "Call index() before retrieve()"

        query_embeddings = self.model.encode(self._segment(queries), **self._encode_kwargs).astype('float32')
        faiss.normalize_L2(query_embeddings)  # type: ignore[arg-type]

        _, indices = self._index.search(query_embeddings, min(top_k, len(self.cids)))  # type: ignore[arg-type]

        return [
            [self.cids[i] for i in row if i != -1]
            for row in indices
        ]

    # --- index persistence: FAISS index + cids to disk ---
    def index_signature(self) -> str:
        """Config that affects the stored vectors — changing it invalidates the cache."""
        return f"dense|model={self._model_name}|seg={self._seg_method}|kw={sorted(self._encode_kwargs)}"

    def save(self, path: str | Path) -> None:
        assert self._index is not None, "Call index() before save()"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "dense.faiss"))
        np.save(p / "cids.npy", np.asarray(self.cids, dtype=np.int64))

    def load(self, path: str | Path) -> None:
        p = Path(path)
        self._index = faiss.read_index(str(p / "dense.faiss"))
        self.cids = np.load(p / "cids.npy").tolist()
