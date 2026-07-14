from __future__ import annotations

from pathlib import Path

import bm25s
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from vnlegal_rag_v2.utils.text import SegmentationMethod, segment_text

_VARIANT_MAP = {
    "bm25": "robertson",
    "bm25+": "bm25+",
}

# Module-level token cache: (id(texts), method) -> tokens
_token_cache: dict[tuple[int, str], object] = {}


def _get_tokens(texts: list[str], method: SegmentationMethod) -> object:
    """Tokenize texts with bm25s, keyed by (id(texts), method) so re-tokenizing the same
    list (e.g. index then query the same strings) is cached instead of recomputed."""
    cache_key = (id(texts), method)
    if cache_key not in _token_cache:
        segmented = [segment_text(t, method) for t in texts]
        _token_cache[cache_key] = bm25s.tokenize(
            segmented, stopwords=None, stemmer=None, show_progress=False,
        )
    return _token_cache[cache_key]


class BM25Retriever:
    def __init__(
        self,
        segmentation_method: SegmentationMethod = None,
        variant: str = "bm25",
        **bm25_kwargs,
    ):
        if variant not in _VARIANT_MAP:
            raise ValueError(f"Unknown BM25 variant: {variant!r}. Use {list(_VARIANT_MAP.keys())}")
        self._method: SegmentationMethod = segmentation_method
        self._variant = variant
        self._bm25_kwargs = bm25_kwargs
        self._bm25: bm25s.BM25 | None = None
        self.cids: list[int] = []

    def index(self, documents: list[str], cids: list[int]) -> None:
        """Tokenize corpus and build the BM25 index (variant maps bm25→robertson, bm25+→bm25+)."""
        assert len(documents) == len(cids)
        self.cids = list(cids)

        corpus_tokens = _get_tokens(documents, self._method)
        bm25_method = _VARIANT_MAP[self._variant]
        self._bm25 = bm25s.BM25(method=bm25_method, **self._bm25_kwargs)
        self._bm25.index(corpus_tokens, show_progress=False)

    def retrieve(self, queries: list[str], top_k: int = 100) -> list[list[int]]:
        assert self._bm25 is not None, "Call index() before retrieve()"

        query_tokens = _get_tokens(queries, self._method)
        results, _ = self._bm25.retrieve(query_tokens, k=top_k, show_progress=False)

        return [[self.cids[i] for i in row] for row in results]

    # --- index persistence: bm25s index (incl. vocab_dict) + cids to disk ---
    def index_signature(self) -> str:
        """Config that affects the stored index — changing it invalidates the cache."""
        return f"bm25|variant={self._variant}|seg={self._method}|kw={sorted(self._bm25_kwargs.items())}"

    def save(self, path: str | Path) -> None:
        assert self._bm25 is not None, "Call index() before save()"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(p / "bm25s"), show_progress=False)
        np.save(p / "cids.npy", np.asarray(self.cids, dtype=np.int64))

    def load(self, path: str | Path) -> None:
        p = Path(path)
        # vocab_dict is saved inside the bm25s index, so the standard query
        # tokenize path (no vocab passed) reproduces in-memory retrieval exactly.
        self._bm25 = bm25s.BM25.load(str(p / "bm25s"), load_corpus=False)
        self.cids = np.load(p / "cids.npy").tolist()


class TFIDFRetriever:
    def __init__(self, segmentation_method: SegmentationMethod = None, **kwargs):
        self._method: SegmentationMethod = segmentation_method
        self._vectorizer = TfidfVectorizer(
            tokenizer=lambda text: segment_text(text, self._method).split(),
            token_pattern="",
            **kwargs,
        )
        self._doc_matrix: spmatrix | None = None
        self.cids: list[int] = []

    def index(self, documents: list[str], cids: list[int]) -> None:
        assert len(documents) == len(cids)
        self.cids = list(cids)
        self._doc_matrix = self._vectorizer.fit_transform(documents)

    def retrieve(self, queries: list[str], top_k: int = 100, batch_size: int = 1024) -> list[list[int]]:
        assert self._doc_matrix is not None, "Call index() before retrieve()"

        results = []
        for start in tqdm(range(0, len(queries), batch_size), desc="Retrieving (TF-IDF)", total=(len(queries) + batch_size - 1) // batch_size):
            batch = queries[start:start + batch_size]
            query_matrix = self._vectorizer.transform(batch)
            scores = cosine_similarity(query_matrix, self._doc_matrix)
            for row in scores:
                top_indices = np.argsort(row)[::-1][:top_k]
                results.append([self.cids[i] for i in top_indices])

        return results
