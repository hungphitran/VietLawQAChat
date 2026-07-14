"""Disk cache for the built retrieval index.

On startup the app can either re-encode the whole corpus (minutes on the full
262k-doc set) or, far better, load a persisted FAISS/bm25s index (seconds).
`load_or_build_index` does the load-or-build dance with automatic invalidation:
the manifest stores sha256(corpus bytes + retriever.index_signature() + n_docs),
so changing the corpus, the model, the segmentation, or a fusion weight rebuilds
the cache from scratch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MiB chunks
            h.update(chunk)
    return h.hexdigest()


def _manifest_key(corpus_path: str | Path, retriever, n_docs: int) -> str:
    sig = f"{_file_hash(corpus_path)}|{retriever.index_signature()}|n={n_docs}"
    return hashlib.sha256(sig.encode()).hexdigest()


def load_or_build_index(
    pipeline,
    documents: list[str],
    cids: list[int],
    corpus_path: str | Path,
    index_dir: str | Path,
) -> bool:
    """Load the retriever index from `index_dir` if its manifest matches; otherwise
    build it via `pipeline.index(...)` and persist. Returns True on a cache hit.

    On a hit the FAISS/bm25s structures are read straight from disk and the
    corpus is never re-segmented or re-encoded. Rerank still needs the raw docs in
    memory (looked up by cid), so those are reattached from the already-loaded CSV.
    """
    index_dir = Path(index_dir)
    manifest = index_dir / "manifest.json"
    key = _manifest_key(corpus_path, pipeline.retriever, len(documents))

    if manifest.exists():
        try:
            if json.loads(manifest.read_text()).get("key") == key:
                pipeline.retriever.load(index_dir)
                pipeline._documents, pipeline._cids = list(documents), list(cids)
                return True
        except (OSError, KeyError, ValueError, FileNotFoundError):
            pass  # corrupt/missing artifacts → fall through to a clean rebuild

    pipeline.index(documents, cids)  # sets pipeline._documents/_cids itself
    index_dir.mkdir(parents=True, exist_ok=True)
    pipeline.retriever.save(index_dir)
    manifest.write_text(json.dumps({"key": key}))
    return False
