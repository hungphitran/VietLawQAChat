"""FastAPI app — Vietnamese Legal RAG chat.

Loads the single best hybrid pipeline + corpus at startup (lifespan), then
serves a thin chat UI and a JSON `/chat` endpoint backed by the agentic RAG.

Run:  python scripts/serve_app.py   (or) uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import os
import platform
# Avoid a flaky OpenMP/torch/MPS segfault on macOS conda (sklearn + torch both
# bundle libomp). Linux/CUDA keeps full threading. Must precede torch import.
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.index_cache import load_or_build_index
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation as CitationOut,
    RetrievedDoc as RetrievedDocOut,
    Round as RoundOut,
)
from vnlegal_rag_v2.factory import build_pipeline
from vnlegal_rag_v2.rag.agent import AgenticRAG
from vnlegal_rag_v2.rag.llm import LLMClient

STATIC_DIR = Path(__file__).resolve().parent / "static"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # silence the transformers "overflowing tokens ... longest_first" advisory —
    # the cross-encoder emits it once per rerank batch (~40x per question). Same
    # set_verbosity_error used in tune_hybrid.
    import transformers
    transformers.logging.set_verbosity_error()

    cfg = load_config()
    agent_cfg = cfg.get("agent", {})

    print(f"[app] loading corpus: {cfg['corpus_path']}")
    df = pd.read_csv(cfg["corpus_path"])
    documents, cids = df["text"].astype(str).tolist(), df["cid"].tolist()
    print(f"[app] corpus: {len(documents):,} docs")

    print(f"[app] building pipeline: {cfg['pipeline_config']}")
    with open(cfg["pipeline_config"]) as f:
        pipeline = build_pipeline(yaml.safe_load(f))

    print("[app] indexing (one-time, may take a while on full corpus)…")
    index_dir = cfg.get("index_dir")
    if index_dir:
        hit = load_or_build_index(pipeline, documents, cids, cfg["corpus_path"], index_dir)
        print(f"[app] index: {'cache hit (loaded from disk)' if hit else 'built + cached'} → {index_dir}")
    else:
        pipeline.index(documents, cids)

    llm_cfg = cfg.get("llm", {})
    llm = LLMClient(base_url=llm_cfg.get("base_url"), model=llm_cfg.get("model"))
    print(f"[app] LLM: {llm.model} @ {llm.client.base_url}")

    _state["agent"] = AgenticRAG(
        pipeline=pipeline,
        llm=llm,
        cid2text=dict(zip(cids, documents)),
        max_rounds=agent_cfg.get("max_rounds", 2),
        top_k_retrieval=agent_cfg.get("top_k_retrieval", 100),
        top_k_rerank=agent_cfg.get("top_k_rerank", 10),
        context_cap=agent_cfg.get("context_cap", 15),
    )
    _state["ready"] = True
    _state["n_docs"] = len(documents)
    print("[app] ready ✓")
    yield
    _state.clear()


app = FastAPI(title="Vietnamese Legal RAG", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "ready": _state.get("ready", False), "n_docs": _state.get("n_docs", 0)}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # browsers auto-request this; return 204 so the log stays quiet (no icon file shipped)
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not _state.get("ready"):
        return ChatResponse(
            answer="Hệ thống đang khởi động, vui lòng thử lại sau.",
            sufficient=False, citations=[], rounds=[], missing="hệ thống chưa sẵn sàng",
        )
    res = _state["agent"].answer(req.question)
    return ChatResponse(
        answer=res.answer,
        sufficient=res.sufficient,
        citations=[CitationOut(cid=c.cid, section=c.section, quote=c.quote) for c in res.citations],
        rounds=[RoundOut(
            query=r.query,
            docs=[RetrievedDocOut(cid=d.cid, text=d.text, rank=d.rank) for d in r.docs],
        ) for r in res.rounds],
        missing=res.missing,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat: emits the agent's events as SSE (text/event-stream).

    The agent's `answer_stream` is a blocking sync generator; Starlette runs it
    in a threadpool inside StreamingResponse, so blocking retrieve/LLM calls
    stay off the event loop — same model as the sync /chat above.
    """
    if not _state.get("ready"):
        async def _not_ready():
            yield _sse({"type": "error", "message": "hệ thống đang khởi động, vui lòng thử lại sau"})
        return StreamingResponse(_not_ready(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def sse_gen():
        try:
            for ev in _state["agent"].answer_stream(req.question):
                yield _sse(ev)
        except Exception as e:  # surface agent errors as a final SSE event
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(ev: dict) -> str:
    """Format one event dict as a single SSE frame."""
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
