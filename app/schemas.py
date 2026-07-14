"""Request/response models for the chat API.

Mirrors the agentic-RAG dataclasses, but as Pydantic models so FastAPI
serializes them to clean JSON for the frontend.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    cid: int
    text: str
    rank: int


class Round(BaseModel):
    query: str
    docs: list[RetrievedDoc]


class Citation(BaseModel):
    cid: int
    section: str
    quote: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Câu hỏi pháp luật của người dùng")


class ChatResponse(BaseModel):
    answer: str
    sufficient: bool
    citations: list[Citation]
    rounds: list[Round]
    missing: str | None = None
