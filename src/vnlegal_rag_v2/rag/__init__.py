from __future__ import annotations

"""Agentic RAG layer — retrieval (existing pipeline) + LLM generation loop.

No FastAPI dependency here; this is reusable library code. The app layer
(`app/`) wraps this. Designed for ONE fixed pipeline (the best config) —
no model selection, kept simple.
"""
from .agent import AgentResponse, AgenticRAG, Citation, RetrievedDoc, Round
from .llm import LLMClient

__all__ = ["AgenticRAG", "AgentResponse", "LLMClient", "Citation", "RetrievedDoc", "Round"]
