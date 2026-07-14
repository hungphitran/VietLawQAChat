"""OpenAI-compatible chat client — one SDK, many providers.

Defaults to OpenRouter. Swap provider at runtime via env vars, no code change:
  - OpenRouter : LLM_BASE_URL=https://openrouter.ai/api/v1   (default)
  - OpenAI     : LLM_BASE_URL=https://api.openai.com/v1
  - Groq       : LLM_BASE_URL=https://api.groq.com/openai/v1
  - Together   : LLM_BASE_URL=https://api.together.xyz/v1
  - Ollama     : LLM_BASE_URL=http://localhost:11434/v1        (local, no key)
  - vLLM       : LLM_BASE_URL=http://localhost:8000/v1         (local, no key)

Set LLM_API_KEY (or OPENAI_API_KEY) and LLM_MODEL accordingly.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"


class LLMClient:
    """Thin wrapper over the OpenAI SDK speaking any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "none"),
        )
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> str:
        """Call the LLM and return the response text.

        Always sends `extra_body={"reasoning": {"enabled": False}}` to suppress
        chain-of-thought tokens that would break JSON parsing downstream.
        Pass `response_format` for structured JSON output (OpenAI-compatible only).
        """
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # "No think": disable chain-of-thought/reasoning tokens. Required by our
            # JSON agent (a <think> block before the JSON would break parsing).
            # OpenRouter honours `reasoning`; other providers ignore unknown body
            # params, so this is safe everywhere.
            extra_body={"reasoning": {"enabled": False}},
        )
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Like chat() but streams token deltas (no JSON mode).

        Used for the live final answer. Same reasoning-off extra_body as
        chat() so the first token isn't a <think> block.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"reasoning": {"enabled": False}},
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat_json(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> dict:
        """Like chat() but parses the reply as JSON.

        Prefers server-enforced JSON (`response_format: json_object`) so the
        provider guarantees valid output — this prevents the common LLM failure
        of unescaped quotes inside legal-text citation fields. Falls back to a
        plain call if the provider rejects json_object, then robustly parses.
        """
        try:
            raw = self.chat(
                messages, temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return parse_json(raw)

    def reformulate(self, question: str, missing: str) -> str:
        """Write a tighter search query to fill the missing information."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tạo truy vấn tìm kiếm pháp luật. Dựa trên câu hỏi gốc và thông tin "
                    "còn thiếu, viết lại thành MỘT truy vấn tìm kiếm ngắn gọn bằng tiếng Việt để tìm "
                    "bổ sung. Chỉ trả về truy vấn, không giải thích."
                ),
            },
            {"role": "user", "content": f"Câu hỏi gốc: {question}\nThông tin còn thiếu: {missing}"},
        ]
        return self.chat(messages, temperature=0.0, max_tokens=128)


def _close_truncated(text: str) -> str:
    """Best-effort repair for JSON truncated by max_tokens.

    Rewinds to the last safe boundary (after a ',' or a closing '}'/']')
    then appends the closers needed to balance open brackets. Returns text
    unchanged if no safe boundary exists. Truncated string values are handled
    by rewinding before them.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    safe = -1  # index+1 of the last safe (balanced) boundary
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = i + 1
        elif ch == ",":
            safe = i + 1
    if safe < 0:
        return text
    text = re.sub(r"[\s,]+$", "", text[:safe])
    closers = {"{": "}", "[": "]"}
    while stack:
        text += closers[stack.pop()]
    return text


def _dump_debug(text: str, err: Exception, tag: str = "raw"):
    """Write the failing JSON payload to /tmp for diagnosis (no-op if disabled)."""
    if os.getenv("LLM_DEBUG"):
        import pathlib
        pathlib.Path(f"/tmp/llm_json_fail_{tag}.txt").write_text(
            f"ERR: {err}\nLEN: {len(text)}\n===\n{text}\n===\n",
            encoding="utf-8",
        )


def parse_json(text: str) -> dict:
    """Extract and parse a JSON object from an LLM reply.

    Handles ```json fences, prose wrapping, smart quotes, and trailing commas.
    Tries strict JSON first, then a lenient repair pass. For genuinely broken
    output (e.g. unescaped inner quotes) callers should prefer server-enforced
    JSON mode (`LLMClient.chat_json`).
    """
    text = text.strip()
    # strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # fall back to the outermost { ... }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        _dump_debug(text, e)
    # lenient repair: smart quotes -> straight, drop trailing commas
    repaired = text
    for smart, plain in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]:
        repaired = repaired.replace(smart, plain)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        _dump_debug(repaired, e, tag="repaired")
    # last resort: the model may have been truncated mid-output (hit max_tokens).
    # Best-effort: rewind to the last complete value, then auto-close open
    # brackets/strings. Safe because the agent validates cids afterward.
    closed = _close_truncated(repaired)
    if closed != repaired:
        return json.loads(closed)
    raise
