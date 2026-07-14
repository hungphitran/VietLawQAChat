"""Simple agentic RAG: retrieve → generate → (if insufficient) refine query and repeat.

The "agent" is deliberately minimal — no agent framework, just a loop:
  1. retrieve top-K (reranked) candidates via the existing hybrid RAGPipeline
  2. ask the LLM to answer in STRICT JSON, citing documents by cid + section
  3. if the LLM reports insufficient info and rounds remain, reformulate the
     query and retrieve again; context ACCUMULATES across rounds (dedup by cid)

Each round records the query used + its top-K docs, so the UI can show what
each query retrieved. The final answer carries structured citations.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .llm import LLMClient

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam. Trả lời câu hỏi CHỈ dựa trên ngữ cảnh được cung cấp.\n"
    "Quy tắc:\n"
    "- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.\n"
    "- Chỉ trích dẫn cid THỰC SỬ trong ngữ cảnh; không bịa cid.\n"
    "- Mỗi nhận định quan trọng phải có ít nhất một trích dẫn (cid + điều/khoản liên quan).\n"
    "- Chỉ liệt kê tối đa 3 trích dẫn quan trọng NHẤT (cid + điều/khoản).\n"
    "- Nếu ngữ cảnh KHÔNG ĐỦ, đặt sufficient=false và nêu rõ còn thiếu gì ở trường missing.\n"
    "- Tuyệt đối không bịa đặt thông tin ngoài ngữ cảnh.\n\n"
    "Phản hồi BẮT BUỘC là JSON duy nhất theo đúng định dạng (KHÔNG kèm văn bản trích, chỉ cid):\n"
    "{\n"
    '  "answer": "câu trả lời tổng quan bằng tiếng Việt",\n'
    '  "sufficient": true|false,\n'
    '  "missing": "rỗng nếu sufficient=true, ngược lại mô tả thông tin còn thiếu",\n'
    '  "citations": [\n'
    '    {"cid": <số nguyên>, "section": "Điều/Khoản hoặc mô tả ngắn"}\n'
    "  ]\n"
    "}"
)

CONTEXT_HEADER = "Ngữ cảnh (cid là mã tài liệu, hãy trích dẫn theo cid):\n"
CONTEXT_ENTRY = "[{rank}] (cid {cid}) {text}"
CONTEXT_TEMPLATE = "{header}{context}\n\nCâu hỏi: {question}"

# --- streaming-phase prompts (decide loop, streamed answer, citation extraction) ---

DECIDE_PROMPT = (
    "Bạn đánh giá xem ngữ cảnh đã ĐỦ để trả lời câu hỏi gốc chưa.\n"
    "Trả về JSON duy nhất, không kèm văn bản khác:\n"
    "{\n"
    '  "sufficient": true|false,\n'
    '  "missing": "rỗng nếu đủ, ngược lại mô tả ngắn thông tin còn thiếu",\n'
    '  "next_query": "nếu chưa đủ, MỘT truy vấn tìm kiếm ngắn bằng tiếng Việt để tìm bổ sung; rỗng nếu đã đủ"\n'
    "}"
)

ANSWER_PROMPT = (
    "Bạn là trợ lý pháp luật Việt Nam. Trả lời câu hỏi CHỈ dựa trên ngữ cảnh (mỗi tài liệu có cid).\n"
    "Quy tắc:\n"
    "- Trả lời bằng tiếng Việt, rõ ràng, đầy đủ ý, dùng markdown hợp lý (tiêu đề ###, gạch đầu dòng).\n"
    "- CHỈ ĐƯỢC trích dẫn cid CÓ TRONG Ngữ cảnh. cid nào không xuất hiện trong Ngữ cảnh = bịa → BỊ CẤM tuyệt đối. "
    "Trước khi viết [cid:N], hãy kiểm N có mặt trong Ngữ cảnh.\n"
    "- Mỗi nhận định quan trọng PHẢI kèm [cid:<số>] với cid THỰC SỰ HỖ TRỢ cho nhận định đó (1–3 cid).\n"
    "- Khi một nhận định dẫn nhiều cid, viết gộp: [cid:1, 2, 3].\n"
    "- KHÔNG liệt kê toàn bộ cid đã truy xuất; chỉ cid thực sự hỗ trợ nhận định. "
    "Nếu Ngữ cảnh KHÔNG liên quan đến câu hỏi, KHÔNG thêm cid nào cả.\n"
    "- Tổng số cid khác nhau trong toàn bài nên ≤ 5. Thà ít mà đúng còn hơn nhiều mà bịa.\n"
    "- Bôi đậm các nhận định/khoản quan trọng bằng **...** (markdown bold).\n"
    "- Tuyệt đối không bịa cid hay thông tin ngoài ngữ cảnh.\n"
    "- KHÔNG trả về JSON; chỉ trả về câu trả lời bằng tiếng Việt."
)

EXTRACT_PROMPT = (
    "Dựa vào câu trả lời và ngữ cảnh, liệt kê các trích dẫn hỗ trợ quan trọng nhất (tối đa 5).\n"
    "Chỉ trả về cid THỰC SỰ xuất hiện trong CẢ câu trả lời LẪN Ngữ cảnh. cid không có ở đâu cả = bịa, bỏ qua.\n"
    "Trả về JSON duy nhất:\n"
    "{\n"
    '  "citations": [\n'
    '    {"cid": <số nguyên cid thực sự>, "section": "Điều/Khoản hoặc mô tả ngắn",\n'
    '     "quote": "đoạn văn bản CHÍNH XÁC, copy y nguyên từ tài liệu cid trong Ngữ cảnh (không suy luận, không bịa)"}\n'
    "  ]\n"
    "}"
)

# matches [cid:19], [cid: 19, 69], [cid 19, 69], [cid19] — colon/space/none tolerant
_CID_RE = re.compile(r"\[cid[:\s]*([\d\s,]+)\]")


@dataclass
class RetrievedDoc:
    cid: int
    text: str
    rank: int


@dataclass
class Round:
    query: str
    docs: list[RetrievedDoc]


@dataclass
class Citation:
    cid: int
    section: str
    quote: str


@dataclass
class AgentResponse:
    answer: str
    sufficient: bool
    citations: list[Citation]
    rounds: list[Round]
    missing: str | None = None


class AgenticRAG:
    def __init__(
        self,
        pipeline,
        llm: LLMClient,
        cid2text: dict[int, str],
        max_rounds: int = 2,
        top_k_retrieval: int = 100,
        top_k_rerank: int = 10,
        context_cap: int = 15,
    ):
        self.pipeline = pipeline
        self.llm = llm
        self.cid2text = cid2text
        self.max_rounds = max(1, max_rounds)
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.context_cap = context_cap

    def answer(self, question: str) -> AgentResponse:
        """Non-streaming answer: retrieve → generate → (if insufficient) reformulate & repeat.

        Retrieved docs accumulate across rounds (dedup by cid, insertion-ordered). Stops when
        the LLM reports `sufficient=true` or `max_rounds` is reached.
        """
        rounds: list[Round] = []
        seen: dict[int, str] = {}  # cid -> text, insertion-ordered
        query = question
        answer, sufficient, citations, missing = "", False, [], None

        for i in range(self.max_rounds):
            cids = self.pipeline.query([query], self.top_k_retrieval, self.top_k_rerank)[0]

            rounds.append(Round(
                query=query,
                docs=[RetrievedDoc(cid=c, text=self.cid2text.get(c, ""), rank=r)
                      for r, c in enumerate(cids[: self.top_k_rerank], 1)],
            ))
            for c in cids[: self.top_k_retrieval]:
                seen.setdefault(c, self.cid2text.get(c, ""))

            context = self._build_context(seen)
            answer, sufficient, citations, missing = self._generate(question, context)

            if sufficient or i + 1 >= self.max_rounds:
                break
            query = (self.llm.reformulate(question, missing or answer) or question).strip()

        return AgentResponse(
            answer=answer,
            sufficient=sufficient,
            citations=citations,
            rounds=rounds,
            missing=missing,
        )

    # --- streaming variant (POST /chat/stream): yields event dicts, then streams the answer ---

    def answer_stream(self, question: str) -> Iterator[dict]:
        """Streaming variant of `answer()`: yields SSE-ready event dicts so the UI can render
        the agent's reasoning live (search/analyze/refine tool calls, per-doc reveals, then
        the token-streamed answer + citations). Same multi-round loop, same context accumulation."""
        seen: dict[int, str] = {}  # cid -> text, insertion-ordered (accumulates across rounds)
        query = question
        decide: dict = {"sufficient": False, "missing": "", "next_query": ""}
        n_rounds = 0

        for i in range(self.max_rounds):
            n_rounds = i + 1
            yield {"type": "step", "round": n_rounds}
            yield {"type": "tool_call", "tool": "search", "input": {"query": query}}

            cids = self.pipeline.query([query], self.top_k_retrieval, self.top_k_rerank)[0]
            for rank, c in enumerate(cids[: self.top_k_rerank], 1):
                seen.setdefault(c, self.cid2text.get(c, ""))
                time.sleep(0.03)  # progressive reveal of the ranked list (faiss is batch)
                yield {"type": "result", "round": n_rounds,
                       "doc": {"cid": c, "rank": rank, "text": self.cid2text.get(c, "")}}

            yield {"type": "tool_call", "tool": "analyze"}
            context = self._build_context(seen)
            decide = self._decide(question, query, context)
            yield {"type": "tool_result", "tool": "analyze",
                   "sufficient": decide["sufficient"], "missing": decide["missing"]}

            if decide["sufficient"] or i + 1 >= self.max_rounds:
                break
            query = (decide["next_query"] or query).strip()
            yield {"type": "tool_call", "tool": "refine", "input": {"query": query}}

        # final synthesis: stream the prose answer, then attach citations
        context = self._build_context(seen)
        context_cids = list(seen.keys())[: self.context_cap]
        valid_cids = set(context_cids)
        # client uses this allow-list to strip any [cid:N] the LLM hallucinated
        yield {"type": "answer_start", "context_cids": context_cids}
        answer = ""
        for delta in self._answer_stream(question, context):
            answer += delta
            yield {"type": "answer_delta", "text": delta}

        # drop any hallucinated cids before extraction (LLM may cite outside the context)
        cited = [c for c in self._cited_cids(answer) if c in valid_cids]
        citations = self._extract_citations(question, answer, context, valid_cids) if cited else []
        yield {"type": "citations", "citations": [
            {"cid": c.cid, "section": c.section, "quote": c.quote} for c in citations]}
        yield {"type": "done", "sufficient": decide["sufficient"], "n_rounds": n_rounds}

    @staticmethod
    def _cited_cids(answer: str) -> list[int]:
        """cids referenced by [cid:N] / [cid:N, M, ...] markers, in first-seen order."""
        seen: list[int] = []
        for m in _CID_RE.finditer(answer):
            for tok in re.split(r"[,\s]+", m.group(1)):
                if tok.isdigit():
                    c = int(tok)
                    if c not in seen:
                        seen.append(c)
        return seen

    def _decide(self, question: str, query: str, context: str) -> dict:
        """Ask the LLM whether the current context sufficiently answers the question.

        Returns {"sufficient": bool, "missing": str, "next_query": str}.
        Used by the multi-round loop to decide whether to stop or reformulate.
        """
        data = self.llm.chat_json(
            [
                {"role": "system", "content": DECIDE_PROMPT},
                {"role": "user", "content": f"{context}\n\nCâu hỏi gốc: {question}\nTruy vấn vừa tìm: {query}"},
            ],
            max_tokens=256,
        )
        return {
            "sufficient": bool(data.get("sufficient", False)),
            "missing": str(data.get("missing", "") or ""),
            "next_query": str(data.get("next_query", "") or ""),
        }

    def _answer_stream(self, question: str, context: str) -> Iterator[str]:
        """Stream token deltas of the final answer from the LLM.

        Used by the streaming endpoint (`answer_stream()`) after all retrieval
        rounds are done. Each yield is one text delta for SSE forwarding.
        """
        yield from self.llm.chat_stream(
            [
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"{context}\n\nCâu hỏi: {question}"},
            ],
            max_tokens=2048,
        )

    def _extract_citations(
        self, question: str, answer: str, context: str, valid_cids: set[int]
    ) -> list[Citation]:
        """Parse LLM-generated answer into structured citations.

        Each cited cid is validated against `valid_cids` (kills hallucinations).
        Each quote is checked verbatim in the source doc; falls back to the doc head
        if the quote doesn't match (avoids broken <mark> highlights on the frontend).
        """
        data = self.llm.chat_json(
            [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"{context}\n\nCâu hỏi: {question}\nCâu trả lời:\n{answer}"},
            ],
            max_tokens=1024,
        )
        out: list[Citation] = []
        for c in data.get("citations", []):
            try:
                cid_i = int(c["cid"]) if c.get("cid") is not None else None
            except (TypeError, ValueError):
                continue
            # hard guard: only cids in the LLM's context are eligible (kills hallucinated cids)
            if cid_i is None or cid_i not in valid_cids:
                continue
            doc_text = self.cid2text.get(cid_i, "")
            quote = str(c.get("quote", "") or "").strip()
            # only trust the quote if it actually appears verbatim in the source doc;
            # otherwise the <mark> highlight on the frontend silently no-ops. Fall back to
            # the head of the doc so the popup still shows the real paragraph.
            if quote and quote not in doc_text:
                head = quote[:40]
                idx = doc_text.find(head)
                quote = doc_text[idx:idx + 200] if idx >= 0 else doc_text[:200]
            elif not quote:
                quote = doc_text[:200]
            out.append(Citation(
                cid=cid_i,
                section=str(c.get("section", "") or ""),
                quote=quote,
            ))
        return out

    def _build_context(self, seen: dict[int, str]) -> str:
        """Assemble retrieved docs into the LLM prompt context block.

        Documents are numbered by rank (insertion order). Capped at `context_cap`.
        Uses CONTEXT_HEADER + CONTEXT_ENTRY templates from prompts.py.
        """
        items = list(seen.items())[: self.context_cap]
        body = "\n\n".join(
            CONTEXT_ENTRY.format(rank=r, cid=cid, text=txt)
            for r, (cid, txt) in enumerate(items, 1)
        )
        return CONTEXT_HEADER + body

    def _generate(self, question: str, context: str):
        """One-shot JSON generation (non-streaming path): ask the LLM for answer + citations,
        then re-validate each cited cid against the real corpus (drops hallucinated cids) and
        resolve the supporting quote server-side instead of trusting the LLM's snippet."""
        data = self.llm.chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CONTEXT_TEMPLATE.format(header="", context=context, question=question)},
        ], max_tokens=2048)
        valid = set(self.cid2text)
        citations = []
        for c in data.get("citations", []):
            cid = c.get("cid")
            if cid is None:
                continue
            try:
                cid_i = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_i in valid:
                # resolve the supporting quote server-side from the corpus
                # (we keep it out of the LLM JSON to avoid breakage/truncation)
                snippet = self.cid2text[cid_i][:300]
                citations.append(
                    Citation(cid=cid_i, section=str(c.get("section", "")), quote=snippet)
                )
        return (
            str(data.get("answer", "")),
            bool(data.get("sufficient", False)),
            citations,
            str(data["missing"]) if data.get("missing") else None,
        )
