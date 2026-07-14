// Streaming agentic-RAG UI. Reads SSE from POST /chat/stream, renders:
//  - a live "thinking process" trace (search/analyze/refine tool calls) in a sidebar
//  - the streamed final answer (markdown via marked.js) with [cid:N] citation badges
//  - a Sources panel with LLM-highlighted paragraphs bolded (<mark>)

const messagesEl = document.getElementById("messages");
const statusEl = document.getElementById("status");
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const aside = document.getElementById("thinking");
const traceHost = document.getElementById("trace");

// global cid -> {text, section, quote} registry, fed by `result` + `citations`
// events so any [cid] badge (even outside the capped Sources panel) can open a popup.
const cidInfo = {};

// --- tiny DOM + escape helpers ---
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
function esc(s) {
  return (s == null ? "" : String(s)).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// --- markdown answer: marked.js, then turn [cid:N] markers into citation badges ---
// validCids (optional Set of cid strings) = the cids the LLM actually saw in its context.
// Any [cid:N] whose N is NOT in validCids is a hallucination → silently dropped so no
// broken/unclickable badge ever renders.
function renderMarkdown(text, cit, validCids) {
  let html;
  if (typeof marked !== "undefined") {
    try { html = marked.parse(text || ""); }
    catch { html = "<p>" + esc(text || "") + "</p>"; }
  } else {
    html = "<p>" + esc(text || "") + "</p>";
  }
  html = html.replace(/\[cid[:\s]*([\d\s,]+)\]/g, (m, list) =>
    list.split(/[,\s]+/).filter(Boolean).map((cid) => {
      if (validCids && validCids.size > 0 && !validCids.has(cid)) return "";
      const c = cit && cit[cid];
      const title = c ? `${c.section}\n“${c.quote}”` : `cid ${cid}`;
      return `<a class="cite" data-cid="${cid}" title="${esc(title)}">[${cid}]</a>`;
    }).join("")
  );
  return html;
}

// bold LLM-highlighted spans inside a source document's text
function highlightText(text, highlights) {
  let html = esc(text);
  (highlights || []).forEach((h) => {
    if (!h) return;
    const eh = esc(h);
    if (html.includes(eh)) html = html.split(eh).join(`<mark>${eh}</mark>`);
  });
  return html;
}

// --- thinking-process sidebar ---
let activeTrace = null;
function sidebarOpen() { return aside.classList.contains("open"); }
function openSidebar() { aside.classList.add("open"); aside.setAttribute("aria-hidden", "false"); }
function closeSidebar() { aside.classList.remove("open"); aside.setAttribute("aria-hidden", "true"); }
function showTrace(node) {
  activeTrace = node;
  traceHost.innerHTML = "";
  traceHost.appendChild(node);
  openSidebar();
}
document.getElementById("thinking-close").onclick = closeSidebar;
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { if (!modal.hidden) closeCiteModal(); else closeSidebar(); }
});

// click a [cid] badge → open a popup overlay with that paragraph's full text
const modal = document.getElementById("cite-modal");
function openCiteModal(cid) {
  const info = cidInfo[cid] || {};
  modal.querySelector(".modal-cid").textContent = `cid ${cid}`;
  modal.querySelector(".modal-section").textContent = info.section || "";
  const body = modal.querySelector(".modal-body");
  body.innerHTML = info.text
    ? highlightText(info.text, info.quote ? [info.quote] : [])
    : `(chưa có nội dung cho cid ${cid})`;
  modal.hidden = false;
}
function closeCiteModal() { modal.hidden = true; }
modal.addEventListener("click", (e) => { if (e.target.hasAttribute("data-close")) closeCiteModal(); });

messagesEl.addEventListener("click", (e) => {
  const a = e.target.closest(".cite");
  if (!a || !a.dataset.cid) return;
  openCiteModal(a.dataset.cid);
});

function setStatus(t, cls) { statusEl.textContent = t; statusEl.className = `status ${cls || ""}`; }
function scrollDown() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function scrollTrace() { traceHost.scrollTop = traceHost.scrollHeight; }

// --- per-question assistant message + its streaming state ---
function newMsg(question) {
  const wrap = el("div", "msg assistant");
  const tools = el("div", "answer-tools");
  const traceBtn = el("button", "trace-btn", "💭 Quá trình suy nghĩ");
  tools.appendChild(traceBtn);
  const bubble = el("div", "bubble md");
  bubble.innerHTML = '<span class="typing-dots"><i></i><i></i><i></i></span>';
  const meta = el("div", "meta-row");
  const sources = el("div", "sources");
  wrap.append(tools, bubble, meta, sources);
  messagesEl.appendChild(wrap);

  const s = {
    question, wrap, bubble, meta, sources,
    traceEl: el("div", "trace-root"),
    answerBuf: "", cit: {}, cards: {},
    rounds: [], curBody: null, pending: {},
    contextCids: null,  // set of cid strings the LLM saw; null until answer_start fires
  };
  traceBtn.onclick = () => showTrace(s.traceEl);
  return s;
}

function toolItem(ic, html) {
  const it = el("div", "tool-item");
  it.appendChild(el("span", "ic", ic));
  it.appendChild(el("span", "txt", html));
  return it;
}

// --- SSE event handling ---
function handleEvent(ev, s) {
  switch (ev.type) {
    case "step": {
      s.rounds.forEach((r) => r.classList.remove("open")); // collapse completed rounds
      const card = el("div", "round open");
      const head = el("div", "round-head", `▶ Vòng ${ev.round}`);
      const body = el("div", "round-body");
      head.onclick = () => card.classList.toggle("open");
      card.append(head, body);
      s.traceEl.appendChild(card);
      s.rounds.push(card);
      s.curBody = body;
      setStatus("đang tìm kiếm tài liệu…", "typing");
      break;
    }
    case "tool_call": {
      if (ev.tool === "search")
        s.curBody.appendChild(toolItem("🔍", `Tìm kiếm: <span class="q">“${esc(ev.input.query)}”</span>`));
      else if (ev.tool === "analyze") {
        const it = toolItem("🧠", "Phân tích kết quả…");
        s.curBody.appendChild(it);
        s.pending.analyze = it;
      } else if (ev.tool === "refine")
        s.curBody.appendChild(toolItem("✏️", `Đặt lại truy vấn: <span class="q">“${esc(ev.input.query)}”</span>`));
      break;
    }
    case "result": {
      const t = ev.doc.text || "";
      cidInfo[ev.doc.cid] = { ...(cidInfo[ev.doc.cid] || {}), text: t };
      const preview = t.length > 200 ? t.slice(0, 200) + "…" : t;
      s.curBody.appendChild(el("div", "doc",
        `<span class="rank">#${ev.doc.rank}</span><span class="cid">cid ${ev.doc.cid}</span><span class="preview">${esc(preview)}</span>`));
      break;
    }
    case "tool_result": {
      if (ev.tool === "analyze" && s.pending.analyze) {
        const it = s.pending.analyze;
        it.classList.add(ev.sufficient ? "ok" : "bad");
        it.querySelector(".ic").textContent = ev.sufficient ? "✅" : "🧠";
        const tag = el("span", `tag ${ev.sufficient ? "ok" : "bad"}`,
          ev.sufficient ? "Đủ thông tin" : "Chưa đủ");
        if (!ev.sufficient && ev.missing) tag.title = ev.missing;
        it.appendChild(tag);
      }
      break;
    }
    case "answer_start": {
      s.rounds.forEach((r) => r.classList.remove("open")); // collapse all search steps
      s.contextCids = new Set((ev.context_cids || []).map(String));
      s.bubble.innerHTML = "";
      setStatus("đang viết câu trả lời…", "typing");
      break;
    }
    case "answer_delta": {
      s.answerBuf += ev.text;
      s.bubble.innerHTML = renderMarkdown(s.answerBuf, s.cit, s.contextCids) + '<span class="caret"></span>';
      break;
    }
    case "citations": {
      applyCitations(s, ev.citations);
      break;
    }
    case "sources": {
      break; // sources panel removed — message ends at the answer; [cid] badges open popups
    }
    case "done": {
      s.bubble.innerHTML = renderMarkdown(s.answerBuf, s.cit, s.contextCids);
      s.bubble.classList.add("done");
      setStatus("");
      break;
    }
    case "error": {
      s.bubble.innerHTML = `⚠️ ${esc(ev.message || "Lỗi xử lý")}`;
      s.bubble.classList.add("error");
      setStatus(`lỗi: ${ev.message || ""}`, "error");
      break;
    }
  }
  scrollDown();
  if (sidebarOpen() && activeTrace === s.traceEl) scrollTrace();
}

// extraction populates cidInfo (section label + exact quote span) for the inline
// [cid] click-popups; there's no Sources panel, so it only re-renders the answer.
function applyCitations(s, citations) {
  (citations || []).forEach((c) => {
    s.cit[c.cid] = c;
    cidInfo[c.cid] = { ...(cidInfo[c.cid] || {}), section: c.section, quote: c.quote };
    const entry = s.cards[c.cid];
    if (!entry) return;
    const sec = entry.head.querySelector(".section");
    if (sec && c.section) sec.textContent = c.section;
    if (c.quote) entry.body.innerHTML = highlightText(entry.body.textContent, [c.quote]);
    entry.card.classList.add("cited");
  });
  if (s.answerBuf) s.bubble.innerHTML = renderMarkdown(s.answerBuf, s.cit, s.contextCids) + '<span class="caret"></span>';
}

// --- stream reader: fetch POST + ReadableStream, split SSE frames on blank line ---
async function streamAnswer(question, s) {
  const res = await fetch("/chat/stream", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder("utf-8");
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      handleEvent(JSON.parse(line.slice(5).trim()), s);
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  const uwrap = el("div", "msg user");
  uwrap.appendChild(el("div", "bubble", esc(question)));
  messagesEl.appendChild(uwrap);

  input.value = "";
  sendBtn.disabled = true;
  const s = newMsg(question);
  if (sidebarOpen()) showTrace(s.traceEl); // split sidebar open: watch this question live
  scrollDown();
  try {
    await streamAnswer(question, s);
  } catch (err) {
    if (!s.bubble.classList.contains("error")) {
      s.bubble.innerHTML = "⚠️ Lỗi kết nối tới máy chủ.";
      setStatus(`lỗi: ${err.message}`, "error");
    }
  } finally {
    sendBtn.disabled = false;
    input.focus();
    scrollDown();
  }
});

fetch("/health").then((r) => r.json()).then((h) => {
  if (!h.ready) setStatus("hệ thống đang khởi động (đánh chỉ mục kho văn bản)…", "typing");
});
