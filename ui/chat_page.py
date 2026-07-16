"""Chat page: the agent conversation, citation cards, and reasoning trace."""

import time
import uuid

import streamlit as st

from agent import build_direct_prompt, build_generation_prompt
from llm_utils import stream_without_thinking

from .styles import CITATION_ICONS, CITATION_TYPE_TO_TOOL, TOOL_COLORS, TOOL_ICONS, escape_html, safe_url


def render_trace(trace: list[str]):
    """
    Trace step text is built from f-strings that can embed LLM output
    (a rewritten search query, a one-sentence grading rationale) which
    in turn can be influenced by adversarial content in a retrieved
    document or web page (prompt injection trying to get the model to
    echo raw HTML) -- escaped before interpolation into raw HTML for
    the same reason as render_citations below.
    """
    icons = {
        "analyze_query": "🧭", "retrieve": "📚", "rerank": "🎯",
        "grade_documents": "✅", "rewrite_query": "✏️", "query_specs_db": "🗄️",
        "web_search": "🌐", "gather_other_tools": "🔧", "generate": "💬",
    }
    for step in trace:
        node = step.split(" -> ")[0].split("(")[0]
        st.markdown(
            f'<div class="trace-step">{icons.get(node, "•")} <code>{escape_html(step)}</code></div>',
            unsafe_allow_html=True,
        )


def render_citations(citations: list[dict]):
    """
    Citation cards sorted by their pre-assigned number, matching the
    inline [1], [2], [3]... order used in the answer text -- a reader
    can go straight from "[3]" in the answer to the third card here,
    instead of hunting across type-grouped sections. Color-coding by
    source type is kept per-card (docs=blue, specs=purple, web=green) so
    source authority stays visually distinguishable even in a flat list.

    SECURITY: source/snippet/section/URL are NOT trusted -- web
    citations carry a title/URL/snippet straight from live Tavily
    results (fully attacker-controlled if a malicious page gets cited),
    and document citations carry a section heading/snippet from corpus
    files the Documents tab lets anyone add. Confirmed exploitable prior
    to this fix: these were interpolated directly into
    unsafe_allow_html=True markdown with no escaping, so a crafted page
    title or document heading (e.g. containing a <script> tag) executed
    in the viewer's browser the moment that source got cited. Every
    value below goes through escape_html/safe_url first.
    """
    sorted_citations = sorted(citations, key=lambda c: c.get("number", 0))

    for c in sorted_citations:
        ctype = c.get("type", "document")
        tool = CITATION_TYPE_TO_TOOL.get(ctype, "docs")
        color = TOOL_COLORS.get(tool, "#999")
        icon = CITATION_ICONS.get(ctype, "•")
        num = c.get("number", "?")
        source = escape_html(c.get("source", ""))

        if ctype == "web":
            url = safe_url(c.get("source_url", ""))
            source_html = f'<a href="{url}" target="_blank">{source}</a>' if url else source
            st.markdown(
                f'<div class="citation-card" style="--accent-color:{color}">'
                f'<div class="citation-source">[{num}] {icon} {source_html}</div>'
                f'<div class="citation-meta">web result — verify independently, not a curated source</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        elif ctype == "specs_db":
            url = safe_url(c.get("source_url", ""))
            link = f' · <a href="{url}" target="_blank">manufacturer spec sheet</a>' if url else ""
            snippet = escape_html(c.get("snippet", ""))
            st.markdown(
                f'<div class="citation-card" style="--accent-color:{color}">'
                f'<div class="citation-source">[{num}] {icon} {source}</div>'
                f'<div class="citation-meta">sourced hardware spec{link}</div>'
                f'<div>{snippet}...</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            section = escape_html(c.get("section", ""))
            snippet = escape_html(c.get("snippet", ""))
            st.markdown(
                f'<div class="citation-card" style="--accent-color:{color}">'
                f'<div class="citation-source">[{num}] {icon} {source}</div>'
                f'<div class="citation-meta">{section} — curated knowledge base</div>'
                f'<div>{snippet}...</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _stream_tokens(llm, prompt: str):
    """Generator wrapping llm.stream(), filtered through
    stream_without_thinking() so <think> reasoning never displays as if
    it were the answer (reproduced live on qwen3.6-27b)."""

    def raw_chunks():
        for chunk in llm.stream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if text:
                yield text

    yield from stream_without_thinking(raw_chunks())


def render_chat_tab(app, llm, show_trace, query, model_name):
    """
    query is passed in rather than called via st.chat_input() here --
    st.chat_input() loses its sticky-bottom-of-page positioning when
    called from inside a layout container like st.tabs() or st.columns()
    (a confirmed, documented Streamlit behavior:
    github.com/streamlit/streamlit/issues/8564), rendering inline at the
    top instead. The fix is to call st.chat_input() once at the top
    level of main(), never nested in a container, and pass its return
    value in here.
    """
    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user", avatar="🧑"):
            st.write(turn["query"])
        with st.chat_message("assistant", avatar="🔎"):
            if turn.get("model_name"):
                st.caption(f"🧠 {turn['model_name']}")
            st.write(turn["answer"])
            if turn.get("citations"):
                with st.expander(f"📎 {len(turn['citations'])} source citations"):
                    render_citations(turn["citations"])
            if show_trace and turn.get("trace"):
                with st.expander("🧠 Agent reasoning trace"):
                    render_trace(turn["trace"])

    if query:
        with st.chat_message("user", avatar="🧑"):
            st.write(query)
        with st.chat_message("assistant", avatar="🔎"):
            st.caption(f"🧠 {model_name}")
            t0 = time.time()
            with st.spinner("Routing, retrieving, and reasoning..."):
                chat_history = [
                    {"query": t["query"], "answer": t["answer"]}
                    for t in st.session_state.history
                ]
                initial_state = {
                    "original_query": query,
                    "query": query,
                    "chat_history": chat_history,
                    "tools_selected": [],
                    "retrieved_docs": [],
                    "reranked_docs": [],
                    "grade": "",
                    "grade_reasoning": "",
                    "retries": 0,
                    "specs_result": {},
                    "web_results": [],
                    "trace": [],
                    "answer": "",
                    "citations": [],
                }
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                paused_state = app.invoke(initial_state, config=config)
                prep_elapsed = time.time() - t0

            tools = paused_state.get("tools_selected", [])
            if not tools:
                final_prompt = build_direct_prompt(paused_state)
                citations = []
                generate_trace_msg = "generate -> direct answer (no tools used)"
            else:
                final_prompt, citations = build_generation_prompt(paused_state)
                generate_trace_msg = f"generate -> answer with {len(citations)} citations"

            answer = st.write_stream(_stream_tokens(llm, final_prompt))
            elapsed = time.time() - t0

            trace = paused_state.get("trace", []) + [generate_trace_msg]

            tool_badges = " ".join(
                f'<span class="tool-badge" style="background:{TOOL_COLORS[t]}">{TOOL_ICONS[t]} {t}</span>'
                for t in tools
            )
            st.markdown(
                f'<span style="opacity:0.7">⏱️ {elapsed:.1f}s total '
                f"({prep_elapsed:.1f}s routing/retrieval) · {len(trace)} agent steps</span> "
                f"{tool_badges}",
                unsafe_allow_html=True,
            )

            if citations:
                with st.expander(f"📎 {len(citations)} source citations"):
                    render_citations(citations)

            if show_trace:
                with st.expander("🧠 Agent reasoning trace", expanded=True):
                    render_trace(trace)

        st.session_state.history.append(
            {
                "query": query,
                "answer": answer,
                "citations": citations,
                "trace": trace,
                "tools_selected": tools,
                "elapsed": elapsed,
                "prep_elapsed": prep_elapsed,
                "model_name": model_name,
            }
        )
