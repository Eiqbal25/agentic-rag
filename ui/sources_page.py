"""Sources page: session-wide citation log, deduplicated and ranked by
reuse."""

import streamlit as st

from .styles import CITATION_ICONS, CITATION_TYPE_TO_TOOL, TOOL_COLORS, escape_html


def render_sources_tab():
    """
    Session-wide citation log: every unique source cited across all
    turns so far, deduplicated and ranked by how often it's been reused
    -- the multi-turn counterpart to each answer's own per-message
    citation expander (which only shows that one turn's sources).

    SECURITY: `source` is untrusted (a web page title from live Tavily
    results, or a document section/filename from the Documents tab's
    user-addable corpus) -- escaped before interpolation into
    unsafe_allow_html=True markdown. See chat_page.render_citations'
    docstring for the confirmed-exploitable details this mirrors.
    """
    st.subheader("📎 Sources cited this session")
    history = st.session_state.get("history", [])
    all_citations = [c for turn in history for c in turn.get("citations", [])]
    if not all_citations:
        st.info("No citations yet this session — ask something in the Chat tab that needs docs, specs, or web sources.")
        return

    seen: dict[tuple, dict] = {}
    for c in all_citations:
        key = (c.get("type"), c.get("source"))
        if key not in seen:
            seen[key] = {"citation": c, "count": 0}
        seen[key]["count"] += 1

    st.caption(f"{len(seen)} unique source(s) across {len(history)} answer(s) this session.")
    for (ctype, source), info in sorted(seen.items(), key=lambda kv: -kv[1]["count"]):
        c = info["citation"]
        tool = CITATION_TYPE_TO_TOOL.get(ctype, "docs")
        color = TOOL_COLORS.get(tool, "#999")
        icon = CITATION_ICONS.get(ctype, "•")
        st.markdown(
            f'<div class="citation-card" style="--accent-color:{color}">'
            f'<div class="citation-source">{icon} {escape_html(source)} '
            f'<span class="tool-badge" style="background:{color}">cited {info["count"]}×</span></div>'
            f'<div class="citation-meta">{escape_html(ctype.replace("_", " "))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
