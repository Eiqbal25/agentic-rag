"""
Streamlit demo: Agentic RAG (single agent + tools) over AI infrastructure
knowledge -- documents, a structured hardware specs DB, and live web search.

This module is the orchestrator; see the top-level app.py (project root)
for the actual entry point run by `streamlit run app.py`.
"""

import streamlit as st

from .analytics_page import render_dashboard_tab, render_quality_metrics_tab
from .chat_page import render_chat_tab
from .config import DEFAULT_MODEL, SAME_AS_ANSWER_MODEL
from .documents_page import render_documents_tab
from .resources import get_app
from .settings_page import render_settings_tab
from .sidebar import render_nav_sidebar, render_setup_sidebar
from .sources_page import render_sources_tab
from .styles import CUSTOM_CSS

st.set_page_config(page_title="Agentic RAG — Multi-Tool Agent", page_icon="🔎", layout="wide")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🔎 Agentic RAG — Multi-Tool Agent")
    st.caption(
        "Single orchestrating agent with three tools: a document knowledge "
        "base (RAG concepts, on-prem AI infra), a structured hardware "
        "specs database (real GPU/SSD specs), and live web search — "
        "routed per-query, with a self-correcting retrieval loop on the "
        "document tool. Built with LangGraph, hybrid RRF retrieval, and Groq."
    )

    nav = render_nav_sidebar()
    groq_key, tavily_key = render_setup_sidebar()

    if not groq_key:
        st.info(
            "Add GROQ_API_KEY to your .env file to start (free at "
            "console.groq.com), or paste one in the sidebar for this "
            "session only."
        )
        st.stop()

    # Deliberately NOT os.environ["GROQ_API_KEY"] = groq_key here --
    # confirmed live as a real bug: os.environ is process-global, but a
    # single Streamlit process serves many concurrent browser sessions,
    # so one user's pasted session-only key would silently leak into
    # every other concurrent session's LLM/web-search calls. groq_key
    # and tavily_key are threaded explicitly into get_app() instead (see
    # its docstring), which passes them all the way down to the actual
    # ChatGroq/TavilyClient construction points.

    # Read from session_state rather than a return value, since the
    # Models & Settings page (which owns these widgets) may not have
    # run yet this session -- see render_nav_sidebar's docstring.
    model_name = st.session_state.get("model_name", DEFAULT_MODEL)
    show_trace = st.session_state.get("show_trace", True)
    fast_model_choice = st.session_state.get("fast_model_choice", SAME_AS_ANSWER_MODEL)

    app, llm = get_app(groq_key, model_name, fast_model_choice, tavily_key)

    # Not nested in any layout container (tabs/columns/expanders), so
    # it keeps its sticky-bottom-of-page CSS positioning -- only called
    # at all while the Chat page is active.
    query = None
    if nav == "💬 Chat":
        query = st.chat_input(
            "Ask about RAG concepts, GPU/SSD specs, or anything needing live search..."
        )

    if nav == "💬 Chat":
        render_chat_tab(app, llm, show_trace, query, model_name)
    elif nav == "📄 Documents":
        render_documents_tab()
    elif nav == "⚙️ Models & Settings":
        render_settings_tab()
    elif nav == "📎 Sources":
        render_sources_tab()
    elif nav == "📊 Analytics":
        render_dashboard_tab()
        st.divider()
        render_quality_metrics_tab()
