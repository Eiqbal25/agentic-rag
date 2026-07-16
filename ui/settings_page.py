"""Models & Settings page: answer/fast model selection, trace toggle,
and the tool legend."""

import streamlit as st

from .config import DEFAULT_MODEL, MODEL_OPTIONS, SAME_AS_ANSWER_MODEL
from .styles import TOOL_COLORS, TOOL_ICONS


def render_settings_tab():
    """
    Model choice, trace visibility, and the tool legend -- its own page
    rather than sidebar clutter.

    IMPORTANT: each widget's persisted value lives under a plain
    st.session_state key (e.g. "model_name") that main() reads directly;
    the widget itself uses a SEPARATE "*_widget" key and has its index/
    value recomputed from the persisted key on every render, with the
    result written straight back after. This is more code than just
    giving the widget the persisted key directly, but that simpler
    version is a confirmed live bug: reproduced here, a widget whose key
    doubles as the persisted value silently reverted to its index
    default the first time the user navigated away from this page (so
    the widget didn't render that turn) and back -- Streamlit did not
    keep the prior value in session_state across the skipped render.
    Recomputing index= from an independently-maintained value sidesteps
    that regardless of the underlying cause.
    """
    st.subheader("🧠 Model")
    prior_model = st.session_state.get("model_name", DEFAULT_MODEL)
    model_name = st.selectbox(
        "Answer model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(prior_model) if prior_model in MODEL_OPTIONS else 0,
        key="model_name_widget",
        help=(
            "llama-3.3-70b-versatile is deprecated on Groq (shutdown "
            "2026-08-16). qwen3.6-27b is the default here for "
            "reliability (gpt-oss models intermittently fail with a "
            "Groq output_parse_failed error, reproduced live during "
            "testing) -- switch models to compare speed/quality, not "
            "as a reliability guarantee."
        ),
    )
    st.session_state["model_name"] = model_name

    prior_trace = st.session_state.get("show_trace", True)
    show_trace = st.checkbox(
        "Show agent reasoning trace", value=prior_trace, key="show_trace_widget"
    )
    st.session_state["show_trace"] = show_trace

    st.divider()
    st.subheader("⚡ Token optimization")
    st.caption(
        "Routing, reranking, grading, query rewriting, and text-to-SQL "
        "all run an LLM call on every query -- final answer generation "
        "runs once. Pointing those at a smaller/cheaper model than the "
        "answer model cuts real token spend where it's actually being "
        "spent, not on the one call the user is judging output quality on."
    )
    fast_options = [SAME_AS_ANSWER_MODEL] + MODEL_OPTIONS
    prior_fast = st.session_state.get("fast_model_choice", SAME_AS_ANSWER_MODEL)
    fast_model_choice = st.selectbox(
        "Fast model (routing/reranking/grading/rewriting/text-to-SQL)",
        fast_options,
        index=fast_options.index(prior_fast) if prior_fast in fast_options else 0,
        key="fast_model_choice_widget",
        help=(
            "'Same as answer model' reuses the answer model but with "
            "reasoning disabled/minimized via Groq's reasoning_effort "
            "(the existing default) -- no second model spun up, just "
            "cheaper calls on the same one. Picking an explicit model "
            "here (e.g. openai/gpt-oss-20b) spins up a second, genuinely "
            "smaller client for these calls instead. Note: gpt-oss-20b "
            "shares gpt-oss-120b's known Harmony-format parsing "
            "reliability issue on Groq (see README) -- it's offered here "
            "as an explicit opt-in, not the default, for exactly that "
            "reason."
        ),
    )
    st.session_state["fast_model_choice"] = fast_model_choice

    st.divider()
    st.subheader("🔧 Tools available")
    for tool, label in [
        ("docs", "12-document knowledge base (RAG/AI infra)"),
        ("specs", "GPU/SSD hardware specs database"),
        ("web", "live internet search (Tavily)"),
    ]:
        color = TOOL_COLORS[tool]
        st.markdown(
            f'<span class="tool-badge" style="background:{color}">{TOOL_ICONS[tool]} {tool}</span> {label}',
            unsafe_allow_html=True,
        )

    return model_name, show_trace
