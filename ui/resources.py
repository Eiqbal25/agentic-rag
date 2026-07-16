"""Cached, expensive-to-build resources: the retrieval index and the
compiled agent graph."""

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

from agent import build_fast_llm, build_graph, build_llm
from retrieval import HybridRetriever

from .config import SAME_AS_ANSWER_MODEL


@st.cache_resource(show_spinner="Loading retrieval indexes...")
def get_retriever():
    return HybridRetriever()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_app(groq_api_key: str, model_name: str, fast_model_choice: str, tavily_api_key: str = ""):
    """
    Returns (graph, llm). Compiled with a checkpointer and
    interrupt_before=["generate"] -- runs routing/retrieval/reranking/
    grading/tool-gathering automatically, then pauses right before the
    final generation call so the answer can be streamed token-by-token
    instead of appearing all at once.

    fast_model_choice: SAME_AS_ANSWER_MODEL leaves fast_llm=None, so
    build_graph auto-derives it from `llm` itself (same model, reasoning
    disabled/minimized -- see agent.llm_factory._build_default_fast_llm).
    Any other value is an explicit, genuinely different (typically
    smaller) model the user picked in Models & Settings specifically to
    cut token spend on routing/reranking/grading/rewriting/text-to-SQL --
    calls that run every query, unlike the single final-generation call
    that always uses the main answer model.

    SECURITY NOTE, confirmed live as a real bug: these key params used
    to be named `_api_key` (underscore-prefixed), which tells Streamlit
    to EXCLUDE the argument from this cache's key hash -- meaning the
    cache was keyed only on (model_name, fast_model_choice), and the
    key value itself was never even read inside this function (llm/
    fast_llm were built from os.environ instead). Two compounding
    problems from that: (1) whichever session's key happened to be
    active in the shared process when a given (model_name,
    fast_model_choice) combo was FIRST requested got baked into the
    cached graph/llm forever, silently reused for every other session
    requesting that same combo regardless of their own key; (2) even
    ignoring the cache, os.environ is process-global, so concurrent
    sessions could race and use each other's keys. Fixed by dropping the
    underscore (so different keys correctly get different cached
    instances) and threading the key explicitly into build_llm/
    build_fast_llm/build_graph instead of relying on env vars at all.
    """
    llm = build_llm(model_name=model_name, api_key=groq_api_key)
    fast_llm = (
        None
        if fast_model_choice == SAME_AS_ANSWER_MODEL
        else build_fast_llm(fast_model_choice, api_key=groq_api_key)
    )
    retriever = get_retriever()
    checkpointer = MemorySaver()
    graph = build_graph(
        llm=llm,
        fast_llm=fast_llm,
        retriever=retriever,
        checkpointer=checkpointer,
        interrupt_before=["generate"],
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key or None,
    )
    return graph, llm
