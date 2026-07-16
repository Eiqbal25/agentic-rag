"""
Constructs Groq-backed LLM clients: the main "answer" model, and the
"fast" tier used for structured micro-decisions (routing, reranking,
grading, query rewriting, text-to-SQL).
"""

import os

from langchain_groq import ChatGroq


def build_llm(
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    api_key: str | None = None,
):
    """
    Default model: openai/gpt-oss-120b.

    api_key: pass explicitly when a caller has a session-scoped key (e.g.
    the Streamlit app's sidebar override) -- falls back to the
    GROQ_API_KEY env var otherwise. SECURITY NOTE: the app used to rely
    entirely on `os.environ["GROQ_API_KEY"] = <session override>` before
    calling this, which is process-global state in a process that can
    serve multiple concurrent Streamlit sessions -- one user's pasted
    key would silently leak into every other concurrent session's LLM
    calls. Accepting api_key explicitly lets callers thread a real
    per-session value through instead of mutating shared process state.

    NOTE: llama-3.3-70b-versatile (a common default in older LangChain/Groq
    tutorials) was deprecated by Groq on 2026-06-17 with a shutdown date of
    2026-08-16 -- see https://console.groq.com/docs/deprecations. Groq's own
    migration guidance points to openai/gpt-oss-120b or qwen/qwen3.6-27b.

    This project briefly defaulted to qwen/qwen3.6-27b instead, after a
    live run reproduced a real, known gpt-oss-120b failure mode: gpt-oss
    models use OpenAI's "Harmony" response format internally (reasoning/
    analysis/final channels), which intermittently fails to parse cleanly
    through Groq's API layer -- a `groq.BadRequestError: output_parse_failed`
    with an empty `failed_generation` field, unrelated to prompt content
    (multiple open langchain-ai/langchain GitHub issues reproduce the same
    error against this exact model on Groq). In practice, though, qwen3.6-27b
    turned out to error out for this user's actual usage more often than
    gpt-oss-120b's intermittent parse failures did -- so the default was
    switched back to gpt-oss-120b. Both known failure modes are real; pick
    whichever one you hit less in your own usage via the Models & Settings
    page, this is a live tradeoff, not a solved problem.

    reasoning_effort: passed directly to Groq's API. For qwen3 models,
    'none' fully disables reasoning; 'default' or None lets it reason
    normally. For gpt-oss models, valid values are 'low'/'medium'/'high'
    (no 'none' option). See build_graph's fast_llm auto-configuration for
    why this matters: reproduced live, qwen3.6-27b burned its entire
    token budget reasoning about a single SQL-generation call (an
    ambiguous GPU comparison question) and never emitted any SQL at all --
    exactly the kind of structured micro-decision that shouldn't be doing
    open-ended chain-of-thought in the first place.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it as an environment variable before running the app."
        )
    kwargs = {"model": model_name, "temperature": temperature, "api_key": api_key}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def _reasoning_effort_for(model_name: str) -> str | None:
    """
    Shared tuning table: qwen3 models fully disable reasoning ('none');
    gpt-oss (Harmony-format) models don't support 'none', so 'low' is
    their fastest supported level. Unrecognized families return None
    (caller decides what that means).
    """
    if model_name.startswith("qwen/"):
        return "none"
    elif model_name.startswith("openai/gpt-oss"):
        return "low"
    return None


def build_fast_llm(model_name: str, api_key: str | None = None):
    """
    Constructs an LLM tuned for structured micro-decisions (routing,
    reranking, grading, query rewriting, text-to-SQL) for an EXPLICITLY
    chosen model_name -- e.g. a genuinely smaller/cheaper model than the
    main answer model, picked by the user specifically to cut token
    spend on these calls (which run every query, unlike the single
    final-generation call). Same reasoning_effort tuning as
    _build_default_fast_llm, factored out so both paths share one
    tuning table instead of two copies drifting apart.

    api_key: see build_llm's docstring for why this should be passed
    explicitly rather than relying on a mutated GROQ_API_KEY env var.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it as an environment variable before running the app."
        )
    kwargs = {"model": model_name, "temperature": 0.0, "api_key": api_key}
    reasoning_effort = _reasoning_effort_for(model_name)
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def _build_default_fast_llm(llm, api_key: str | None = None):
    """
    Auto-constructs a fast_llm tuned for structured micro-decisions
    (routing, reranking, grading, SQL generation) from the SAME model as
    `llm`, but with reasoning disabled or minimized via Groq's
    reasoning_effort parameter.

    Why this exists: reproduced live -- a real user question ("Adakah
    gtx1050 lebih baik dari rtx5060", asking about two GPU models not in
    the specs database) caused qwen3.6-27b's text-to-SQL call to spend
    its ENTIRE token budget reasoning out loud about whether the model
    names might be typos, whether it was a "trick question," etc. -- and
    got cut off before ever emitting a single line of SQL. The query
    correctly failed validation afterward (no hallucinated data was
    returned), but the deeper problem is that a simple structured
    decision should never have been doing open-ended chain-of-thought in
    the first place.

    api_key: see build_llm's docstring -- pass the caller's actual
    session-scoped key explicitly rather than relying on the
    GROQ_API_KEY env var, which is shared process-global state.

    Falls back to reusing the exact same `llm` instance for any
    other/unrecognized model family, or if no key is available to
    construct a second client -- never raises, always returns something
    usable.
    """
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or ""
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return llm

    reasoning_effort = _reasoning_effort_for(model_name)
    if reasoning_effort is None:
        return llm  # unrecognized model family -- don't guess, just reuse llm

    try:
        return ChatGroq(
            model=model_name, temperature=0.0, api_key=api_key, reasoning_effort=reasoning_effort
        )
    except Exception:
        return llm  # never let fast_llm construction be a hard failure
