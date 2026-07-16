"""
Web search tool via Tavily -- the "live internet" source in the
architecture. Distinct from the document corpus (static, self-authored)
and the specs DB (structured, local): this is the only source that can
answer questions about anything after this project's data was written, or
anything outside its scope entirely.

search_depth="advanced" + include_raw_content=True: the original
"basic" depth returns short (~500 char) NLP-summarized snippets per
result -- enough to judge relevance, not enough to complete a detailed
answer (this was caught live: a sourdough recipe question correctly
routed to web, got real citations, but the agent had to decline to give
exact quantities because the snippets didn't contain them). "advanced"
depth does deeper content extraction per result, and include_raw_content
returns the actual page content (markdown-cleaned) rather than just a
summary, at a higher per-call cost (2 Tavily credits instead of 1) --
worth it here since most queries this agent will route to `web` for are
exactly the kind that need real detail, not a one-line summary.
"""

import os

from tavily import TavilyClient

MAX_CONTENT_CHARS = 2000  # per result, after the depth/raw_content upgrade


def get_tavily_client(api_key: str | None = None) -> TavilyClient:
    """
    api_key: pass explicitly for a session-scoped key rather than
    relying on the TAVILY_API_KEY env var, which is shared process-wide
    state -- see agent.llm_factory.build_llm's docstring for the same
    reasoning (a mutated env var leaks one session's key into every
    other concurrent session in the same process).
    """
    api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at "
            "https://tavily.com and add it to .env."
        )
    return TavilyClient(api_key=api_key)


def web_search(query: str, max_results: int = 4, api_key: str | None = None) -> list[dict]:
    """Returns a list of {title, url, content} dicts from live web search."""
    client = get_tavily_client(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=True,
    )
    results = []
    for r in response.get("results", []):
        # Prefer raw_content (full page text) when available; fall back to
        # the summarized `content` field if raw extraction failed for a
        # given URL (e.g. JS-heavy pages, paywalls).
        text = r.get("raw_content") or r.get("content", "")
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": text[:MAX_CONTENT_CHARS],
            }
        )
    return results
