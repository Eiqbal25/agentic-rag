"""
Prompt builders for the final generation step (both the no-tools
"direct" path and the tools-based cited-answer path). Extracted from
the generate() node so app.py can build the identical prompt itself for
streaming, without duplicating this logic or diverging from what the
non-streaming graph path actually sends.
"""

from .state import _format_history


def build_direct_prompt(state: dict) -> str:
    """
    Builds the prompt for the no-tools ("direct response") path.
    """
    history_str = _format_history(state.get("chat_history", []))
    return (
        "You are the direct-response step of an agentic system "
        "whose core design principle is that factual answers must "
        "be grounded in retrieved evidence (documents, a specs "
        "database, or web search) -- never given from unverified "
        "model memory alone. The routing step decided this "
        "message needs no tool, which should only be correct for "
        "genuinely conversational messages: greetings, thanks, "
        "small talk, or questions about the assistant itself.\n\n"
        "If the message below IS genuinely conversational, "
        "respond briefly and naturally.\n\n"
        "If the message is actually asking for facts, "
        "instructions, a recipe, an explanation, or any "
        "real-world informational content, do NOT answer it from "
        "memory even if you're confident you know it. Instead, "
        "say that this needs to be looked up and you don't have "
        "grounded information for it right now, and suggest the "
        "user rephrase so the right tool can be used.\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        f"Message: {state['original_query']}"
    )


def build_generation_prompt(state: dict) -> tuple[str, list[dict]]:
    """
    Builds the prompt and citation list for the tools-based generation
    path.

    Citations are pre-assigned numbers ([1], [2], ...) in source order,
    BEFORE the LLM ever sees them -- the LLM is told to cite using only
    these numbers, not to invent its own scheme. This is deliberately
    more reliable than letting the LLM format citations itself (e.g.
    [filename.md § section]): same "structured output over free-text
    generation" principle used for the router's tool selection and the
    reranker's scores elsewhere in this codebase, for the same reason --
    a model formatting its own citation string is a source of drift and
    inconsistency a fixed numbering scheme doesn't have. The same source
    (same document+section, same specs model, same web URL) reuses its
    original number if cited again later in the answer, rather than
    getting renumbered.
    """
    tools = state.get("tools_selected", [])
    history_str = _format_history(state.get("chat_history", []))

    # Collect raw source entries in order: docs, then specs, then web.
    raw_sources = []

    if "docs" in tools:
        docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
        for d, _ in docs:
            source = d.metadata["source"]
            section = d.metadata.get("section", "")
            raw_sources.append(
                {
                    "dedupe_key": ("document", source, section),
                    "citation": {
                        "type": "document",
                        "source": source,
                        "section": section,
                        "snippet": d.page_content[:100],
                    },
                    "context_header": f"{source}, section: {section or 'N/A'}",
                    "context_body": d.page_content,
                }
            )

    specs_error_block = None
    if "specs" in tools:
        specs = state.get("specs_result")
        if specs and specs.get("rows"):
            for r in specs["rows"]:
                model = r.get("model", "specs.db")
                raw_sources.append(
                    {
                        "dedupe_key": ("specs_db", model),
                        "citation": {
                            "type": "specs_db",
                            "source": model,
                            "section": "hardware specs",
                            "snippet": str(r)[:100],
                            "source_url": r.get("source_url", ""),
                        },
                        "context_header": "specs database row",
                        "context_body": str(r),
                    }
                )
        elif specs and specs.get("error"):
            specs_error_block = f"SPECS DATABASE: query failed ({specs['error']}), no data available."

    if "web" in tools:
        web_results = state.get("web_results") or []
        for r in web_results:
            raw_sources.append(
                {
                    "dedupe_key": ("web", r["url"]),
                    "citation": {
                        "type": "web",
                        "source": r["title"],
                        "section": "",
                        "snippet": r["content"][:100],
                        "source_url": r["url"],
                    },
                    "context_header": f"{r['title']} ({r['url']})",
                    "context_body": r["content"],
                }
            )

    # Assign numbers in first-seen order, reusing for duplicate sources.
    number_by_key: dict = {}
    citations = []
    for entry in raw_sources:
        key = entry["dedupe_key"]
        if key not in number_by_key:
            number_by_key[key] = len(number_by_key) + 1
            cit = dict(entry["citation"])
            cit["number"] = number_by_key[key]
            citations.append(cit)

    context_blocks = []

    doc_entries = [e for e in raw_sources if e["dedupe_key"][0] == "document"]
    if doc_entries:
        parts = [
            f"[{number_by_key[e['dedupe_key']]}] {e['context_header']}\n{e['context_body']}"
            for e in doc_entries
        ]
        context_blocks.append("DOCUMENT SOURCES:\n" + "\n\n---\n\n".join(parts))

    specs_entries = [e for e in raw_sources if e["dedupe_key"][0] == "specs_db"]
    if specs_entries:
        parts = [
            f"[{number_by_key[e['dedupe_key']]}] {e['context_body']}" for e in specs_entries
        ]
        context_blocks.append("SPECS DATABASE:\n" + "\n".join(parts))
    elif specs_error_block:
        context_blocks.append(specs_error_block)

    web_entries = [e for e in raw_sources if e["dedupe_key"][0] == "web"]
    if web_entries:
        parts = [
            f"[{number_by_key[e['dedupe_key']]}] {e['context_header']}: {e['context_body']}"
            for e in web_entries
        ]
        context_blocks.append("WEB SEARCH RESULTS:\n" + "\n\n".join(parts))

    full_context = "\n\n===\n\n".join(context_blocks) if context_blocks else "(no results found from any tool)"

    prompt = (
        "Answer the user's question using ONLY the information in the "
        "provided sources below. Each source is labeled with a number "
        "like [1], [2], etc. After every factual claim, cite it using "
        "ONLY that number in square brackets -- e.g. 'Quantization "
        "reduces memory bandwidth needs [1].' If the same source "
        "supports another claim later in your answer, reuse its exact "
        "same number. Do not invent new numbers, do not renumber "
        "sources yourself, and do not cite using filenames or titles -- "
        "use ONLY the [N] numbers shown before each source below.\n"
        "If the sources don't contain enough information, say so "
        "explicitly rather than guessing.\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        f"Question: {state['original_query']}\n\n"
        f"Sources:\n{full_context}\n\n"
        "Answer (with inline [N] citations):"
    )
    return prompt, citations
