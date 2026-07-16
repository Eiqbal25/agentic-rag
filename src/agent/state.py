"""
Shared state schemas and history-formatting for the agentic and
traditional graphs.
"""

from typing import Annotated, TypedDict

MAX_RETRIES = 2
RETRIEVE_K = 5
RERANK_TOP_N = 4
MAX_HISTORY_TURNS = 3


class AgentState(TypedDict):
    original_query: str
    query: str
    chat_history: list  # list[{"query": str, "answer": str}], most recent last
    tools_selected: list  # subset of ["docs", "specs", "web"]
    retrieved_docs: list  # list[(Document, score)]
    reranked_docs: list  # list[(Document, score)]
    grade: str
    grade_reasoning: str
    retries: int
    specs_result: dict  # {"sql": str, "rows": list[dict], "error": str|None}
    web_results: list  # list[{"title","url","content"}]
    trace: Annotated[list[str], lambda a, b: a + b]
    answer: str
    citations: list[dict]


class TraditionalRAGState(TypedDict):
    query: str
    retrieved_docs: list
    reranked_docs: list
    trace: Annotated[list[str], lambda a, b: a + b]
    answer: str
    citations: list[dict]


def _format_history(history: list) -> str:
    if not history:
        return "(no earlier turns in this conversation)"
    recent = history[-MAX_HISTORY_TURNS:]
    return "\n".join(
        f"Q{i+1}: {t['query']}\nA{i+1}: {t['answer'][:300]}"
        for i, t in enumerate(recent)
    )
