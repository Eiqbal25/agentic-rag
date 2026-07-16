"""
The orchestrating agent: LangGraph state graph (routing, corrective
retrieval loop, tool gathering, cited generation) plus a minimal
traditional-RAG comparison graph. See agent/graph.py for the full
design notes.

Re-exports the public surface so callers can do `from agent import
build_graph, build_llm` etc. -- private/internal helpers (e.g.
`_build_default_fast_llm`) are NOT re-exported here; import those
directly from their defining submodule (e.g. `agent.llm_factory`).
"""

from dotenv import load_dotenv

load_dotenv()

from .graph import (  # noqa: E402
    build_graph,
    build_traditional_graph,
    run_agentic_rag,
    run_traditional_rag,
)
from .llm_factory import build_fast_llm, build_llm  # noqa: E402
from .parsing import parse_grade_response, parse_tool_selection  # noqa: E402
from .prompts import build_direct_prompt, build_generation_prompt  # noqa: E402
from .state import AgentState, TraditionalRAGState  # noqa: E402

__all__ = [
    "AgentState",
    "TraditionalRAGState",
    "build_llm",
    "build_fast_llm",
    "parse_tool_selection",
    "parse_grade_response",
    "build_direct_prompt",
    "build_generation_prompt",
    "build_graph",
    "build_traditional_graph",
    "run_agentic_rag",
    "run_traditional_rag",
]
