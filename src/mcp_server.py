"""
MCP server exposing this project's three tools (document search, specs
database, web search) through the Model Context Protocol.

WHY THIS EXISTS (worth being precise about, since MCP is easy to justify
badly): our Streamlit app calls these tools directly as Python functions
-- that's the right choice for a single-consumer app, since it avoids the
serialization/IPC overhead MCP adds for no benefit when there's only one
caller. This server exists to create a SECOND, genuinely different
consumer: any MCP-compatible client (Claude Desktop, Cursor, etc.) can
connect here and use the exact same tools -- the same retrieval index,
the same specs database, the same web search -- without any of this
project's Python code being duplicated or reimplemented. That's the
actual, defensible value of MCP: one tool implementation, multiple
independent consumers, not "MCP because the tool list is nice to look at."

Run standalone (for use with Claude Desktop, MCP Inspector, etc.):
    uv run python src/mcp_server.py

Then point an MCP client at this script (stdio transport).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from retrieval import HybridRetriever  # noqa: E402
from tools.specs_tool import run_specs_query  # noqa: E402
from tools.web_tool import web_search as _web_search  # noqa: E402

mcp = FastMCP("agentic-rag-tools")

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@mcp.tool()
def search_documents(query: str, k: int = 5) -> str:
    """
    Search the AI infrastructure document knowledge base (RAG, embeddings,
    vector databases, fine-tuning, quantization, inference optimization,
    on-prem AI infra -- 12 documents, hybrid dense+BM25 retrieval).
    Returns the top-k matching chunks with their source filenames.
    """
    docs = _get_retriever().retrieve(query, k=k)
    if not docs:
        return "No matching documents found."
    return "\n\n---\n\n".join(
        f"[{d.metadata['source']}]\n{d.page_content}" for d, _ in docs
    )


@mcp.tool()
def query_specs_database(sql: str) -> str:
    """
    Run a read-only SQL SELECT query against the hardware specs database.

    Schema:
    Table: gpus (model, vram_gb, memory_type, memory_bandwidth_gbps,
      fp16_tflops_dense, nvlink_gbps, tdp_watts, price_tier, notes, source_url)
    Table: ssds (model, interface, capacity_tb, seq_read_mbps,
      seq_write_mbps, endurance_dwpd, form_factor, notes, source_url)
    """
    try:
        rows = run_specs_query(sql)
        if not rows:
            return "Query returned no rows."
        return "\n".join(str(r) for r in rows)
    except Exception as e:
        return f"Query error: {e}"


@mcp.tool()
def web_search(query: str, max_results: int = 4) -> str:
    """
    Search the live web via Tavily. Use for current information or
    anything outside the document knowledge base and specs database.
    """
    try:
        results = _web_search(query, max_results=max_results)
    except RuntimeError as e:
        return f"Web search unavailable: {e}"
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['content']}" for r in results
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
