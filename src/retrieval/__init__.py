"""
The `docs` tool: embedding models, the ingestion pipeline (chunk ->
embed -> Chroma + BM25), and the hybrid dense+sparse retriever with LLM
reranking.

Re-exports the public surface so callers can do `from retrieval import
HybridRetriever, rerank_with_llm` as before. For ingestion
(`build_indexes`) or embedding classes directly, import from the
specific submodule: `retrieval.ingest`, `retrieval.embeddings`.
"""

from .embeddings import LSAEmbeddings, Qwen3Embeddings
from .retriever import HybridRetriever, rerank_with_llm

__all__ = [
    "Qwen3Embeddings",
    "LSAEmbeddings",
    "HybridRetriever",
    "rerank_with_llm",
]
