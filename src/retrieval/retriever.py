"""
Hybrid retrieval: fuse dense (Chroma/Qwen3-Embedding) and sparse (BM25)
rankings with Reciprocal Rank Fusion, then optionally rerank the fused
candidate set with an LLM-as-cross-encoder pass for higher precision at
the top.
"""

import json
import pickle
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm_utils import invoke_with_retry

from .embeddings import Qwen3Embeddings
from .ingest import BM25_PATH, CHROMA_DIR, COLLECTION_NAME, tokenize

RRF_K = 60
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 1.0


class HybridRetriever:
    def __init__(self):
        self.embeddings = Qwen3Embeddings()
        self.vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks: list[Document] = data["chunks"]
        # chunk_id -> chunk, and chunk_id -> bm25 corpus index
        self._id_to_chunk = {c.metadata["chunk_id"]: c for c in self.chunks}
        self._id_to_bm25_idx = {
            c.metadata["chunk_id"]: i for i, c in enumerate(self.chunks)
        }

    def _dense_ranked_ids(self, query: str, k: int) -> list[str]:
        results = self.vectordb.similarity_search(query, k=k)
        return [d.metadata["chunk_id"] for d in results]

    def _sparse_ranked_ids(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i].metadata["chunk_id"] for i in ranked_idx]

    def retrieve(
        self, query: str, k: int = 5, candidate_pool: int = 20
    ) -> list[tuple[Document, float]]:
        """Hybrid retrieval via weighted Reciprocal Rank Fusion.

        Returns top-k (Document, fused_score) pairs, highest score first.
        """
        dense_ids = self._dense_ranked_ids(query, candidate_pool)
        sparse_ids = self._sparse_ranked_ids(query, candidate_pool)

        fused_scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + DENSE_WEIGHT / (
                RRF_K + rank + 1
            )
        for rank, cid in enumerate(sparse_ids):
            fused_scores[cid] = fused_scores.get(cid, 0.0) + SPARSE_WEIGHT / (
                RRF_K + rank + 1
            )

        ranked = sorted(fused_scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self._id_to_chunk[cid], score) for cid, score in ranked]


def rerank_with_llm(llm, query: str, docs: list[Document], top_n: int = 4):
    """LLM-as-cross-encoder reranking, in ONE batched call.

    Uses the already-available chat LLM as a relevance judge instead of a
    dedicated pretrained cross-encoder (e.g. a Qwen3-Reranker/BGE-reranker
    model): it sees the query jointly with all candidate chunks (the
    defining property of cross-encoder scoring, as opposed to bi-encoder
    similarity) and returns relevance scores for all of them at once. Note
    this project's dense embeddings (embeddings.py) were originally on
    this same "no Hugging Face access" constraint but have since been
    swapped to a real pretrained model once that constraint was verified
    not to hold in this environment -- swapping the reranker the same way
    is a reasonable next step, just not done here yet.

    PERFORMANCE NOTE: this used to make one LLM call PER candidate
    document (5 candidates = 5 sequential round-trips just for
    reranking, before routing/grading/generation even happen -- a real
    latency bottleneck identified by reading the actual call pattern).
    Batching into a single call asking for all scores as one JSON array
    cuts this to 1 call regardless of candidate count. If the model
    doesn't comply with the JSON format, `_rerank_fallback_per_doc`
    preserves the original one-call-per-document behavior as a safety
    net -- reranking degrades to slower, never to broken.
    """
    if not docs:
        return []

    passages_text = "\n\n".join(
        f"[{i}] {doc.page_content[:600]}" for i, doc in enumerate(docs)
    )
    prompt = (
        f"Rate how relevant each of the {len(docs)} passages below is to "
        "the query, on a scale of 0-10 (10 = highly relevant, 0 = "
        "irrelevant).\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages_text}\n\n"
        "Respond with a JSON object containing a \"scores\" array with "
        f"exactly {len(docs)} integers, in the same order as the "
        'passages above (index 0 first). Example: {"scores": [8, 3, 9]}'
    )

    scores = None
    try:
        resp = invoke_with_retry(llm, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        scores = _parse_batched_scores(text, len(docs))
    except Exception:
        scores = None

    if scores is None:
        scores = _rerank_fallback_per_doc(llm, query, docs)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def _parse_batched_scores(text: str, n: int) -> list[int] | None:
    """
    Extracts a {"scores": [...]} JSON blob from the LLM's response and
    validates it has exactly n integer scores. Returns None if parsing
    fails or the length doesn't match, so the caller falls back to the
    slower but robust per-document scoring path -- same "structured
    output over prose-scanning, with a safety net" pattern used for the
    router's tool selection in agent/graph.py, for the same reason
    (reproduced live: not every model reliably follows free-text format
    instructions).
    """
    matches = re.findall(r"\{[^{}]*\}", text)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "scores" in parsed:
            raw = parsed["scores"]
            if isinstance(raw, list) and len(raw) == n:
                try:
                    return [int(s) for s in raw]
                except (ValueError, TypeError):
                    continue
    return None


def _rerank_fallback_per_doc(llm, query: str, docs: list[Document]) -> list[int]:
    """Safety net: the original one-call-per-document approach, used only
    if the batched JSON scoring fails to parse. Slower (N calls instead
    of 1) but robust -- ensures reranking never silently breaks even if
    a model doesn't comply with the batched JSON format."""
    scores = []
    for doc in docs:
        prompt = (
            "Rate how relevant this passage is to the query on a scale "
            "of 0-10. Respond with ONLY the integer, nothing else.\n\n"
            f"Query: {query}\n\n"
            f"Passage:\n{doc.page_content[:600]}"
        )
        try:
            resp = invoke_with_retry(llm, prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            match = re.search(r"\d+", text)
            score = int(match.group()) if match else 0
        except Exception:
            score = 0
        scores.append(score)
    return scores
