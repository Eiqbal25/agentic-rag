"""
Unit tests for src/retrieval.py -- Reciprocal Rank Fusion math in
isolation, plus end-to-end retrieval sanity checks against the real
built index.
"""

import pytest

from retrieval.retriever import RRF_K, HybridRetriever, rerank_with_llm, _parse_batched_scores


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


class TestRRFMath:
    """
    Tests the RRF formula in isolation: score = weight / (k + rank + 1).
    Reimplements the same computation the retriever does internally, to
    verify the fusion arithmetic itself is correct, independent of what
    the actual dense/sparse retrievers return for any given query.
    """

    def _rrf_score(self, rank: int, weight: float = 1.0, k: int = RRF_K) -> float:
        return weight / (k + rank + 1)

    def test_rank_zero_scores_higher_than_rank_one(self):
        assert self._rrf_score(0) > self._rrf_score(1)

    def test_score_decreases_monotonically_with_rank(self):
        scores = [self._rrf_score(r) for r in range(10)]
        assert scores == sorted(scores, reverse=True)

    def test_fused_score_is_sum_across_retrievers(self):
        # a document ranked #1 in both dense and sparse should score
        # exactly double a document ranked #1 in only one retriever
        both = self._rrf_score(0) + self._rrf_score(0)
        one_only = self._rrf_score(0)
        assert both == pytest.approx(2 * one_only)

    def test_weighting_changes_relative_contribution(self):
        # a heavily-weighted dense retriever should be able to outrank a
        # low-ranked-but-present sparse hit
        dense_heavy = self._rrf_score(0, weight=2.0)
        sparse_light = self._rrf_score(0, weight=0.1)
        assert dense_heavy > sparse_light

    def test_k_dampens_high_rank_influence(self):
        # increasing k should shrink the score gap between rank 0 and
        # rank 5 (that's the entire purpose of the k constant)
        gap_small_k = self._rrf_score(0, k=1) - self._rrf_score(5, k=1)
        gap_large_k = self._rrf_score(0, k=1000) - self._rrf_score(5, k=1000)
        assert gap_small_k > gap_large_k


class TestHybridRetrieverEndToEnd:
    def test_returns_requested_k(self, retriever):
        results = retriever.retrieve("what is RAG", k=5)
        assert len(results) == 5

    def test_results_are_score_sorted_descending(self, retriever):
        results = retriever.retrieve("quantization inference speed", k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_found_for_known_query(self, retriever):
        # regression check: corpus content collision fixed earlier
        # (02_agentic_rag_patterns.md no longer contains a literal phrase
        # collision with the quantization doc)
        results = retriever.retrieve("How does quantization affect inference speed?", k=3)
        top_sources = [d.metadata["source"] for d, _ in results]
        assert "08_quantization_techniques.md" in top_sources[:2]

    def test_empty_query_does_not_crash(self, retriever):
        results = retriever.retrieve("", k=3)
        assert isinstance(results, list)


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Always returns the same canned response, counts calls made."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return FakeResp(self.response)


class TestParseBatchedScores:
    def test_clean_json(self):
        assert _parse_batched_scores('{"scores": [8, 3, 9]}', 3) == [8, 3, 9]

    def test_json_with_reasoning_before_it(self):
        text = 'Let me think about each passage.\n\n{"scores": [5, 7, 2]}'
        assert _parse_batched_scores(text, 3) == [5, 7, 2]

    def test_wrong_length_returns_none(self):
        # model returned scores for 2 passages when 3 were asked for
        assert _parse_batched_scores('{"scores": [8, 3]}', 3) is None

    def test_no_json_returns_none(self):
        assert _parse_batched_scores("I think it's pretty relevant, an 8.", 3) is None

    def test_non_integer_scores_returns_none(self):
        assert _parse_batched_scores('{"scores": ["high", "low", "mid"]}', 3) is None


class TestBatchedReranking:
    def test_happy_path_makes_exactly_one_call(self, retriever):
        docs = [d for d, _ in retriever.retrieve("quantization", k=5)]
        llm = FakeLLM('{"scores": [8, 3, 9, 2, 7]}')
        rerank_with_llm(llm, "quantization", docs, top_n=3)
        assert llm.calls == 1

    def test_returns_correct_top_n_sorted_by_score(self, retriever):
        docs = [d for d, _ in retriever.retrieve("quantization", k=5)]
        llm = FakeLLM('{"scores": [8, 3, 9, 2, 7]}')
        result = rerank_with_llm(llm, "quantization", docs, top_n=3)
        assert len(result) == 3
        scores = [s for _, s in result]
        assert scores == [9, 8, 7]  # sorted descending, top 3 of [8,3,9,2,7]

    def test_verbose_response_with_json_still_one_call(self, retriever):
        docs = [d for d, _ in retriever.retrieve("quantization", k=5)]
        llm = FakeLLM(
            "Let me evaluate each passage carefully for relevance.\n\n"
            '{"scores": [5, 5, 5, 5, 5]}'
        )
        rerank_with_llm(llm, "quantization", docs, top_n=3)
        assert llm.calls == 1

    def test_falls_back_to_per_document_when_batch_format_ignored(self, retriever):
        """
        Safety net: if the model doesn't comply with the JSON format at
        all, reranking must not silently break -- it falls back to the
        original one-call-per-document approach (slower, but robust).
        """
        docs = [d for d, _ in retriever.retrieve("quantization", k=5)]
        llm = FakeLLM("I think this passage is quite relevant, maybe a 7 out of 10.")
        result = rerank_with_llm(llm, "quantization", docs, top_n=3)
        # 1 failed batch attempt + 5 fallback per-doc calls
        assert llm.calls == 1 + len(docs)
        assert len(result) == 3

    def test_empty_docs_list_makes_no_calls(self):
        llm = FakeLLM('{"scores": []}')
        result = rerank_with_llm(llm, "anything", [], top_n=3)
        assert result == []
        assert llm.calls == 0
