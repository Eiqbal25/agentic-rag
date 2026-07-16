"""
Unit tests for the traditional RAG graph (src/graph.py:
build_traditional_graph / run_traditional_rag).

The whole point of this graph is what it does NOT do -- these tests
mostly verify absence of agentic mechanisms (no routing, no grading, no
retry, no memory), since that's the entire basis for using it as a fair
comparison baseline against the agentic graph.
"""

from agent import build_traditional_graph
from retrieval import HybridRetriever


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, generate_response="A generated answer."):
        self.generate_response = generate_response
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if "Rate how relevant" in prompt:
            return FakeResp("7")
        if "Using the following context" in prompt:
            return FakeResp(self.generate_response)
        return FakeResp("ok")


def _fresh_state(query: str) -> dict:
    return {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "trace": [],
        "answer": "",
        "citations": [],
    }


class TestTraditionalGraphStructure:
    def test_runs_end_to_end(self):
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("What is corrective RAG?"))
        assert result["answer"]
        assert isinstance(result["citations"], list)

    def test_never_grades(self):
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("What is corrective RAG?"))
        trace_text = " ".join(result["trace"]).lower()
        assert "grade" not in trace_text

    def test_never_retries(self):
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("some vague query"))
        # check for actual retry-loop step names, not a naive substring
        # scan -- the generate step's own trace message says "no retry"
        # as part of describing what it does NOT do, which would
        # false-positive on a blind "retry" in trace_text check
        step_names = [t.split(" -> ")[0].split("(")[0] for t in result["trace"]]
        assert "rewrite_query" not in step_names
        # a single retrieve call only -- no second retrieve from a retry loop
        retrieve_steps = [s for s in step_names if s == "retrieve"]
        assert len(retrieve_steps) == 1

    def test_never_routes_tools(self):
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("anything"))
        trace_text = " ".join(result["trace"]).lower()
        assert "analyze_query" not in trace_text
        assert "tools_selected" not in result  # state schema itself has no tool concept

    def test_exactly_three_trace_steps(self):
        # retrieve -> rerank -> generate, no branches, no loops -- always
        # exactly 3 steps regardless of retrieval quality
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("What is QLoRA?"))
        assert len(result["trace"]) == 3

    def test_generates_even_from_weak_retrieval_no_self_check(self):
        """
        This is the entire point of traditional mode: it has no mechanism
        to notice or react to weak retrieval, unlike the agentic graph's
        grade_documents step. A vague/irrelevant query still produces a
        confident single-pass answer, first try, no matter what.
        """
        retriever = HybridRetriever()
        llm = FakeLLM(generate_response="A confident-sounding but ungrounded answer.")
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(
            _fresh_state("tell me about the thing with checking results before trusting them")
        )
        assert result["answer"] == "A confident-sounding but ungrounded answer."
        # exactly one generate call, no retry attempt regardless of quality
        generate_calls = [c for c in llm.calls if "Using the following context" in c]
        assert len(generate_calls) == 1

    def test_uses_same_retriever_as_agentic_mode(self):
        """
        Deliberate design choice: traditional mode shares the exact same
        HybridRetriever instance/logic as agentic mode, so any behavioral
        difference between the two modes is attributable to the
        decision-making layer, not to different retrieval quality.
        """
        retriever = HybridRetriever()
        llm = FakeLLM()
        app = build_traditional_graph(llm=llm, retriever=retriever)
        result = app.invoke(_fresh_state("What is HNSW indexing?"))
        sources = [c["source"] for c in result["citations"]]
        assert "03_vector_databases.md" in sources  # same retrieval quality as agentic mode's tests
