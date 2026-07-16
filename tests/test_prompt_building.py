"""
Unit tests for src/graph.py's build_direct_prompt / build_generation_prompt
-- extracted from the generate() node so app.py's streaming path can build
the identical prompt without duplicating logic. These tests exist to catch
drift between what the graph node sends and what these functions produce,
since app.py now depends on them being the single source of truth.
"""

from langchain_core.documents import Document

from agent import build_direct_prompt, build_generation_prompt


class TestBuildDirectPrompt:
    def test_includes_the_message(self):
        state = {"original_query": "hey, how's it going?", "chat_history": []}
        prompt = build_direct_prompt(state)
        assert "hey, how's it going?" in prompt

    def test_includes_chat_history(self):
        state = {
            "original_query": "follow-up question",
            "chat_history": [{"query": "first question", "answer": "first answer"}],
        }
        prompt = build_direct_prompt(state)
        assert "first question" in prompt

    def test_instructs_against_answering_from_memory(self):
        # this is the actual fix for the sourdough hallucination bug --
        # regression-test that the instruction is still present
        state = {"original_query": "anything", "chat_history": []}
        prompt = build_direct_prompt(state)
        assert "do not" in prompt.lower() or "not answer" in prompt.lower()


class TestBuildGenerationPrompt:
    def _doc(self, source="08_quantization_techniques.md", content="Quantization reduces memory."):
        return Document(page_content=content, metadata={"source": source, "section": "intro"})

    def test_docs_only_produces_document_citations(self):
        state = {
            "original_query": "What is quantization?",
            "chat_history": [],
            "tools_selected": ["docs"],
            "reranked_docs": [(self._doc(), 8)],
            "retrieved_docs": [],
            "specs_result": {},
            "web_results": [],
        }
        prompt, citations = build_generation_prompt(state)
        assert "quantization" in prompt.lower()
        assert len(citations) == 1
        assert citations[0]["type"] == "document"
        assert citations[0]["source"] == "08_quantization_techniques.md"

    def test_falls_back_to_retrieved_docs_if_no_reranked_docs(self):
        state = {
            "original_query": "q",
            "chat_history": [],
            "tools_selected": ["docs"],
            "reranked_docs": [],
            "retrieved_docs": [(self._doc(), 5)],
            "specs_result": {},
            "web_results": [],
        }
        _, citations = build_generation_prompt(state)
        assert len(citations) == 1

    def test_specs_produces_specs_db_citations(self):
        state = {
            "original_query": "What is the H100 bandwidth?",
            "chat_history": [],
            "tools_selected": ["specs"],
            "reranked_docs": [],
            "retrieved_docs": [],
            "specs_result": {
                "sql": "SELECT * FROM gpus",
                "rows": [{"model": "NVIDIA H100 80GB SXM", "memory_bandwidth_gbps": 3350.0, "source_url": "https://example.com"}],
                "error": None,
            },
            "web_results": [],
        }
        prompt, citations = build_generation_prompt(state)
        assert "3350" in prompt
        assert len(citations) == 1
        assert citations[0]["type"] == "specs_db"
        assert citations[0]["source_url"] == "https://example.com"

    def test_web_produces_web_citations(self):
        state = {
            "original_query": "current news",
            "chat_history": [],
            "tools_selected": ["web"],
            "reranked_docs": [],
            "retrieved_docs": [],
            "specs_result": {},
            "web_results": [{"title": "Some Site", "url": "https://example.com/x", "content": "content here"}],
        }
        prompt, citations = build_generation_prompt(state)
        assert len(citations) == 1
        assert citations[0]["type"] == "web"
        assert citations[0]["source_url"] == "https://example.com/x"

    def test_multi_tool_combines_citations_from_both_sources(self):
        state = {
            "original_query": "compare docs and specs",
            "chat_history": [],
            "tools_selected": ["docs", "specs"],
            "reranked_docs": [(self._doc(), 8)],
            "retrieved_docs": [],
            "specs_result": {
                "sql": "SELECT * FROM gpus",
                "rows": [{"model": "NVIDIA A100 80GB SXM", "source_url": ""}],
                "error": None,
            },
            "web_results": [],
        }
        _, citations = build_generation_prompt(state)
        types = {c["type"] for c in citations}
        assert types == {"document", "specs_db"}

    def test_instructs_section_level_citation_via_numbered_sources(self):
        """
        Regression test for a real gap flagged against 2026 production RAG
        best practices: 'the minimum bar for production RAG is source
        attribution at the claim level... not according to company
        documents, but according to the Q3 Financial Review, page 12.'
        Our citation instruction previously only asked for [filename.md],
        even though section-level metadata was already being sent in the
        context -- this data was available but unused in the instruction.

        This was first fixed by asking the LLM to cite [filename.md §
        section] directly, then superseded by pre-assigned numbered
        citations ([1], [2], ...) for reliability (same "structured
        output over free-text generation" principle as the router's tool
        selection). The section-level distinguishing data is still fully
        present -- in the context shown to the LLM and in the structured
        citation list the UI renders -- just referenced by number in the
        answer text rather than spelled out inline by the LLM itself.
        """
        state = {
            "original_query": "q",
            "chat_history": [],
            "tools_selected": ["docs"],
            "reranked_docs": [(self._doc(), 5)],
            "retrieved_docs": [],
            "specs_result": {},
            "web_results": [],
        }
        prompt, citations = build_generation_prompt(state)
        # section data still reaches the LLM's context
        assert "section" in prompt.lower()
        # the LLM is told to cite via number, not to format its own citation string
        assert "[1]" in prompt or "[N]" in prompt
        assert "filename.md]" not in prompt  # old, coarser LLM-formatted style should be gone
        # but the structured citation data (used by the UI) still has section-level detail
        assert citations[0]["section"] != ""

    def test_two_chunks_same_file_different_sections_are_distinguishable(self):
        """
        The concrete scenario the gap was about: if two claims in one
        answer come from different sections of the SAME file, a reader
        needs to be able to tell which section backs which claim -- both
        in what the LLM is shown (context) and in the structured citation
        data used by the UI panel.
        """
        doc1 = Document(
            page_content="Quantization reduces memory bandwidth needs.",
            metadata={"source": "08_quantization_techniques.md", "section": "Effects on inference"},
        )
        doc2 = Document(
            page_content="GPTQ and AWQ use calibration data to minimize error.",
            metadata={"source": "08_quantization_techniques.md", "section": "Post-training quantization"},
        )
        state = {
            "original_query": "How does quantization work and affect inference?",
            "chat_history": [],
            "tools_selected": ["docs"],
            "reranked_docs": [(doc1, 8), (doc2, 7)],
            "retrieved_docs": [],
            "specs_result": {},
            "web_results": [],
        }
        prompt, citations = build_generation_prompt(state)

        # both distinct sections appear in what the LLM sees
        assert "Effects on inference" in prompt
        assert "Post-training quantization" in prompt

        # both distinct sections appear in the structured citation data,
        # not collapsed into one entry per filename -- and they get
        # DIFFERENT numbers since they're genuinely different sources
        assert len(citations) == 2
        assert citations[0]["number"] != citations[1]["number"]
        assert citations[0]["source"] == citations[1]["source"]  # same file
        assert citations[0]["section"] != citations[1]["section"]  # different sections

    def test_citation_numbers_start_at_one_and_are_sequential(self):
        docs = [
            (self._doc(source="a.md", content="a"), 8),
            (self._doc(source="b.md", content="b"), 7),
            (self._doc(source="c.md", content="c"), 6),
        ]
        state = {
            "original_query": "q", "chat_history": [], "tools_selected": ["docs"],
            "reranked_docs": docs, "retrieved_docs": [],
            "specs_result": {}, "web_results": [],
        }
        _, citations = build_generation_prompt(state)
        numbers = [c["number"] for c in citations]
        assert numbers == [1, 2, 3]

    def test_duplicate_source_reuses_same_number_not_a_new_one(self):
        """
        The same (source, section) appearing twice in the retrieved docs
        (e.g. returned by both dense and sparse retrieval, deduped
        upstream but hypothetically present here) must reuse its number,
        not get counted as a second citation.
        """
        doc_a = self._doc(source="08_quantization_techniques.md", content="First mention.")
        doc_a_again = self._doc(source="08_quantization_techniques.md", content="Second mention, same source+section.")
        doc_b = self._doc(source="09_inference_optimization.md", content="Different source.")

        state = {
            "original_query": "q", "chat_history": [], "tools_selected": ["docs"],
            "reranked_docs": [(doc_a, 8), (doc_a_again, 7), (doc_b, 6)],
            "retrieved_docs": [], "specs_result": {}, "web_results": [],
        }
        _, citations = build_generation_prompt(state)
        # only 2 unique citations, not 3 -- doc_a and doc_a_again share a number
        assert len(citations) == 2
        assert citations[0]["number"] == 1
        assert citations[1]["number"] == 2

    def test_numbers_continue_sequentially_across_docs_specs_web(self):
        """
        Numbering must not reset per tool -- docs get 1..N, then specs
        continues N+1.., then web continues after that, all in one
        sequence, since the LLM references them as one flat numbered list.
        """
        state = {
            "original_query": "q",
            "chat_history": [],
            "tools_selected": ["docs", "specs", "web"],
            "reranked_docs": [(self._doc(), 8)],
            "retrieved_docs": [],
            "specs_result": {
                "sql": "SELECT * FROM gpus",
                "rows": [
                    {"model": "NVIDIA A100 80GB SXM", "source_url": ""},
                    {"model": "NVIDIA H100 80GB SXM", "source_url": ""},
                ],
                "error": None,
            },
            "web_results": [{"title": "News", "url": "https://example.com", "content": "x"}],
        }
        _, citations = build_generation_prompt(state)
        numbers = [c["number"] for c in citations]
        assert numbers == [1, 2, 3, 4]
        types_in_order = [c["type"] for c in citations]
        assert types_in_order == ["document", "specs_db", "specs_db", "web"]

    def test_specs_error_does_not_crash_and_produces_no_specs_citation(self):
        state = {
            "original_query": "q",
            "chat_history": [],
            "tools_selected": ["specs"],
            "reranked_docs": [],
            "retrieved_docs": [],
            "specs_result": {"sql": "SELECT bad", "rows": [], "error": "syntax error"},
            "web_results": [],
        }
        prompt, citations = build_generation_prompt(state)
        assert "syntax error" in prompt
        assert citations == []

    def test_instructs_citation_and_honesty(self):
        # regression-test that the anti-hallucination instruction survived
        # the extraction refactor unchanged
        state = {
            "original_query": "q",
            "chat_history": [],
            "tools_selected": ["docs"],
            "reranked_docs": [(self._doc(), 5)],
            "retrieved_docs": [],
            "specs_result": {},
            "web_results": [],
        }
        prompt, _ = build_generation_prompt(state)
        assert "cite" in prompt.lower()
        assert "say so" in prompt.lower() or "insufficient" in prompt.lower() or "don't contain enough" in prompt.lower()
