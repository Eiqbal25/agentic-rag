"""
Unit tests for the LLM-output parsing functions in src/graph.py.

These were extracted from inline closures specifically to make them
unit-testable without a real or stubbed LLM -- malformed/unexpected LLM
output is a real failure mode (wrong case, extra whitespace, invalid
tokens, missing expected format) and worth covering directly rather than
only through end-to-end graph runs.
"""

from agent import parse_grade_response, parse_tool_selection


class TestParseToolSelection:
    def test_single_tool(self):
        assert parse_tool_selection("docs") == ["docs"]

    def test_multiple_tools(self):
        assert parse_tool_selection("docs,specs") == ["docs", "specs"]

    def test_handles_whitespace_around_commas(self):
        assert parse_tool_selection("docs, specs , web") == ["docs", "specs", "web"]

    def test_handles_uppercase(self):
        assert parse_tool_selection("DOCS,WEB") == ["docs", "web"]

    def test_none_returns_empty_list(self):
        assert parse_tool_selection("none") == []

    def test_empty_string_returns_empty_list(self):
        assert parse_tool_selection("") == []

    def test_filters_out_invalid_tokens(self):
        # LLM hallucinating a tool name that doesn't exist shouldn't crash
        # or silently get treated as a valid selection
        assert parse_tool_selection("docs,database,web") == ["docs", "web"]

    def test_ignores_trailing_punctuation_noise(self):
        # LLM adding an explanation despite instructions not to
        result = parse_tool_selection("docs, specs.")
        assert "docs" in result
        assert "specs" not in result  # "specs." with punctuation is correctly rejected, not silently accepted

    def test_all_three_tools(self):
        assert parse_tool_selection("docs,specs,web") == ["docs", "specs", "web"]

    # --- JSON layer (current primary format) ---

    def test_clean_json_single_tool(self):
        assert parse_tool_selection('{"tools": ["docs"]}') == ["docs"]

    def test_clean_json_multiple_tools(self):
        assert parse_tool_selection('{"tools": ["docs", "specs"]}') == ["docs", "specs"]

    def test_json_explicit_empty_list_means_none(self):
        assert parse_tool_selection('{"tools": []}') == []

    def test_json_with_reasoning_text_before_it(self):
        text = "This question is about corrective RAG concepts.\n\n{\"tools\": [\"docs\"]}"
        assert parse_tool_selection(text) == ["docs"]

    def test_json_filters_invalid_tool_names(self):
        text = '{"tools": ["docs", "database"]}'
        assert parse_tool_selection(text) == ["docs"]

    def test_json_case_insensitive_values(self):
        text = '{"tools": ["DOCS", "Web"]}'
        assert parse_tool_selection(text) == ["docs", "web"]

    # --- ANSWER: line fallback ---

    def test_explicit_answer_line_format(self):
        text = "Some reasoning about the question.\n\nANSWER: docs"
        assert parse_tool_selection(text) == ["docs"]

    def test_answer_line_with_multiple_tools(self):
        text = "Reasoning here.\nANSWER: docs,specs"
        assert parse_tool_selection(text) == ["docs", "specs"]

    def test_answer_line_case_insensitive_label(self):
        text = "reasoning\nanswer: web"
        assert parse_tool_selection(text) == ["web"]

    def test_explicit_answer_none(self):
        assert parse_tool_selection("ANSWER: none") == []

    # --- Regression: v1 bug (silent tools=['none'] for everything) ---

    def test_regression_v1_bug_verbose_response_no_structured_format_at_all(self):
        """
        v1 regression: switching Groq models (gpt-oss-120b -> qwen3.6-27b)
        caused the router to return tools=['none'] for EVERY query,
        because the new model added explanation text despite being told
        to respond with ONLY the list, and the parser at the time
        required an exact clean comma-separated match against the whole
        response. Since this response has no JSON and no ANSWER: line, a
        parse failure here is EXPECTED and safe (empty list, not a wrong
        guess) -- the actual fix was the prompt now requesting JSON,
        which real models comply with far more reliably than a bare
        "respond with only X" instruction.
        """
        verbose_no_format_compliance = (
            "This question is about RAG concepts, so I will use the docs "
            "knowledge base to answer it."
        )
        # no structured marker present -> safe empty default, not a guess
        assert parse_tool_selection(verbose_no_format_compliance) == []

    # --- Regression: v2 bug (over-selection from blind keyword scan) ---

    def test_regression_v2_bug_does_not_overselect_from_reasoning_text(self):
        """
        v2 regression: an earlier fix added a fallback that scanned the
        ENTIRE response for any mention of docs/specs/web, with no
        negation-awareness. A model reasoning aloud about which tools it
        considered (even ones it explicitly ruled out) got ALL mentioned
        tools extracted -- reproduced live: simple docs-only questions
        and even out-of-scope questions started returning
        ['docs','specs','web']. This is the exact failure mode, fixed by
        removing the scan-anywhere fallback entirely in favor of JSON.
        """
        text = (
            "Let me think: this could need docs for the explanation, "
            "specs for hardware numbers, or web for current info. But "
            "this is really just about a documented concept, so:\n\n"
            '{"tools": ["docs"]}'
        )
        assert parse_tool_selection(text) == ["docs"]

    def test_regression_v2_bug_out_of_scope_question_stays_empty(self):
        text = (
            "This is a general knowledge question not covered by docs or "
            "specs, and doesn't need current information either.\n\n"
            '{"tools": []}'
        )
        assert parse_tool_selection(text) == []

    def test_total_parse_failure_defaults_to_safe_empty_not_a_guess(self):
        garbage = "I am not sure what tools to use for this, let me think about it more."
        assert parse_tool_selection(garbage) == []


class TestParseGradeResponse:
    def test_well_formatted_relevant(self):
        text = "GRADE: RELEVANT\nREASON: the passages directly answer the question"
        grade, reason = parse_grade_response(text)
        assert grade == "RELEVANT"
        assert "directly answer" in reason

    def test_well_formatted_irrelevant(self):
        text = "GRADE: IRRELEVANT\nREASON: passages are off-topic"
        grade, reason = parse_grade_response(text)
        assert grade == "IRRELEVANT"

    def test_case_insensitive_grade_keyword(self):
        text = "grade: relevant\nreason: fine"
        grade, reason = parse_grade_response(text)
        assert grade == "RELEVANT"

    def test_missing_grade_defaults_to_irrelevant_fail_safe(self):
        # this is a deliberate design choice: a parse failure must NOT
        # silently skip the correction loop by defaulting to RELEVANT
        text = "I think this is probably fine but I'm not totally sure."
        grade, reason = parse_grade_response(text)
        assert grade == "IRRELEVANT"

    def test_missing_reason_falls_back_to_truncated_text(self):
        text = "GRADE: RELEVANT"
        grade, reason = parse_grade_response(text)
        assert grade == "RELEVANT"
        assert reason  # non-empty fallback, not a crash

    def test_extra_surrounding_text_still_parses(self):
        text = "Sure, here's my assessment:\nGRADE: IRRELEVANT\nREASON: no overlap\nLet me know if you need more."
        grade, reason = parse_grade_response(text)
        assert grade == "IRRELEVANT"
        assert "no overlap" in reason
