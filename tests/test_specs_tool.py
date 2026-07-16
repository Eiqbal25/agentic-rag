"""
Unit tests for src/specs_tool.py -- SQL validation guardrails, and the
exact-match -> LIKE fallback that fixed the live 'H100 SXM returns 0 rows'
bug.
"""

import pytest

from tools.specs_tool import (
    SpecsQueryError,
    _loosen_exact_match_to_like,
    query_specs_db,
    run_specs_query,
    validate_sql,
)


class TestValidateSql:
    def test_allows_plain_select(self):
        result = validate_sql("SELECT model FROM gpus")
        assert result.startswith("SELECT model FROM gpus")

    def test_appends_limit_when_missing(self):
        result = validate_sql("SELECT * FROM ssds")
        assert "LIMIT" in result.upper()

    def test_preserves_existing_limit(self):
        result = validate_sql("SELECT * FROM ssds LIMIT 5")
        assert result.upper().count("LIMIT") == 1
        assert "LIMIT 5" in result

    @pytest.mark.parametrize(
        "malicious_sql",
        [
            "DROP TABLE gpus",
            "UPDATE gpus SET vram_gb=0",
            "DELETE FROM gpus",
            "ALTER TABLE gpus ADD COLUMN hacked TEXT",
            "PRAGMA table_info(gpus)",
            "ATTACH DATABASE 'x.db' AS x",
            "CREATE TABLE evil (id INT)",
        ],
    )
    def test_blocks_non_select_statements(self, malicious_sql):
        with pytest.raises(SpecsQueryError):
            validate_sql(malicious_sql)

    def test_blocks_multi_statement_injection(self):
        with pytest.raises(SpecsQueryError):
            validate_sql("SELECT * FROM gpus; DROP TABLE gpus")

    def test_blocks_select_disguising_a_write(self):
        # forbidden keyword inside an otherwise SELECT-shaped string
        with pytest.raises(SpecsQueryError):
            validate_sql("SELECT * FROM gpus WHERE 1=1; UPDATE gpus SET vram_gb=0")


class TestRunSpecsQuery:
    def test_real_query_returns_expected_columns(self):
        rows = run_specs_query("SELECT model, vram_gb FROM gpus WHERE vram_gb > 60")
        assert len(rows) > 0
        assert all("model" in r and "vram_gb" in r for r in rows)
        assert all(r["vram_gb"] > 60 for r in rows)

    def test_injection_attempt_raises_before_execution(self):
        with pytest.raises(SpecsQueryError):
            run_specs_query("DROP TABLE gpus")
        # verify the table is genuinely untouched
        rows = run_specs_query("SELECT model FROM gpus")
        assert len(rows) > 0


class TestLoosenExactMatchToLike:
    """
    Regression tests for the bug found live: 'WHERE model = "H100 SXM"'
    returned 0 rows because the real stored value is
    'NVIDIA H100 80GB SXM' -- the short name isn't a contiguous substring.
    """

    def test_detects_exact_match_pattern(self):
        sql = "SELECT * FROM gpus WHERE model = 'H100 SXM'"
        loosened = _loosen_exact_match_to_like(sql)
        assert loosened is not None
        assert "LIKE" in loosened
        assert "H100" in loosened and "SXM" in loosened

    def test_returns_none_when_no_exact_match_present(self):
        sql = "SELECT * FROM gpus WHERE vram_gb > 60"
        assert _loosen_exact_match_to_like(sql) is None

    def test_loosened_query_actually_finds_the_row(self):
        # this is the exact regression case from the live bug report
        sql = "SELECT memory_bandwidth_gbps FROM gpus WHERE model = 'H100 SXM'"
        loosened = _loosen_exact_match_to_like(sql)
        rows = run_specs_query(loosened)
        assert len(rows) == 1
        assert rows[0]["memory_bandwidth_gbps"] == 3350.0

    def test_words_are_and_joined_not_or(self):
        # AND-joined per-word LIKE clauses should NOT match a GPU that
        # only contains one of the two words
        sql = "SELECT * FROM gpus WHERE model = 'H100 RTX'"  # nonsense combo
        loosened = _loosen_exact_match_to_like(sql)
        rows = run_specs_query(loosened)
        assert len(rows) == 0  # no GPU model contains both "H100" and "RTX"


class FakeResp:
    def __init__(self, content):
        self.content = content


class TestQuerySpecsDbThinkingTags:
    """
    Regression tests for a live bug: a question comparing two GPU models
    NOT in the specs database ("Adakah gtx1050 lebih baik dari rtx5060")
    caused the model to spend its entire response reasoning inside a
    <think> block about whether the names might be typos -- and get cut
    off before ever producing any SQL at all. The safety net (validate_sql
    rejecting non-SELECT text) worked correctly and no data was
    hallucinated, but query_specs_db was missing the same
    strip_thinking_tags() fix already applied elsewhere in the codebase.
    """

    def test_strips_think_block_when_sql_follows_it(self):
        class FakeLLM:
            def invoke(self, prompt):
                return FakeResp(
                    "<think>Let me consider the schema...</think>"
                    "SELECT model FROM gpus WHERE model LIKE '%H100%'"
                )

        result = query_specs_db("What GPUs do we have?", FakeLLM())
        assert result["sql"].startswith("SELECT")
        assert "<think>" not in result["sql"]
        assert result["error"] is None

    def test_unclosed_think_block_with_no_sql_fails_validation_not_execution(self):
        """
        The exact failure mode reproduced live: the model never got past
        thinking, so there's no SQL to extract even after stripping.
        This should fail SAFELY (validate_sql rejects it, error is set,
        zero rows) rather than crash or execute garbage as SQL.
        """

        class FakeLLM:
            def invoke(self, prompt):
                return FakeResp(
                    "<think>The user is asking about gtx1050 and rtx5060, "
                    "neither of which exists in the gpus table... "
                    "let me consider whether this might be a typo for "
                    "one of the existing models... this requires careful "
                    "consideration of the available data before I can "
                    "determine the appropriate query structure to use"
                    # deliberately no closing tag, no SQL -- cut off mid-thought
                )

        result = query_specs_db(
            "Adakah gtx1050 lebih baik dari rtx5060", FakeLLM()
        )
        assert result["rows"] == []
        assert result["error"] is not None  # fails validation, not silently

    def test_no_think_tag_at_all_still_works_normally(self):
        class FakeLLM:
            def invoke(self, prompt):
                return FakeResp("SELECT model FROM gpus WHERE vram_gb > 60")

        result = query_specs_db("Which GPUs have more than 60GB VRAM?", FakeLLM())
        assert result["error"] is None
        assert len(result["rows"]) > 0
