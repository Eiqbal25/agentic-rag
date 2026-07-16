"""
Unit tests for src/llm_utils.py -- retry-with-backoff wrapper for LLM
calls, and <think>-tag filtering for reasoning models.

The retry tests were added after a live run hit groq.RateLimitError
mid-eval (Groq free-tier tokens-per-minute cap, exceeded by the eval
suite's call volume across 16 test cases).

The thinking-tag tests were added after a live run showed qwen3.6-27b's
internal <think>...</think> reasoning leaking into the displayed answer
on a plain "hi" greeting -- confirmed live, not hypothetical.
"""

import time

import pytest

from llm_utils import invoke_with_retry, strip_thinking_tags, stream_without_thinking


class FakeResp:
    def __init__(self, content):
        self.content = content


class FlakyLLM:
    """Fails with a rate-limit-shaped error N times, then succeeds."""

    def __init__(self, fail_times: int, delay_hint: str = "0.01s"):
        self.calls = 0
        self.fail_times = fail_times
        self.delay_hint = delay_hint

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception(
                f"Error code: 429 - rate_limit_exceeded. Please try again in {self.delay_hint}"
            )
        return FakeResp("success")


class AlwaysFailsWithRateLimit:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise Exception("Error code: 429 - rate_limit_exceeded, try again in 0.01s")


class AlwaysFailsWithOtherError:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise ValueError("some unrelated real bug, not a rate limit")


class TestInvokeWithRetry:
    def test_succeeds_immediately_when_no_error(self):
        llm = FlakyLLM(fail_times=0)
        resp = invoke_with_retry(llm, "prompt")
        assert resp.content == "success"
        assert llm.calls == 1

    def test_retries_and_eventually_succeeds(self):
        llm = FlakyLLM(fail_times=2, delay_hint="0.01s")
        resp = invoke_with_retry(llm, "prompt", max_retries=4)
        assert resp.content == "success"
        assert llm.calls == 3  # 2 failures + 1 success

    def test_non_rate_limit_error_is_not_retried(self):
        llm = AlwaysFailsWithOtherError()
        with pytest.raises(ValueError):
            invoke_with_retry(llm, "prompt", max_retries=3)
        assert llm.calls == 1  # no retries attempted

    def test_exhausting_retries_reraises_original_error(self):
        llm = AlwaysFailsWithRateLimit()
        with pytest.raises(Exception, match="rate_limit_exceeded"):
            invoke_with_retry(llm, "prompt", max_retries=2)
        assert llm.calls == 3  # initial attempt + 2 retries

    def test_respects_server_suggested_delay(self):
        llm = FlakyLLM(fail_times=1, delay_hint="0.05s")
        t0 = time.time()
        invoke_with_retry(llm, "prompt", max_retries=2)
        elapsed = time.time() - t0
        # should wait roughly the suggested 0.05s (+ 0.5s safety margin),
        # not the much larger exponential-backoff default
        assert elapsed < 2.0


class TestStripThinkingTags:
    """
    Regression tests for a live bug: a plain 'hi' greeting produced a
    displayed answer that included the model's raw <think>...</think>
    reasoning before the actual answer, because nothing stripped it.
    """

    def test_removes_a_single_think_block(self):
        text = "<think>internal reasoning here</think>Hi there!"
        assert strip_thinking_tags(text) == "Hi there!"

    def test_passthrough_when_no_think_tag_present(self):
        text = "Just a plain answer, no reasoning block at all."
        assert strip_thinking_tags(text) == text

    def test_removes_multiple_think_blocks(self):
        text = "<think>first</think>Middle<think>second</think>End"
        assert strip_thinking_tags(text) == "MiddleEnd"

    def test_case_insensitive_tags(self):
        text = "<THINK>reasoning</THINK>Answer"
        assert strip_thinking_tags(text) == "Answer"

    def test_unclosed_think_tag_strips_to_empty(self):
        # model was cut off mid-reasoning (e.g. hit a token limit) --
        # a dangling incomplete reasoning fragment shouldn't be shown as
        # if it were the answer
        text = "<think>reasoning that never closes because it got cut off"
        assert strip_thinking_tags(text) == ""

    def test_strips_surrounding_whitespace(self):
        text = "  <think>x</think>  Answer with padding  "
        assert strip_thinking_tags(text) == "Answer with padding"

    def test_multiline_think_block(self):
        text = "<think>\nline one\nline two\n</think>\nFinal answer here."
        assert strip_thinking_tags(text) == "Final answer here."


class TestStreamWithoutThinking:
    """
    Regression tests for the streaming variant of the same bug. Harder
    than the non-streaming case: a tag like "<think>" can be split across
    multiple chunks (e.g. one chunk ends "<th", the next starts "ink>"),
    so these tests specifically probe different chunk-size splits of the
    same text to make sure tag boundaries are never missed just because
    they happened to fall across a chunk boundary.
    """

    @staticmethod
    def _chunks(s: str, size: int):
        for i in range(0, len(s), size):
            yield s[i : i + size]

    def _run(self, text: str, chunk_size: int) -> str:
        return "".join(stream_without_thinking(self._chunks(text, chunk_size)))

    def test_think_tag_in_one_large_chunk(self):
        text = "<think>internal reasoning</think>Hi there! How can I help?"
        assert self._run(text, 1000) == "Hi there! How can I help?"

    def test_think_tag_split_char_by_char(self):
        # worst case: every chunk is a single character, so every tag
        # boundary falls on a chunk boundary
        text = "<think>internal reasoning</think>Hi there! How can I help?"
        assert self._run(text, 1) == "Hi there! How can I help?"

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 7, 10, 13])
    def test_various_chunk_sizes_all_produce_correct_output(self, chunk_size):
        text = "<think>reasoning block</think>The actual answer text."
        assert self._run(text, chunk_size) == "The actual answer text."

    def test_no_think_tag_passes_through_unchanged(self):
        text = "Just a normal streamed answer with no reasoning block."
        assert self._run(text, 5) == text

    def test_multiple_think_blocks_in_stream(self):
        text = "<think>one</think>Part A<think>two</think>Part B"
        assert self._run(text, 4) == "Part APart B"

    def test_content_before_think_tag_is_preserved(self):
        # some models might emit a little text before starting to think
        text = "Sure, let me consider this. <think>reasoning</think>Final answer."
        assert self._run(text, 6) == "Sure, let me consider this. Final answer."

    def test_unclosed_think_tag_at_end_of_stream_yields_nothing_after(self):
        text = "Some text<think>reasoning that never closes"
        # content before the tag is still yielded; the unclosed block is discarded
        assert self._run(text, 5) == "Some text"
