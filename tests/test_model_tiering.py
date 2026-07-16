"""
Unit tests for src/graph.py's _build_default_fast_llm -- auto-configures
a fast_llm tuned for structured micro-decisions (routing, reranking,
grading, SQL generation) using Groq's reasoning_effort parameter.

Added after a live bug: a GPU comparison question about models NOT in the
specs database caused qwen3.6-27b's text-to-SQL call to spend its ENTIRE
token budget reasoning inside a <think> block about whether the model
names might be typos, and got cut off before producing any SQL at all.
The fix disables/minimizes reasoning specifically for structured
micro-decisions, which never needed open-ended chain-of-thought.

Requires a fake GROQ_API_KEY to construct ChatGroq instances (no real API
calls are made -- construction alone doesn't hit the network).
"""

import os

import pytest
from langchain_groq import ChatGroq

os.environ.setdefault("GROQ_API_KEY", "fake-test-key-construction-only")

from agent.llm_factory import _build_default_fast_llm  # noqa: E402


class TestBuildDefaultFastLlm:
    def test_qwen_model_gets_reasoning_disabled(self):
        llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert getattr(fast, "reasoning_effort", None) == "none"

    def test_qwen_fast_llm_is_a_distinct_instance(self):
        llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert fast is not llm

    def test_gpt_oss_model_gets_low_reasoning_effort(self):
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert getattr(fast, "reasoning_effort", None) == "low"

    def test_gpt_oss_20b_also_gets_low_reasoning_effort(self):
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert getattr(fast, "reasoning_effort", None) == "low"

    def test_unrecognized_model_falls_back_to_same_instance(self):
        # deprecated/unknown model family -- don't guess at a
        # reasoning_effort value that might not even be valid for it
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert fast is llm

    def test_fast_llm_preserves_the_same_model_name(self):
        # tiering changes HOW the model reasons, not WHICH model is used
        llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert getattr(fast, "model_name", None) == "qwen/qwen3.6-27b"

    def test_missing_api_key_falls_back_safely(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.0, api_key="fake")
        fast = _build_default_fast_llm(llm)
        assert fast is llm  # can't construct a second client without a key
