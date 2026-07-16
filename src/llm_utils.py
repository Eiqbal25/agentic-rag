"""
Shared helpers for calling an LLM: retry-on-rate-limit backoff, and
filtering out <think>...</think> reasoning blocks from model output.

Why the thinking-tag filtering exists: reproduced live -- qwen3.6-27b
(and reasoning models generally) can emit internal chain-of-thought
wrapped in <think>...</think> tags as part of the visible response
content, not hidden metadata. Two real bugs from this, found in a single
live test of a plain "hi" greeting:
  1. The final answer shown to the user included the raw <think> block
     before the actual answer, instead of just the answer.
  2. (Checked separately, not yet seen live but a real risk): the
     rewritten search query in the corrective retry loop uses raw model
     text directly as a retrieval query -- if that also contains a
     <think> block, it would badly pollute the query sent to the
     retriever, not just look bad on screen.
Routing (parse_tool_selection) and grading (parse_grade_response) were
checked and are NOT affected -- both use regex search for a pattern
anywhere in the text, which still finds the JSON/GRADE line correctly
even with a <think> block surrounding it. Only places that use raw
response text AS-IS (the final answer, the rewritten query) needed a fix.
"""

import re
import time


def invoke_with_retry(llm, prompt: str, max_retries: int = 4, base_delay: float = 2.0):
    """
    Calls llm.invoke(prompt), retrying with exponential backoff on rate
    limit errors. Respects the server's suggested wait time when the
    error message includes one (Groq's 429 responses include "try again
    in Xs"); falls back to exponential backoff otherwise.

    Re-raises the original exception if max_retries is exhausted, so
    callers still see a real failure rather than this silently returning
    something wrong.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
            if not is_rate_limit or attempt == max_retries:
                raise
            last_error = e

            # Respect the server's suggested retry delay if present
            # (Groq's 429 body includes e.g. "Please try again in 2.0475s")
            match = re.search(r"try again in (\d+(?:\.\d+)?)s", error_str)
            if match:
                delay = float(match.group(1)) + 0.5  # small safety margin
            else:
                delay = base_delay * (2**attempt)

            time.sleep(delay)
    raise last_error  # pragma: no cover -- unreachable, loop always returns or raises


def strip_thinking_tags(text: str) -> str:
    """
    Removes <think>...</think> reasoning blocks from a complete (non-
    streaming) response. Handles multiple blocks and is case-insensitive.

    Also handles an UNCLOSED <think> tag (the model was cut off mid-
    reasoning, e.g. hit a token limit) by stripping everything from the
    unclosed tag onward -- showing a dangling, incomplete reasoning
    fragment as if it were the answer would be worse than an empty/
    shorter result.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def stream_without_thinking(chunk_iterator):
    """
    Wraps a stream of text chunks (e.g. from llm.stream()), filtering out
    <think>...</think> content so only the actual answer is yielded to
    the caller (e.g. st.write_stream()).

    This has to be more careful than strip_thinking_tags: chunks arrive
    incrementally, and a tag like "<think>" can be split across multiple
    chunks (e.g. one chunk ends in "<th" and the next starts with
    "ink>"). A small buffer tail is held back at all times so a
    tag boundary is never missed just because it happened to fall on a
    chunk boundary.
    """
    TAG_HOLDBACK = 20  # longer than "<think>" or "</think>", with margin
    buffer = ""
    in_thinking = False

    for chunk in chunk_iterator:
        buffer += chunk
        while True:
            if not in_thinking:
                start_idx = buffer.lower().find("<think>")
                if start_idx == -1:
                    if len(buffer) > TAG_HOLDBACK:
                        yield buffer[:-TAG_HOLDBACK]
                        buffer = buffer[-TAG_HOLDBACK:]
                    break
                else:
                    if start_idx > 0:
                        yield buffer[:start_idx]
                    buffer = buffer[start_idx + len("<think>"):]
                    in_thinking = True
            else:
                end_idx = buffer.lower().find("</think>")
                if end_idx == -1:
                    if len(buffer) > TAG_HOLDBACK:
                        buffer = buffer[-TAG_HOLDBACK:]
                    break
                else:
                    buffer = buffer[end_idx + len("</think>"):]
                    in_thinking = False

    # Flush whatever's left in the buffer once the stream ends -- unless
    # we're still inside an unclosed <think> block, in which case discard
    # it (same reasoning as strip_thinking_tags: don't show a dangling
    # incomplete reasoning fragment as if it were the answer).
    if not in_thinking and buffer:
        yield buffer
