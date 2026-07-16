"""
Parsers for the router's and grader's raw LLM responses into validated,
structured decisions.
"""

import json
import re


def parse_tool_selection(raw_text: str) -> list[str]:
    """
    Parses the router LLM's raw response into a validated list of tools.

    Three-layer fallback, most-reliable first:
      1. Find and parse a JSON object with a "tools" key (what the
         current prompt asks for). JSON has explicit structural
         delimiters, which is far less ambiguous to extract from
         surrounding prose than free-text keyword scanning.
      2. Fall back to an explicit "ANSWER: ..." comma-separated line, for
         models that ignore the JSON instruction but still follow a
         simpler format instruction.
      3. Fall back to treating the ENTIRE trimmed response as a bare
         comma-separated list (e.g. just "docs,specs" with nothing else).
         This is safe unlike scanning-anywhere (see history below):
         ordinary prose doesn't happen to consist ONLY of valid
         comma-separated single-word tokens, so this can't false-positive
         on a paragraph of reasoning the way a keyword scan can.

    If all three fail, this returns an empty list (no tools) rather than
    guessing. That's intentional and safe-by-design: the `generate` node
    has a separate safety net for the no-tools path that refuses to
    answer factual questions from unverified memory (see that node's
    docstring) -- so a genuine parse failure degrades to "ask the user to
    rephrase" rather than to a wrong guess.

    History of this function (two real regressions found live, in order):
      - v1 (only layer 3 existed, as the sole strategy): broke when a
        model added explanation text despite "respond with ONLY X" --
        returned tools=['none'] for every query, including ones that
        obviously needed `docs`, because the whole response was never a
        bare comma list once explanation text was present.
      - v2 (added a blind regex scan for tool-name words ANYWHERE in the
        text, as a last-resort fallback after v1's layer): this
        OVER-corrected -- a model reasoning aloud with e.g. "this doesn't
        need web search, just docs" has no negation-awareness in a
        keyword scan, so `web` got extracted anyway. Reproduced live:
        simple docs-only questions and even out-of-scope questions
        started returning ['docs','specs','web'] -- selecting
        everything, the opposite failure mode. That scan-anywhere layer
        was removed and replaced with the current JSON-first design,
        which doesn't need to guess at intent from prose at all.
    """
    text = raw_text.strip()

    # Layer 1: JSON object with a "tools" key, e.g. {"tools": ["docs"]}
    # Search for the LAST JSON-object-shaped substring, since models that
    # reason before answering put the structured decision at the end.
    json_matches = re.findall(r"\{[^{}]*\}", text)
    for candidate in reversed(json_matches):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "tools" in parsed:
            raw_tools = parsed["tools"]
            if isinstance(raw_tools, str):
                raw_tools = [raw_tools]
            if isinstance(raw_tools, list):
                tools = [
                    str(t).strip().lower()
                    for t in raw_tools
                    if str(t).strip().lower() in ("docs", "specs", "web")
                ]
                return tools  # valid JSON found; trust it even if empty (explicit "no tools")

    # Layer 2: explicit "ANSWER: ..." comma-separated line
    answer_match = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).strip().lower()
        tools = [t.strip() for t in candidate.split(",") if t.strip() in ("docs", "specs", "web")]
        if tools:
            return tools

    # Layer 3: the ENTIRE response is a clean comma-separated list, e.g.
    # "docs,specs" with nothing else. Safe unlike a scan-anywhere: a
    # whole-string split can't false-positive on prose the way scanning
    # for keywords inside a longer paragraph did (see docstring) --
    # ordinary sentences don't happen to consist ONLY of valid
    # comma-separated single-word tokens.
    whole_lower = text.lower()
    tools = [t.strip() for t in whole_lower.split(",") if t.strip() in ("docs", "specs", "web")]
    if tools:
        return tools

    # All structured layers failed to find a parseable decision.
    # Deliberately return an empty list rather than guessing via a blind
    # keyword scan (see this function's docstring for why that was tried
    # and reverted). The `generate` node's no-tools path refuses to
    # answer factual questions from memory, so this degrades safely to
    # "ask the user to rephrase" instead of a wrong guess.
    return []


def parse_grade_response(raw_text: str) -> tuple[str, str]:
    """
    Parses the grading LLM's raw response into (grade, reason).

    Defaults to IRRELEVANT if the expected "GRADE: ..." format isn't
    found -- fail-safe rather than fail-open, since defaulting to
    RELEVANT on a parse failure would mean malformed LLM output silently
    skips the correction loop instead of triggering it.
    """
    grade_match = re.search(r"GRADE:\s*(RELEVANT|IRRELEVANT)", raw_text, re.I)
    reason_match = re.search(r"REASON:\s*(.+)", raw_text, re.I)
    grade = grade_match.group(1).upper() if grade_match else "IRRELEVANT"
    reason = reason_match.group(1).strip() if reason_match else raw_text[:200]
    return grade, reason
