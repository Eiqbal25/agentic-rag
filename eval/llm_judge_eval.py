"""
RAGAS-style generation-quality metrics, implemented as direct LLM-judge
calls rather than via the `ragas` package.

WHY NOT THE `ragas` PACKAGE: it was evaluated and installed during
development, but `ragas==0.4.3` imports
`langchain_community.chat_models.vertexai`, a submodule that no longer
exists in current `langchain-community` (0.4.x) -- langchain-community is
being actively sunset and vertexai-specific integrations were split into
a separate package. This is a real, current packaging incompatibility,
not a hypothetical one (`ModuleNotFoundError` reproduced directly). Since
RAGAS's actual metrics are themselves just structured LLM-judge prompts
(confirmed via their published methodology: an LLM decomposes the answer
into claims and checks each against retrieved context for faithfulness;
an LLM judges topical relevance for answer relevancy), reimplementing the
two metrics directly avoids the dependency risk entirely while measuring
the same thing, using the same Groq LLM already in the pipeline rather
than requiring a second, differently-configured judge model.

Metrics implemented:
  - Faithfulness: are the claims in the answer actually supported by the
    retrieved context, or did the model add unsupported content?
  - Answer relevancy: does the answer actually address the question that
    was asked, independent of whether it's grounded?

These are complementary and catch different failures: a faithful answer
to the wrong question is still useless (low relevancy, high faithfulness);
a relevant-sounding answer with fabricated specifics is dangerous (high
apparent relevancy, low faithfulness) -- this is exactly the failure mode
caught live during testing (the hallucinated sourdough recipe before the
routing fix: on-topic and fluent, zero grounding).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_utils import invoke_with_retry  # noqa: E402


def score_faithfulness(question: str, answer: str, context: str, llm) -> dict:
    """
    Decomposes the answer into individual factual claims and checks each
    one against the provided context. Returns a score in [0, 1] plus the
    list of unsupported claims (empty list = fully faithful).
    """
    prompt = (
        "You are evaluating whether an AI-generated answer is faithful "
        "to its source context (i.e. every factual claim in the answer "
        "is actually supported by the context, with nothing fabricated "
        "or added from outside knowledge).\n\n"
        f"Question: {question}\n\n"
        f"Context provided to the model:\n{context}\n\n"
        f"Generated answer:\n{answer}\n\n"
        "List each distinct factual claim in the answer on its own line, "
        "formatted exactly as:\n"
        "CLAIM: <the claim> | SUPPORTED or UNSUPPORTED\n\n"
        "A claim is SUPPORTED only if the context directly states it or "
        "directly implies it. General framing, hedging language, or "
        "explicit statements that information is missing don't count as "
        "claims to evaluate. If the answer makes no factual claims at "
        "all (e.g. it's a refusal or says information is unavailable), "
        "respond with exactly: NO_CLAIMS"
    )
    resp = invoke_with_retry(llm, prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    if "NO_CLAIMS" in text:
        return {"score": None, "unsupported_claims": [], "total_claims": 0}

    lines = re.findall(r"CLAIM:\s*(.+?)\s*\|\s*(SUPPORTED|UNSUPPORTED)", text, re.I)
    if not lines:
        return {"score": None, "unsupported_claims": [], "total_claims": 0}

    total = len(lines)
    unsupported = [claim for claim, verdict in lines if verdict.upper() == "UNSUPPORTED"]
    score = (total - len(unsupported)) / total
    return {"score": score, "unsupported_claims": unsupported, "total_claims": total}


def score_answer_relevancy(question: str, answer: str, llm) -> dict:
    """
    Judges whether the answer actually addresses the question asked,
    independent of whether it's grounded. Returns a score in [0, 1].
    """
    prompt = (
        "Rate how directly this answer addresses the question, on a "
        "scale of 0-10. A high score means the answer is on-topic and "
        "responds to what was actually asked. A low score means the "
        "answer is off-topic, evasive, or addresses a different question "
        "than the one asked -- note that an honest 'I don't have enough "
        "information' response still scores HIGH here if the question "
        "genuinely required information the answer correctly identifies "
        "as unavailable; relevancy is about topical alignment, not "
        "completeness.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Respond with ONLY the integer score, nothing else."
    )
    resp = invoke_with_retry(llm, prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    match = re.search(r"\d+", text)
    raw_score = int(match.group()) if match else 5
    return {"score": min(raw_score, 10) / 10}


def build_context_string(result: dict) -> str:
    """Reconstructs the context string actually available to `generate`
    from a graph invocation's result dict, matching what generate() saw."""
    parts = []
    for c in result.get("citations", []):
        if c["type"] == "document":
            parts.append(f"[{c['source']}] {c['snippet']}")
        elif c["type"] == "specs_db":
            parts.append(f"[specs: {c['source']}] {c['snippet']}")
        elif c["type"] == "web":
            parts.append(f"[{c['source']}] {c['snippet']}")
    return "\n\n".join(parts) if parts else "(no context -- direct answer)"
