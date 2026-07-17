"""
Head-to-head comparison: traditional vs. agentic RAG mode, run on the
SAME questions, with the SAME retriever/reranker/LLM, measuring actual
outcomes -- not a single anecdotal example.

This exists because a table of mechanism differences (routing? grading?
retry?) tells you the two modes ARE different, but not which one performs
better, or by how much. This script answers that with real per-question
scores, aggregated across the docs-relevant subset of the test set.

Only single_hop / multi_hop / adversarial / out_of_scope cases are used:
traditional mode structurally can't reach specs/web/memory, so including
those categories wouldn't be a fair comparison -- it would just show
"traditional can't do X" again, which the mechanism table already covers.

Metrics per question, both modes:
  - faithfulness: are the answer's claims supported by the retrieved
    context? (via eval/llm_judge_eval.py)
  - answer relevancy: does the answer address the question asked?
  - correctness: does the answer contain the expected key terms
    (single_hop/multi_hop/adversarial only)?
  - for out_of_scope specifically: did it correctly decline, or
    hallucinate an answer?
  - LLM call count (from trace length) and wall-clock latency

Run with:
    uv run python eval/compare_modes.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import build_graph, build_llm, build_traditional_graph  # noqa: E402
from llm_judge_eval import build_context_string, score_answer_relevancy, score_faithfulness  # noqa: E402
from retrieval import HybridRetriever  # noqa: E402
from test_cases import TEST_CASES  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "compare_modes_results.json"

COMPARABLE_CATEGORIES = {"single_hop", "multi_hop", "adversarial", "out_of_scope"}

DECLINE_MARKERS = [
    "don't", "do not", "cannot", "no information", "not contain",
    "outside", "not covered", "not available", "unable",
]


def _agentic_state(question: str) -> dict:
    return {
        "original_query": question,
        "query": question,
        "chat_history": [],
        "tools_selected": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "grade": "",
        "grade_reasoning": "",
        "retries": 0,
        "specs_result": {},
        "web_results": [],
        "trace": [],
        "answer": "",
        "citations": [],
    }


def _traditional_state(question: str) -> dict:
    return {
        "query": question,
        "retrieved_docs": [],
        "reranked_docs": [],
        "trace": [],
        "answer": "",
        "citations": [],
    }


def _traditional_context_string(result: dict) -> str:
    citations = result.get("citations", [])
    if not citations:
        return "(no context -- direct answer)"
    return "\n\n".join(f"[{c['source']}] {c['snippet']}" for c in citations)


def run_comparison():
    retriever = HybridRetriever()
    llm = build_llm()
    agentic_app = build_graph(llm=llm, retriever=retriever)
    traditional_app = build_traditional_graph(llm=llm, retriever=retriever)

    cases = [c for c in TEST_CASES if c["category"] in COMPARABLE_CATEGORIES]

    rows = []
    for case in cases:
        q = case["question"]

        t0 = time.time()
        trad_result = traditional_app.invoke(_traditional_state(q))
        trad_latency = time.time() - t0

        t0 = time.time()
        agentic_result = agentic_app.invoke(_agentic_state(q))
        agentic_latency = time.time() - t0

        for mode, result, latency in [
            ("traditional", trad_result, trad_latency),
            ("agentic", agentic_result, agentic_latency),
        ]:
            answer = result["answer"]
            answer_norm = answer.lower().replace(",", "")

            if mode == "traditional":
                context = _traditional_context_string(result)
            else:
                context = build_context_string(result)

            faith = score_faithfulness(q, answer, context, llm)
            rel = score_answer_relevancy(q, answer, llm)

            if case["category"] == "out_of_scope":
                declined = any(m in answer_norm for m in DECLINE_MARKERS)
                correctness = None  # not applicable; decline-rate is the metric
            else:
                terms = case.get("answer_must_contain", [])
                correctness = (not terms) or any(t.lower() in answer_norm for t in terms)
                declined = None

            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "mode": mode,
                    "faithfulness": faith["score"],
                    "relevancy": rel["score"],
                    "correctness": correctness,
                    "declined": declined,
                    "llm_calls": len(result["trace"]),
                    "latency_s": latency,
                }
            )

        print(f"[{case['id']:>4}] done ({case['category']})")

    return rows


def compute_aggregates(rows: list[dict]) -> dict:
    aggregates = {}
    for mode in ["traditional", "agentic"]:
        mode_rows = [r for r in rows if r["mode"] == mode]
        faith_vals = [r["faithfulness"] for r in mode_rows if r["faithfulness"] is not None]
        rel_vals = [r["relevancy"] for r in mode_rows]
        correct_vals = [r["correctness"] for r in mode_rows if r["correctness"] is not None]
        decline_vals = [r["declined"] for r in mode_rows if r["declined"] is not None]
        call_vals = [r["llm_calls"] for r in mode_rows]
        latency_vals = [r["latency_s"] for r in mode_rows]

        aggregates[mode] = {
            "mean_faithfulness": sum(faith_vals) / len(faith_vals) if faith_vals else None,
            "mean_relevancy": sum(rel_vals) / len(rel_vals) if rel_vals else None,
            "correctness_rate": sum(correct_vals) / len(correct_vals) if correct_vals else None,
            "correct_decline_rate": sum(decline_vals) / len(decline_vals) if decline_vals else None,
            "mean_llm_calls": sum(call_vals) / len(call_vals) if call_vals else None,
            "mean_latency_s": sum(latency_vals) / len(latency_vals) if latency_vals else None,
            "n": len(mode_rows),
        }
    return aggregates


def save_results(rows: list[dict], aggregates: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_case": rows,
        "aggregates": aggregates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


def print_report(rows: list[dict], aggregates: dict):
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD: TRADITIONAL vs AGENTIC (same questions, same retriever/LLM)")
    print("=" * 78)

    header = f"{'id':>4} {'category':<14} {'mode':<12} {'faith':>7} {'relev':>6} {'correct':>8} {'declined':>9} {'calls':>6} {'lat(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        faith_str = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "  N/A"
        correct_str = "-" if r["correctness"] is None else ("YES" if r["correctness"] else "no")
        declined_str = "-" if r["declined"] is None else ("YES" if r["declined"] else "no")
        print(
            f"{r['id']:>4} {r['category']:<14} {r['mode']:<12} "
            f"{faith_str:>7} {r['relevancy']:>6.2f} {correct_str:>8} {declined_str:>9} "
            f"{r['llm_calls']:>6} {r['latency_s']:>7.1f}"
        )

    print("\n" + "-" * 78)
    print("AGGREGATES BY MODE")
    print("-" * 78)
    for mode, agg in aggregates.items():
        print(f"\n{mode.upper()}:")
        if agg["mean_faithfulness"] is not None:
            print(f"  Mean faithfulness:      {agg['mean_faithfulness']:.3f}")
        if agg["mean_relevancy"] is not None:
            print(f"  Mean answer relevancy:  {agg['mean_relevancy']:.3f}")
        if agg["correctness_rate"] is not None:
            print(f"  Correctness rate:       {agg['correctness_rate']:.1%}")
        if agg["correct_decline_rate"] is not None:
            print(f"  Correct-decline rate:   {agg['correct_decline_rate']:.1%} [out_of_scope only]")
        print(f"  Mean LLM calls/query:   {agg['mean_llm_calls']:.1f}")
        print(f"  Mean latency/query:     {agg['mean_latency_s']:.1f}s")

    print("\n" + "=" * 78)
    print(
        "NOTE: faithfulness is N/A for answers with no factual claims to "
        "check (e.g. a correct decline on an out_of_scope question -- "
        "'I don't know' has nothing to fact-check against context)."
    )


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. This script needs real LLM calls to produce real numbers.")
        sys.exit(1)
    rows = run_comparison()
    aggregates = compute_aggregates(rows)
    print_report(rows, aggregates)
    save_results(rows, aggregates)
