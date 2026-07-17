"""
Evaluation harness for the agentic RAG system (single agent + tools).

Three layers of evaluation, matching the assignment's "explain test case
build to assure quality" requirement:

1. Retrieval-only metrics (no LLM calls needed): precision@k, recall@k,
   and MRR of the hybrid retriever against hand-labeled expected sources
   for docs-tool cases. Isolates retrieval quality from generation/routing.

2. Tool-routing accuracy (requires GROQ_API_KEY): does the router select
   the correct tool(s) -- docs / specs / web / none -- for each case,
   including multi-tool cases that need two sources at once.

3. End-to-end + memory (requires GROQ_API_KEY): whether the final answer
   contains expected terms, whether out-of-scope questions are declined
   rather than hallucinated, and whether a follow-up question correctly
   uses chat_history to resolve an ambiguous reference ("that").

Results are written to eval/results/eval_results.json alongside the
console output -- the app's Quality Metrics panel reads this file
directly rather than anyone copy-pasting numbers from a terminal into the
UI by hand. Re-running this script automatically updates what the app
displays next time it loads; there is no manually-maintained number
anywhere in this pipeline.

Run with:
    uv run python eval/run_eval.py                 # retrieval-only
    uv run python eval/run_eval.py --full           # + routing + generation + memory
    uv run python eval/run_eval.py --ragas          # + faithfulness + relevancy (implies --full)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import HybridRetriever  # noqa: E402
from test_cases import MEMORY_TEST_CASE, TEST_CASES  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "eval_results.json"


def evaluate_retrieval(retriever: HybridRetriever, k: int = 5) -> dict:
    print("=" * 70)
    print("RETRIEVAL EVALUATION (precision@k, recall@k, MRR) -- docs tool only")
    print("=" * 70)

    results = []
    for case in TEST_CASES:
        if not case["expected_sources"]:
            continue
        retrieved = retriever.retrieve(case["question"], k=k)
        retrieved_sources = [d.metadata["source"] for d, _ in retrieved]
        expected = set(case["expected_sources"])

        hits = [s for s in retrieved_sources if s in expected]
        precision = len(set(hits)) / len(set(retrieved_sources)) if retrieved_sources else 0
        recall = len(set(hits)) / len(expected) if expected else 0

        rr = 0.0
        for rank, s in enumerate(retrieved_sources, start=1):
            if s in expected:
                rr = 1.0 / rank
                break

        results.append(
            {"id": case["id"], "category": case["category"], "precision": precision, "recall": recall, "rr": rr}
        )
        status = "✓" if recall > 0 else "✗"
        print(
            f"{status} [{case['id']:>4}] {case['category']:<12} "
            f"P@{k}={precision:.2f} R@{k}={recall:.2f} RR={rr:.2f}  "
            f"{case['question'][:55]}"
        )

    summary = {"k": k, "per_case": results}
    if results:
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_r = sum(r["recall"] for r in results) / len(results)
        avg_mrr = sum(r["rr"] for r in results) / len(results)
        print("-" * 70)
        print(f"Mean Precision@{k}: {avg_p:.3f}")
        print(f"Mean Recall@{k}:    {avg_r:.3f}")
        print(f"MRR:                {avg_mrr:.3f}")

        by_cat: dict[str, list] = {}
        for r in results:
            by_cat.setdefault(r["category"], []).append(r)
        by_category = {}
        print("\nBy category:")
        for cat, rs in by_cat.items():
            cat_recall = sum(r["recall"] for r in rs) / len(rs)
            by_category[cat] = {"recall": cat_recall, "n": len(rs)}
            print(f"  {cat:<12} recall={cat_recall:.3f}  (n={len(rs)})")

        summary.update(
            {
                "mean_precision": avg_p,
                "mean_recall": avg_r,
                "mrr": avg_mrr,
                "by_category": by_category,
            }
        )
    return summary


def _fresh_state(question: str, chat_history: list | None = None) -> dict:
    return {
        "original_query": question,
        "query": question,
        "chat_history": chat_history or [],
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


def evaluate_routing(app) -> dict:
    print("\n" + "=" * 70)
    print("TOOL ROUTING EVALUATION")
    print("=" * 70)

    passed = 0
    total = 0
    per_case = []
    for case in TEST_CASES:
        if case["category"] == "web_required" and not os.environ.get("TAVILY_API_KEY"):
            print(f"~ [{case['id']:>4}] skipped (no TAVILY_API_KEY, checking routing decision only is still valid)")
        total += 1
        result = app.invoke(_fresh_state(case["question"]))
        actual = set(result.get("tools_selected") or [])
        expected = set(case["expected_tools"])
        ok = actual == expected
        passed += ok
        per_case.append(
            {"id": case["id"], "category": case["category"], "expected": sorted(expected), "actual": sorted(actual), "ok": ok}
        )
        status = "✓" if ok else "✗"
        print(
            f"{status} [{case['id']:>4}] {case['category']:<14} "
            f"expected={sorted(expected) or ['none']} actual={sorted(actual) or ['none']}"
        )
    print("-" * 70)
    print(f"Routing accuracy: {passed}/{total}")
    return {"passed": passed, "total": total, "accuracy": passed / total if total else 0, "per_case": per_case}


def evaluate_end_to_end(app) -> dict:
    print("\n" + "=" * 70)
    print("END-TO-END GENERATION EVALUATION")
    print("=" * 70)

    passed = 0
    per_case = []
    for case in TEST_CASES:
        if case["category"] == "web_required" and not os.environ.get("TAVILY_API_KEY"):
            print(f"~ [{case['id']:>4}] skipped (no TAVILY_API_KEY)")
            continue

        result = app.invoke(_fresh_state(case["question"]))
        answer_normalized = result["answer"].lower().replace(",", "")

        if case["category"] == "no_retrieval":
            ok = not result["citations"]
            note = "skipped tools correctly" if ok else "FAILED: used a tool when it shouldn't have"
        elif case["category"] == "out_of_scope":
            decline_markers = [
                "don't", "do not", "cannot", "no information", "not contain",
                "outside", "not covered", "not available", "unable",
            ]
            ok = any(m in answer_normalized for m in decline_markers)
            note = "declined correctly" if ok else "FAILED: may have hallucinated an answer"
        elif case["category"] == "web_required":
            ok = bool(result["citations"])  # can't hardcode live-data content
            note = "produced a web-grounded answer" if ok else "FAILED: no citations"
        else:
            terms = case["answer_must_contain"]
            ok = not terms or any(t.lower() in answer_normalized for t in terms)
            note = "contains expected terms" if ok else f"FAILED: missing any of {terms}"

        passed += ok
        per_case.append({"id": case["id"], "category": case["category"], "ok": ok, "note": note})
        status = "✓" if ok else "✗"
        print(f"{status} [{case['id']:>4}] {case['category']:<14} {note}")

    print("-" * 70)
    evaluated = len(
        [c for c in TEST_CASES if not (c["category"] == "web_required" and not os.environ.get("TAVILY_API_KEY"))]
    )
    print(f"Passed: {passed}/{evaluated}")
    return {"passed": passed, "total": evaluated, "accuracy": passed / evaluated if evaluated else 0, "per_case": per_case}


def evaluate_memory(app) -> dict:
    print("\n" + "=" * 70)
    print("MULTI-TURN MEMORY EVALUATION")
    print("=" * 70)

    case = MEMORY_TEST_CASE
    turn1 = app.invoke(_fresh_state(case["turn1_question"]))
    print(f"Turn 1: {case['turn1_question']!r}")
    print(f"  tools={turn1.get('tools_selected')}  answer[:100]={turn1['answer'][:100]!r}")

    history = [{"query": case["turn1_question"], "answer": turn1["answer"]}]
    turn2 = app.invoke(_fresh_state(case["turn2_question"], chat_history=history))
    print(f"Turn 2 (follow-up): {case['turn2_question']!r}")
    print(f"  tools={turn2.get('tools_selected')}  answer[:150]={turn2['answer'][:150]!r}")

    actual_tools = set(turn2.get("tools_selected") or [])
    expected_tools = set(case["turn2_expected_tools"])
    routing_ok = actual_tools == expected_tools

    answer_norm = turn2["answer"].lower()
    terms_ok = all(t.lower() in answer_norm for t in case["turn2_answer_must_contain"])

    print("-" * 70)
    print(f"{'✓' if routing_ok else '✗'} Turn 2 routing correct (used chat_history context)")
    print(f"{'✓' if terms_ok else '✗'} Turn 2 answer resolves the follow-up (mentions both models)")
    return {"routing_ok": routing_ok, "terms_ok": terms_ok}


def evaluate_ragas_style(app, llm) -> dict:
    """
    Faithfulness + answer relevancy scoring via eval/llm_judge_eval.py
    (see that module's docstring for why this reimplements RAGAS's
    metrics directly rather than depending on the `ragas` package).
    """
    from llm_judge_eval import build_context_string, score_answer_relevancy, score_faithfulness

    print("\n" + "=" * 70)
    print("RAGAS-STYLE EVALUATION (faithfulness + answer relevancy)")
    print("=" * 70)

    faithfulness_scores = []
    relevancy_scores = []
    per_case = []

    for case in TEST_CASES:
        if case["category"] == "web_required" and not os.environ.get("TAVILY_API_KEY"):
            continue
        if case["category"] == "no_retrieval":
            continue

        result = app.invoke(_fresh_state(case["question"]))
        context = build_context_string(result)

        faith = score_faithfulness(case["question"], result["answer"], context, llm)
        rel = score_answer_relevancy(case["question"], result["answer"], llm)

        if faith["score"] is not None:
            faithfulness_scores.append(faith["score"])
        relevancy_scores.append(rel["score"])

        per_case.append(
            {
                "id": case["id"],
                "category": case["category"],
                "faithfulness": faith["score"],
                "relevancy": rel["score"],
                "unsupported_claims": faith["unsupported_claims"],
            }
        )

        faith_str = f"{faith['score']:.2f}" if faith["score"] is not None else "N/A (no claims)"
        print(
            f"[{case['id']:>4}] {case['category']:<14} "
            f"faithfulness={faith_str:<14} relevancy={rel['score']:.2f}"
        )
        if faith["unsupported_claims"]:
            for c in faith["unsupported_claims"]:
                print(f"       ⚠ unsupported claim: {c}")

    print("-" * 70)
    mean_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    mean_rel = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
    if mean_faith is not None:
        print(f"Mean faithfulness: {mean_faith:.3f}  (n={len(faithfulness_scores)})")
    print(f"Mean answer relevancy: {mean_rel:.3f}  (n={len(relevancy_scores)})")
    return {"mean_faithfulness": mean_faith, "mean_relevancy": mean_rel, "per_case": per_case}


def save_results(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also run routing + generation + memory evals (needs GROQ_API_KEY)")
    parser.add_argument("--ragas", action="store_true", help="also run faithfulness + answer relevancy scoring (needs GROQ_API_KEY, implies --full)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    all_results: dict = {}

    retriever = HybridRetriever()
    all_results["retrieval"] = evaluate_retrieval(retriever, k=args.k)

    if args.full or args.ragas:
        if not os.environ.get("GROQ_API_KEY"):
            print("\n[!] --full/--ragas requested but GROQ_API_KEY is not set. Skipping.")
        else:
            from agent import build_graph, build_llm

            app = build_graph(retriever=retriever)
            all_results["routing"] = evaluate_routing(app)
            all_results["end_to_end"] = evaluate_end_to_end(app)
            all_results["memory"] = evaluate_memory(app)

            if args.ragas:
                judge_llm = build_llm()
                all_results["ragas"] = evaluate_ragas_style(app, judge_llm)

    save_results(all_results)
