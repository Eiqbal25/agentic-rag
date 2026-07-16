"""Analytics page: live session analytics plus pre-computed offline
evaluation metrics."""

import json
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


def render_dashboard_tab():
    """
    Session-level analytics, trimmed to what matters for a live demo:
    total queries, average latency, and tool usage -- not a full log or
    citation breakdown (that lives in each answer's own citation
    expander already, no need to duplicate it here).
    """
    history = st.session_state.get("history", [])
    if not history:
        st.info("No queries yet this session — ask something in the Chat tab first.")
        return

    total_queries = len(history)
    latencies = [t.get("elapsed", 0) for t in history if t.get("elapsed")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    tool_counts = {"docs": 0, "specs": 0, "web": 0}
    for t in history:
        for tool in t.get("tools_selected", []):
            if tool in tool_counts:
                tool_counts[tool] += 1

    col1, col2 = st.columns(2)
    col1.metric("Total queries this session", total_queries)
    col2.metric("Avg. latency", f"{avg_latency:.1f}s")

    st.subheader("Tool usage")
    if sum(tool_counts.values()) > 0:
        st.bar_chart(tool_counts)
    else:
        st.caption("No tool calls yet (all queries answered directly).")

    # Per-model breakdown -- useful when comparing models mid-session
    # (e.g. switching between qwen3.6-27b and gpt-oss-20b to compare
    # speed): without tracking which model answered each query, an
    # average latency across a mixed-model session would be meaningless.
    models_used = {t.get("model_name") for t in history if t.get("model_name")}
    if len(models_used) > 1:
        st.subheader("By model")
        st.caption("Multiple models used this session — broken down separately since averaging across different models would be misleading.")
        rows = []
        for m in sorted(models_used):
            m_turns = [t for t in history if t.get("model_name") == m]
            m_latencies = [t["elapsed"] for t in m_turns if t.get("elapsed")]
            m_avg = sum(m_latencies) / len(m_latencies) if m_latencies else 0
            rows.append((m, len(m_turns), m_avg))
        st.dataframe(
            [{"Model": m, "Queries": n, "Avg. latency": f"{avg:.1f}s"} for m, n, avg in rows],
            hide_index=True,
        )


def _load_json_results(filename: str) -> dict | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def render_quality_metrics_tab():
    """
    Static, pre-computed quality metrics read directly from JSON files
    written by eval/run_eval.py and eval/compare_modes.py -- never
    computed live during a chat turn (that would add 2+ extra LLM calls
    per query just for display, directly working against the speed
    priority), and never manually typed into the UI (the file is the
    single source of truth; re-running eval updates what's shown here
    automatically, with no copy-paste step for anyone to get wrong or
    forget).
    """
    eval_results = _load_json_results("eval_results.json")
    compare_results = _load_json_results("compare_modes_results.json")

    if not eval_results and not compare_results:
        st.info(
            "No evaluation results yet. Run these to generate them:\n\n"
            "```\nuv run python eval/run_eval.py --ragas\n"
            "uv run python eval/compare_modes.py\n```"
        )
        return

    if eval_results:
        st.subheader("📚 Retrieval & generation quality")
        st.caption(f"Last run: {eval_results.get('generated_at', 'unknown')}")

        retrieval = eval_results.get("retrieval", {})
        if retrieval.get("mean_recall") is not None:
            cols = st.columns(3)
            cols[0].metric("Recall@k", f"{retrieval['mean_recall']:.0%}")
            cols[1].metric("MRR", f"{retrieval['mrr']:.2f}")
            cols[2].metric("Precision@k", f"{retrieval['mean_precision']:.0%}")

        routing = eval_results.get("routing")
        if routing:
            st.metric("Tool routing accuracy", f"{routing['accuracy']:.0%} ({routing['passed']}/{routing['total']})")

        ragas = eval_results.get("ragas")
        if ragas:
            cols = st.columns(2)
            if ragas.get("mean_faithfulness") is not None:
                cols[0].metric("Faithfulness", f"{ragas['mean_faithfulness']:.2f}")
            if ragas.get("mean_relevancy") is not None:
                cols[1].metric("Answer relevancy", f"{ragas['mean_relevancy']:.2f}")

    if compare_results:
        st.subheader("⚖️ Traditional vs. Agentic (head-to-head)")
        st.caption(f"Last run: {compare_results.get('generated_at', 'unknown')}")

        agg = compare_results.get("aggregates", {})
        trad = agg.get("traditional", {})
        agentic = agg.get("agentic", {})

        rows = [
            ("Faithfulness", "mean_faithfulness", "{:.2f}"),
            ("Answer relevancy", "mean_relevancy", "{:.2f}"),
            ("Correctness rate", "correctness_rate", "{:.0%}"),
            ("Correct-decline rate (out-of-scope)", "correct_decline_rate", "{:.0%}"),
            ("Avg. LLM calls/query", "mean_llm_calls", "{:.1f}"),
            ("Avg. latency/query", "mean_latency_s", "{:.1f}s"),
        ]
        table_rows = []
        for label, key, fmt in rows:
            t_val = trad.get(key)
            a_val = agentic.get(key)
            t_str = fmt.format(t_val) if t_val is not None else "—"
            a_str = fmt.format(a_val) if a_val is not None else "—"
            table_rows.append({"Metric": label, "Traditional": t_str, "Agentic": a_str})
        st.dataframe(table_rows, hide_index=True)
