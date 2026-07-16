"""
Agentic RAG pipeline (single agent + multiple tools), built as a LangGraph
state graph.

Graph shape:

    analyze_query (router: picks a SUBSET of {docs, specs, web}, using
                   chat_history for context)
         |
         | "docs" selected?
         |
    +----+-----+
    v           v
  retrieve   gather_other_tools
    |              ^
    v              |
  rerank            |
    |              |
    v              |
  grade_documents   |
    | relevant -----+
    | not relevant & retries left -> rewrite_query -> retrieve (loop)
    | retries exhausted ----------------------------> gather_other_tools
                   |
                   v
              generate (merges docs + specs + web context, cites all)
                   |
                   v
                  END

Design notes:
- The corrective doc-retrieval loop (retrieve -> rerank -> grade ->
  rewrite -> retry) is UNCHANGED from the original single-source build.
  It's the core "agentic" mechanism and didn't need to change when the
  system gained more tools.
- "specs" and "web" don't get a grade/retry loop of their own: a SQL
  query against a small structured DB is either right or returns nothing
  (nothing to "grade" the way fuzzy semantic retrieval needs grading),
  and Tavily already returns pre-ranked web results. Only unstructured
  semantic search (docs) has the "did I actually find the right thing"
  ambiguity that justifies a self-correction loop.
- chat_history (recent turns) is threaded into both the router prompt and
  the generation prompt, so follow-up questions ("compare that to...")
  have context to resolve against.
"""

from langgraph.graph import END, StateGraph

from llm_utils import invoke_with_retry, strip_thinking_tags
from retrieval import HybridRetriever, rerank_with_llm
from tools.specs_tool import query_specs_db
from tools.web_tool import web_search

from .llm_factory import _build_default_fast_llm, build_llm
from .parsing import parse_grade_response, parse_tool_selection
from .prompts import build_direct_prompt, build_generation_prompt
from .state import (
    MAX_RETRIES,
    RERANK_TOP_N,
    RETRIEVE_K,
    AgentState,
    TraditionalRAGState,
    _format_history,
)


def build_graph(
    llm=None,
    fast_llm=None,
    retriever: HybridRetriever | None = None,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
    groq_api_key: str | None = None,
    tavily_api_key: str | None = None,
):
    """
    checkpointer / interrupt_before: optional, additive, backward-
    compatible -- both default to None, which preserves the exact
    original behavior (full run to completion, no pausing). These exist
    specifically so app.py can pass interrupt_before=["generate"] with an
    in-memory checkpointer to get the graph to stop right before the
    final generation call, enabling real token-by-token streaming of the
    answer (see app.py) without duplicating this graph's routing/
    retrieval/grading logic in a second code path. eval scripts and
    tests continue to call build_graph() with no checkpointer, running
    the graph to completion exactly as before.

    llm: the "strong" model, used only for the final user-facing answer
        (the generate node) -- this is the one output quality actually
        gets judged on.
    fast_llm: used for routing, reranking, grading, query rewriting, and
        SQL generation -- structured micro-decisions that shouldn't be
        doing open-ended chain-of-thought reasoning at all. If not
        explicitly provided, this is auto-constructed from the SAME
        model as `llm` (see _build_default_fast_llm) but with reasoning
        disabled/minimized via Groq's reasoning_effort parameter --
        reproduced live: qwen3.6-27b burned its ENTIRE token budget
        reasoning about a single SQL-generation call (an ambiguous GPU
        comparison question) and never emitted any SQL at all, because
        nothing told it these structured decisions don't need deep
        reasoning. Pass an explicit fast_llm to use a genuinely different
        (smaller/cheaper) model instead, once one is verified reliable --
        the obvious smaller Groq option (openai/gpt-oss-20b) shares the
        same Harmony-response-format parsing bug already hit live with
        gpt-oss-120b (see build_llm's docstring), so this project doesn't
        default to a different model, only a differently-configured one.
    groq_api_key / tavily_api_key: threaded through to build_llm/
        _build_default_fast_llm and the web_search tool call respectively,
        instead of those functions falling back to GROQ_API_KEY/
        TAVILY_API_KEY env vars. Matters when llm/fast_llm aren't already
        pre-built by the caller (e.g. eval scripts always pass a
        pre-built llm, so these are no-ops there) -- the Streamlit app
        passes a session-scoped key explicitly here specifically so one
        user's pasted key can't leak into another concurrent session's
        calls via shared process-global env var state (see
        llm_factory.build_llm's docstring).
    """
    llm = llm or build_llm(api_key=groq_api_key)
    fast_llm = fast_llm or _build_default_fast_llm(llm, api_key=groq_api_key)
    retriever = retriever or HybridRetriever()

    # ---- Nodes ----

    def analyze_query(state: AgentState) -> dict:
        query = state["original_query"]
        history_str = _format_history(state.get("chat_history", []))
        prompt = (
            "You are the routing step of an agent with THREE tools:\n"
            "  - docs: search a knowledge base of 12 documents about RAG, "
            "embeddings, vector databases, fine-tuning, quantization, "
            "inference optimization, and on-prem AI infrastructure.\n"
            "  - specs: query a structured database of real GPU and "
            "enterprise SSD hardware specs (VRAM, bandwidth, TFLOPS, "
            "endurance, etc.)\n"
            "  - web: live internet search, for anything current or "
            "outside the other two sources.\n\n"
            f"Recent conversation:\n{history_str}\n\n"
            f'Current user message: "{query}"\n\n'
            "Decide which tool(s) are needed to answer this message. "
            "Consider the conversation history if the message is a "
            "follow-up.\n\n"
            "IMPORTANT: select 'none' ONLY for conversational/meta "
            "messages that have no factual content to verify -- "
            "greetings, thanks, small talk, or questions about the "
            "assistant itself (e.g. 'hi', 'what can you do'). For ANY "
            "request asking for facts, instructions, recipes, "
            "explanations, or information about a real-world topic -- "
            "even if you already 'know' the answer, and even if it's "
            "outside docs/specs -- select 'web' rather than 'none'. "
            "This system's entire purpose is to ground answers in "
            "retrieved evidence rather than trust unverified model "
            "knowledge, so never pick 'none' just because you're "
            "confident you can answer from memory.\n\n"
            "IMPORTANT: select the MINIMAL set of tools that actually "
            "answers the question -- do not add a tool 'just in case' if "
            "it isn't needed. Most questions need exactly ONE tool. Only "
            "select more than one if the question genuinely requires "
            "combining information from multiple sources (e.g. comparing "
            "a document's explanation to a specific hardware spec).\n\n"
            "You may briefly explain your reasoning if you want. Whether "
            "you explain or not, your response MUST end with a JSON "
            "object on its own final line, in exactly this format "
            '(nothing after it): {"tools": [...]}\n'
            'Examples of valid final lines:\n'
            '{"tools": ["docs"]}\n'
            '{"tools": ["docs", "specs"]}\n'
            '{"tools": []}   <- use this for "none" (conversational input)'
        )
        resp = invoke_with_retry(fast_llm, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        tools = parse_tool_selection(text)
        return {
            "tools_selected": tools,
            "query": query,
            "retries": 0,
            "trace": [f"analyze_query -> tools={tools or ['none']}"],
        }

    def retrieve(state: AgentState) -> dict:
        docs = retriever.retrieve(state["query"], k=RETRIEVE_K)
        return {
            "retrieved_docs": docs,
            "trace": [
                f"retrieve(query={state['query']!r}) -> "
                f"{[d.metadata['source'] for d, _ in docs]}"
            ],
        }

    def rerank(state: AgentState) -> dict:
        docs = [d for d, _ in state["retrieved_docs"]]
        reranked = rerank_with_llm(
            fast_llm, state["original_query"], docs, top_n=RERANK_TOP_N
        )
        return {
            "reranked_docs": reranked,
            "trace": [
                "rerank -> "
                + str([(d.metadata["source"], s) for d, s in reranked])
            ],
        }

    def grade_documents(state: AgentState) -> dict:
        context = "\n\n---\n\n".join(
            f"[{d.metadata['source']}] {d.page_content[:400]}"
            for d, _ in state["reranked_docs"]
        )
        prompt = (
            "You are a strict relevance grader. Given the user's question "
            "and a set of retrieved passages, decide whether the passages "
            "collectively contain enough information to answer the "
            "question accurately.\n\n"
            f"Question: {state['original_query']}\n\n"
            f"Passages:\n{context}\n\n"
            "Respond in exactly this format:\n"
            "GRADE: RELEVANT or IRRELEVANT\n"
            "REASON: <one sentence>"
        )
        resp = invoke_with_retry(fast_llm, prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        grade, reason = parse_grade_response(text)
        return {
            "grade": grade,
            "grade_reasoning": reason,
            "trace": [f"grade_documents -> {grade} ({reason})"],
        }

    def rewrite_query(state: AgentState) -> dict:
        prompt = (
            "The following search query did not retrieve documents "
            "sufficient to answer the user's question. Rewrite it as a "
            "clearer, more specific search query using terminology likely "
            "to appear in technical documentation (avoid vague pronouns, "
            "add explicit technical terms). Respond with ONLY the "
            "rewritten query, nothing else.\n\n"
            f"Original question: {state['original_query']}\n"
            f"Previous search query: {state['query']}\n"
            f"Why it failed: {state.get('grade_reasoning', '')}"
        )
        resp = invoke_with_retry(fast_llm, prompt)
        new_query = strip_thinking_tags(
            resp.content if hasattr(resp, "content") else str(resp)
        )
        return {
            "query": new_query,
            "retries": state["retries"] + 1,
            "trace": [f"rewrite_query -> {new_query!r}"],
        }

    def gather_other_tools(state: AgentState) -> dict:
        tools = state.get("tools_selected", [])
        updates: dict = {}
        trace: list[str] = []

        if "specs" in tools:
            result = query_specs_db(state["original_query"], fast_llm)
            updates["specs_result"] = result
            trace.append(
                f"query_specs_db(sql={result['sql']!r}) -> "
                f"{len(result['rows'])} rows"
                + (f" (error: {result['error']})" if result["error"] else "")
            )

        if "web" in tools:
            try:
                results = web_search(state["original_query"], api_key=tavily_api_key)
                updates["web_results"] = results
                trace.append(f"web_search -> {len(results)} results")
            except RuntimeError as e:
                updates["web_results"] = []
                trace.append(f"web_search -> skipped ({e})")

        updates["trace"] = trace or ["gather_other_tools -> nothing to gather"]
        return updates

    def generate(state: AgentState) -> dict:
        tools = state.get("tools_selected", [])

        if not tools:
            direct_prompt = build_direct_prompt(state)
            resp = invoke_with_retry(llm, direct_prompt)
            answer = strip_thinking_tags(
                resp.content if hasattr(resp, "content") else str(resp)
            )
            return {
                "answer": answer,
                "citations": [],
                "trace": ["generate -> direct answer (no tools used)"],
            }

        prompt, citations = build_generation_prompt(state)
        resp = invoke_with_retry(llm, prompt)
        answer = strip_thinking_tags(
            resp.content if hasattr(resp, "content") else str(resp)
        )
        return {
            "answer": answer,
            "citations": citations,
            "trace": [f"generate -> answer with {len(citations)} citations"],
        }

    # ---- Conditional edges ----

    def route_after_analyze(state: AgentState) -> str:
        return "retrieve" if "docs" in state.get("tools_selected", []) else "gather_other_tools"

    def route_after_grade(state: AgentState) -> str:
        if state["grade"] == "RELEVANT":
            return "gather_other_tools"
        if state["retries"] >= MAX_RETRIES:
            return "gather_other_tools"
        return "rewrite_query"

    # ---- Build graph ----

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("gather_other_tools", gather_other_tools)
    graph.add_node("generate", generate)

    graph.set_entry_point("analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {"retrieve": "retrieve", "gather_other_tools": "gather_other_tools"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"gather_other_tools": "gather_other_tools", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("gather_other_tools", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def run_agentic_rag(
    query: str,
    llm=None,
    retriever: HybridRetriever | None = None,
    chat_history: list | None = None,
):
    app = build_graph(llm=llm, retriever=retriever)
    initial_state: AgentState = {
        "original_query": query,
        "query": query,
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
    return app.invoke(initial_state)


# ============================================================================
# Traditional RAG (for side-by-side comparison in the demo)
# ============================================================================
#
# A deliberately minimal, textbook single-pass pipeline: embed the query,
# retrieve top-k, stuff it into the prompt, generate once, return whatever
# comes out -- no routing decision, no self-grading, no retry, no memory.
#
# DESIGN CHOICE, worth being explicit about: this uses the EXACT SAME
# hybrid retrieval (dense+BM25 RRF) and LLM reranking as the agentic graph,
# not a weaker/simpler retriever. That's deliberate -- retrieval quality is
# a separate engineering axis from "agentic vs traditional," and conflating
# them would make any observed difference ambiguous (is agentic actually
# better, or did traditional just get worse retrieval?). Holding retrieval
# identical isolates the comparison to the one thing that's actually being
# tested: does having a decision-making layer (route / grade / retry /
# remember) on top of the same retrieval change the outcome?
#
# The generation prompt is also deliberately simpler than the agentic
# version's: it does NOT include the "say so if evidence is insufficient"
# instruction. This matches how naive RAG is actually implemented in most
# real-world tutorials/quick-starts -- the point of this mode is to show
# what happens when nothing catches a bad retrieval, not to quietly
# smuggle in the same safety behavior under a different name.


def build_traditional_graph(llm=None, retriever: HybridRetriever | None = None):
    llm = llm or build_llm()
    retriever = retriever or HybridRetriever()

    def retrieve(state: TraditionalRAGState) -> dict:
        docs = retriever.retrieve(state["query"], k=RETRIEVE_K)
        return {
            "retrieved_docs": docs,
            "trace": [
                f"retrieve(query={state['query']!r}) -> "
                f"{[d.metadata['source'] for d, _ in docs]}"
            ],
        }

    def rerank(state: TraditionalRAGState) -> dict:
        docs = [d for d, _ in state["retrieved_docs"]]
        reranked = rerank_with_llm(llm, state["query"], docs, top_n=RERANK_TOP_N)
        return {
            "reranked_docs": reranked,
            "trace": ["rerank -> " + str([(d.metadata["source"], s) for d, s in reranked])],
        }

    def generate(state: TraditionalRAGState) -> dict:
        docs = state.get("reranked_docs") or state.get("retrieved_docs") or []
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata['source']}]\n{d.page_content}" for d, _ in docs
        )
        # Deliberately naive prompt -- no instruction to decline on weak
        # evidence, no self-check. This is the point of this mode.
        prompt = (
            "Using the following context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\n\n"
            "Answer:"
        )
        resp = invoke_with_retry(llm, prompt)
        answer = strip_thinking_tags(
            resp.content if hasattr(resp, "content") else str(resp)
        )
        citations = [
            {
                "type": "document",
                "source": d.metadata["source"],
                "section": d.metadata.get("section", ""),
                "snippet": d.page_content[:100],
            }
            for d, _ in docs
        ]
        return {
            "answer": answer,
            "citations": citations,
            "trace": ["generate -> single-pass answer, no grading, no retry"],
        }

    graph = StateGraph(TraditionalRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_traditional_rag(query: str, llm=None, retriever: HybridRetriever | None = None):
    app = build_traditional_graph(llm=llm, retriever=retriever)
    initial_state: TraditionalRAGState = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "trace": [],
        "answer": "",
        "citations": [],
    }
    return app.invoke(initial_state)
