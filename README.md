# Agentic RAG — Single Agent, Multiple Tools

An individual project demoing **agentic RAG**: a single LangGraph-orchestrated
agent that decides, per query, which of three tools it needs — a document
knowledge base, a structured hardware specs database, or live web search —
uses conversation memory to resolve follow-up questions, and self-corrects
its own document retrieval when the first pass comes up short.

This follows the "AI Agent" pattern (single orchestrator + memory + planning
+ tools), not the multi-agent pattern (independent specialized agents behind
a separate aggregator) — see [Architecture pattern chosen](#architecture-pattern-chosen-and-why)
for why, and what would justify moving to multi-agent later.

## Architecture

```
User query + chat_history
        │
        ▼
┌───────────────────┐
│   analyze_query     │  Router: picks a SUBSET of {docs, specs, web}
│  (uses memory to     │  based on the query + recent conversation.
│   resolve follow-ups)│  Empty set -> direct answer, no tools.
└─────────┬──────────┘
          │
   "docs" selected? ──────────────────────────────┐
          │ yes                                    │ no
          ▼                                        │
┌───────────────────┐                              │
│     retrieve        │◄─────────────┐              │
│ (hybrid RRF:         │              │              │
│  dense + BM25)       │              │              │
└─────────┬──────────┘              │              │
          ▼                          │              │
┌───────────────────┐               │              │
│      rerank          │               │              │
│ (LLM cross-encoder)  │               │              │
└─────────┬──────────┘               │              │
          ▼                          │              │
┌───────────────────┐   IRRELEVANT  │              │
│  grade_documents      ├───────────────┘ retries<2   │
└─────────┬──────────┘                              │
          │ RELEVANT (or retries exhausted)          │
          ▼                                          │
┌────────────────────────────────────────────────────┘
│   gather_other_tools
│   - "specs" selected? -> text-to-SQL against specs.db
│   - "web" selected?    -> Tavily live search
└─────────┬──────────┘
          ▼
┌───────────────────┐
│      generate        │  Merges whichever sources were actually used,
│  (cited answer)      │  cites every claim, declines if evidence is thin
└─────────┬──────────┘
          ▼
       Answer + citations + reasoning trace
```

**The corrective loop** (retrieve → rerank → grade → rewrite → retry, bounded
to 2 retries) is the original agentic mechanism from the single-source build
— unchanged. It only applies to the `docs` tool: a SQL query against the
specs DB is either right or returns nothing (no fuzzy relevance to grade),
and Tavily already returns pre-ranked web results — only unstructured
semantic search has the "did I actually find the right thing" ambiguity
that justifies self-correction.

**Memory**: recent `(question, answer)` turns are threaded into both the
router prompt and the generation prompt, so "how does that compare to..."
resolves against the previous turn instead of failing silently.

## The three tools

| Tool | What it is | Backing data |
|---|---|---|
| **docs** | Hybrid (dense+BM25, RRF-fused, LLM-reranked) search over a document corpus | 12 self-authored documents on RAG, embeddings, vector DBs, fine-tuning, quantization, inference optimization, on-prem AI infra (~5,800 words, 84 chunks) |
| **specs** | Text-to-SQL agent over a structured database | `data/specs.db` — real GPU (A100/H100/H200/RTX 4090) and enterprise SSD (Samsung/Solidigm/Micron) specs, sourced from manufacturer datasheets (see `data/build_specs_db.py` for citations per row). Stands in for a live "Cloud APIs" / inventory system. |
| **web** | Live internet search via Tavily | Anything current or outside the other two sources |

### Why real, sourced data for the specs DB
Every row in `specs.db` is a real published number from an NVIDIA/Samsung/
Solidigm/Micron datasheet or product page (URLs are stored per-row and
surfaced in citations) — not invented. Facts/specs aren't copyrightable
expression, so citing manufacturer-published numbers here is standard
practice, unlike reproducing article prose.

### Text-to-SQL safety guardrails (`src/tools/specs_tool.py`)
An LLM writing SQL is a real risk surface if unguarded, so:
1. The DB connection opens **read-only** (`mode=ro`) at the SQLite driver
   level — a successful injection still can't write/drop anything.
2. Only `SELECT` statements are accepted; `INSERT/UPDATE/DELETE/DROP/ALTER/
   ATTACH/PRAGMA` are rejected by keyword check before execution.
3. Multi-statement queries (`; DROP TABLE ...`) are rejected.
4. An unbounded query gets a `LIMIT 20` appended automatically.

## Why MCP is an optional layer, not the primary interface

The Streamlit app calls all three tools as **direct Python function calls**
— the correct choice for a single-consumer app, since MCP's serialization/
IPC overhead buys nothing when there's only one caller. `src/mcp_server.py`
additionally exposes the same three tools over MCP, which is only justified
because it creates a **second, genuinely different consumer** (e.g. Claude
Desktop, MCP Inspector, or any other MCP-compatible client) reusing the
exact same retrieval index / specs DB / web search — without duplicating
any of this project's code. Run it standalone:
```bash
uv run python src/mcp_server.py
```
and point any MCP client (stdio transport) at that script.

## Architecture pattern chosen, and why

There are two common agentic RAG shapes: **single agent + tools** (one
orchestrator, memory, planning, multiple tools) and **multi-agent** (several
independent specialized agents, each behind their own tools/MCP servers,
coordinated by a separate aggregator agent). This project uses the former.

2026 production research on this tradeoff is fairly consistent: a review of
47 production deployments found **68% could have matched or beaten their
results with a single well-built agent, at ~3x lower cost**; multi-agent
setups have been measured using up to **15x more tokens** and taking **~3x
longer** for the same task, with debugging time increasing ~3.7x due to
distributed failure surfaces across agent handoffs. Multi-agent decomposition
earns its cost when subtasks are genuinely *parallelizable* or the domain
spans **5+ heterogeneous functions** — this system has 3 tools a single
router holds comfortably, and doc/specs/web don't need to run in parallel
against each other. Multi-agent would be the right call to revisit if the
tool count grew substantially or specs/docs/web needed independent,
simultaneous multi-step reasoning rather than a single routing decision.

## Setup

```bash
uv sync

cp .env.example .env
# then fill in:
#   GROQ_API_KEY       - free at https://console.groq.com
#   TAVILY_API_KEY     - free at https://tavily.com (optional; web tool is
#                        skipped gracefully if omitted)
#   SSL_CERT_FILE /
#   REQUESTS_CA_BUNDLE - Windows + AVG Antivirus only, see the comment
#                        in .env.example. Leave blank otherwise.
# .env is filled in BEFORE the next step specifically because it can
# need those SSL vars to download the embedding model weights.

# Build the document index (run once, or after editing data/corpus/*).
# First run downloads Qwen3-Embedding-0.6B (~1.2GB, one-time, then
# cached under ~/.cache/huggingface/hub) -- takes a minute or two;
# every run after is fast (loads from local cache).
uv run python src/retrieval/ingest.py

# Build the specs database (run once, or after editing data/build_specs_db.py)
uv run python data/build_specs_db.py

uv run streamlit run app.py
```

(`pip install -r requirements.txt` also works if you'd rather not use `uv`
— both are kept in sync. See the comment above `torch` in
`requirements.txt` for the smaller CPU-only wheel flag.)

**Security note on API keys**: the sidebar never pre-fills a text input
with a real secret loaded from `.env` -- a key already present in the
environment shows as a status confirmation ("✅ loaded from
environment"), not a fillable field. This matters because
`type="password"` on a Streamlit `text_input` only masks what's
*displayed*; the actual value still sits in the widget's underlying data
as plain text, extractable via browser dev tools if it were ever
pre-populated with a real secret -- a real credential-leak path once
this app is deployed publicly (see Security & Deployment below), not
just a local concern. Manual override fields are always empty by default.

Navigation is a sidebar radio (not top tabs) so it stays reachable
without scrolling back up a long chat — switching pages used to require
scrolling the top tab bar back into view first, since it lived inside
the scrolling main content area; the sidebar stays fixed regardless of
scroll position. One consequence: unlike tabs (which render every tab's
widgets on every rerun, just hiding the inactive ones), only the active
page's code runs each time, so the Models & Settings page's widgets use
`key=` to persist their values in `st.session_state` for the other
pages to read.

The app has five pages:
- **Chat** — the agent itself
- **Documents** — browser and editor for the document corpus the `docs`
  tool searches. View any document's exact raw file content (so a
  citation's snippet can be checked against the real source instead of
  trusted blindly), add a new `.md` document (upload or paste directly
  in-app), or delete one — each followed by an automatic full reindex
  (`ingest.build_indexes()`, chunk → embed → rebuild Chroma + BM25) so
  the agent's retrieval reflects the change immediately instead of
  silently searching a stale index. Delete is a **soft delete**: the
  file moves to `data/corpus/_deleted/` (restorable from a "Recently
  deleted" expander), not erased — this project has no git history to
  recover from a mistake, so nothing is truly gone from a single click.
  Deleting the last remaining document is blocked (embedding/indexing
  can't fit against an empty corpus). The specs DB stays read-only
  browsing only (GPU/SSD model lists) — it's built offline by
  `data/build_specs_db.py` and out of scope for this editor.
- **Models & Settings** — answer-model selection, the reasoning-trace
  toggle, and the tool legend, moved out of the sidebar into their own
  page so the sidebar stays reserved for navigation, the one hard
  prerequisite (API keys), and the one action needed regardless of
  which page is open (clearing conversation memory). Also has a
  **fast-model selector** for token optimization: routing, reranking,
  grading, query rewriting, and text-to-SQL all run an LLM call on
  *every* query, while final answer generation runs once — so pointing
  those calls at a genuinely smaller/cheaper model (instead of just the
  default same-model-with-reasoning-disabled tuning) cuts real token
  spend where it's actually being spent. Defaults to "Same as answer
  model" (the existing behavior); picking `openai/gpt-oss-20b`
  explicitly is offered but not defaulted, since it shares
  `gpt-oss-120b`'s known Harmony-format parsing reliability issue on
  Groq (see the model table below) — an informed opt-in, not a silent
  default.
- **Sources** — every unique source cited so far this session,
  deduplicated across turns and ranked by how often it's been reused —
  the multi-turn counterpart to each answer's own per-message citation
  expander, which only shows that one turn's sources.
- **Analytics** — live session analytics (query count, average latency,
  tool usage) plus pre-computed, offline evaluation numbers (retrieval
  recall/MRR, routing accuracy, faithfulness/relevancy, traditional-vs-
  agentic comparison), read directly from JSON files written by
  `eval/run_eval.py` and `eval/compare_modes.py`. **Never computed live
  during a chat turn** -- doing so would add 2+ extra LLM calls per query
  just for display, directly working against the speed optimizations
  described above. **Never manually typed into the UI** either -- the
  JSON file is the single source of truth, so re-running eval
  automatically updates what the app shows next time it loads, with no
  copy-paste step for anyone to get wrong or forget. If neither file
  exists yet, the tab shows the exact commands to generate them instead
  of a blank or fabricated number.

Citations are color-coded by source type (docs=blue, specs=purple,
web=green) so source authority is visually distinguishable -- a web
result (Tavily's ranking, sometimes lower-quality) no longer renders
with the same visual weight as a curated document or a sourced
spec-sheet entry.

## Security & Deployment

Audited before this project's first public push. Two real, confirmed-
exploitable bugs were found and fixed; one design limitation is
documented rather than code-fixed, since it's a hosting decision, not a
bug.

**Fixed: stored/reflected XSS in citation rendering.** Citation cards
(`ui/chat_page.py`, `ui/sources_page.py`) interpolate a source's title,
URL, section heading, and snippet directly into `st.markdown(...,
unsafe_allow_html=True)`. None of that data is trusted: web citations
carry a title/URL/content straight from live Tavily results (fully
attacker-controlled if a malicious page gets indexed and cited), and
document citations carry a section heading/snippet from corpus files
the Documents tab lets anyone add. Confirmed exploitable prior to the
fix: a crafted page title or document heading containing a `<script>`
tag executed in the viewer's browser the moment that source got cited.
Fixed by HTML-escaping every interpolated value (`ui/styles.py::
escape_html`) and restricting citation links to `http(s)` URLs only
(`ui/styles.py::safe_url`, which also blocks `javascript:`/`data:` URI
clicks that plain escaping alone wouldn't stop). Verified directly
against real payloads (a `<script>` tag, a `javascript:` URI), not just
inferred from reading the fix.

**Fixed: cross-session API key leakage.** The app used to do
`os.environ["GROQ_API_KEY"] = groq_key` when a user pasted a session-
only key override in the sidebar. `os.environ` is process-global state,
but a single Streamlit process serves many concurrent browser sessions
by default -- one user's pasted key would silently apply to every other
concurrent session's LLM/web-search calls. Made worse by `get_app()`'s
`@st.cache_resource` key being an underscore-prefixed `_api_key`
parameter, which Streamlit deliberately excludes from the cache's hash
-- meaning the cache was keyed only on `(model_name, fast_model_choice)`,
so whichever session's key happened to be active when a given combo was
*first* requested got baked into the cached graph/LLM client and reused
for every other session requesting that same combo, regardless of their
own key. Fixed by threading the actual key explicitly through
`build_llm`/`build_fast_llm`/`build_graph`/`web_search` instead of
reading env vars at call time, and by dropping the cache parameter's
underscore prefix so different keys correctly produce different cached
instances. Only relevant once this app is deployed somewhere serving
concurrent users from one process (e.g. Streamlit Community Cloud);
harmless for local single-user use, where there's only ever one session.

**Documented, not code-fixed: no authentication on document
mutation.** The Documents tab lets anyone viewing the app add, delete,
or restore corpus files with no login check -- that's the feature
working as designed for a personal/local demo. If this app is ever
deployed somewhere reachable by strangers, put it behind authentication
first (a reverse proxy with auth, Streamlit Community Cloud's built-in
viewer auth, etc.) -- without that, anyone can vandalize or poison the
shared knowledge base, and (combined with the XSS class of bug above,
even though the specific instance is now fixed) injecting a document is
a plausible way to get attacker content in front of other users. This is
a hosting/deployment decision, not something a code change can close on
its own.

**Already-solid, confirmed during this audit (no changes needed):**
- No hardcoded secrets anywhere in tracked files; `.env` is gitignored,
  `.env.example` has only blank placeholders, and the real key values
  don't appear anywhere else in the repo.
- No `eval`/`exec`/`os.system`/`subprocess` usage anywhere (no command
  injection surface).
- Text-to-SQL guardrails (`src/tools/specs_tool.py`) layer a read-only
  SQLite connection (the actual backstop -- a successful injection still
  can't write anything at the OS/driver level) with SELECT-only
  enforcement, a forbidden-keyword blocklist, multi-statement rejection,
  and an auto-appended `LIMIT`.
- Document upload filenames are sanitized via a strict allowlist regex
  (`ui/documents_page.py::_sanitize_md_filename`) after stripping any
  directory components, blocking path traversal.
- The API key sidebar never pre-fills a text input with a real secret
  (see the note below) -- unrelated to this audit, fixed in an earlier
  pass, reconfirmed still in place.

**Before making the repo public**, consider also: adding a `LICENSE`
file (a public GitHub repo with none defaults to "all rights reserved"
in most jurisdictions, which blocks the reuse open-sourcing usually
intends); enabling GitHub's secret scanning and Dependabot once pushed,
since this project pins exact dependency versions (`==`) for
reproducibility, which means security patches need a deliberate manual
bump rather than arriving automatically.

## Model selection & rationale

| Component | Model used | Why |
|---|---|---|
| Routing / grading / rewriting / generation / text-to-SQL LLM | `qwen/qwen3.6-27b` (via Groq) | `llama-3.3-70b-versatile` — the default in most Groq/LangChain tutorials — was **deprecated 2026-06-17, shutting down 2026-08-16**. Groq's own migration guidance points to `gpt-oss-120b` or `qwen/qwen3.6-27b`. This project initially defaulted to `gpt-oss-120b`, but switched after a live run reproduced a real, current `groq.BadRequestError: output_parse_failed` -- `gpt-oss-120b` uses OpenAI's "Harmony" response format internally, which intermittently fails to parse cleanly through Groq's API (confirmed via multiple open `langchain-ai/langchain` GitHub issues against this exact model/provider combination, not specific to this project's prompts). `qwen3.6-27b` isn't a Harmony-format model, so it doesn't hit this failure mode, and fits this pipeline's actual job (fast, reliable structured micro-decisions) better than a heavyweight reasoning model anyway. |
| Dense embeddings (docs tool) | `Qwen/Qwen3-Embedding-0.6B` via sentence-transformers, CPU | Originally TF-IDF+SVD (LSA), a self-fitted classical stand-in, under the assumption this environment couldn't reach huggingface.co. That assumption turned out to be wrong once actually tested (see "Windows/AVG SSL note" below) — huggingface.co and pypi.org are both reachable, so the real embedding model recommended from day one is what's actually running retrieval now. `Qwen3-Embedding-0.6B` was picked over the originally-suggested 4B/8B variants specifically to fit this machine's GPU (GTX 1650, 4GB VRAM) and because the tiny 12-doc/84-chunk corpus doesn't need a larger model's capacity; runs on CPU here since even CPU inference is fast at this corpus size. 1024-dim, asymmetric encoding (queries get an instruction-prefixed prompt via `prompt_name="query"`, documents don't — see `src/retrieval/embeddings.py::Qwen3Embeddings`). LSA is kept in `src/retrieval/embeddings.py` as `LSAEmbeddings`, unused by the live pipeline, as a zero-download fallback; its own tests are untouched. |
| Reranker (docs tool) | LLM-as-cross-encoder | Same "assumed no Hugging Face access" reasoning as the embeddings originally were — not yet revisited now that the assumption is known to be false. **Recommended swap**: `Qwen3-Reranker`, same reasoning as the embedding swap above. |
| Web search | Tavily | Purpose-built for AI agents, first-class LangChain integration, free tier. |
| Vector store | Chroma | Fine at this corpus size; FAISS is an equally valid swap. |

## Performance optimizations

Three concrete latency/cost fixes, made after tracing the actual call
pattern rather than guessing at bottlenecks. An explicit non-fix is
listed too, since "we decided not to do X" is as much a real engineering
decision as the fixes themselves.

**1. Batched reranking (`src/retrieval/retriever.py`).** Reranking used to make one
LLM call *per candidate document* -- 5 candidates meant 5 sequential
round-trips before routing, grading, or generation even happened. It now
asks for all scores in a single call as one JSON array
(`{"scores": [8, 3, 9, ...]}`), cutting reranking from N calls to 1 in the
normal case. If a model doesn't comply with the JSON format,
`_rerank_fallback_per_doc` preserves the original per-document behavior
as a safety net -- verified with a test asserting exactly 1 call on the
happy path and exactly N+1 calls when the batch format is ignored
(`tests/test_retrieval.py::TestBatchedReranking`).

**2. Model tiering (`src/agent/graph.py`).** `build_graph` now accepts a
separate `fast_llm` alongside the main `llm` -- structured
micro-decisions (routing, reranking, grading, query rewriting,
text-to-SQL) route to `fast_llm`; the final user-facing answer is the
only thing that uses the full `llm`. Verified directly: a single query
made exactly 3 fast-tier calls and exactly 1 strong-tier call, isolating
the expensive model to the one output actually judged on quality.
**Currently `fast_llm` defaults to the same model as `llm`** rather than
shipping a genuinely smaller model by default -- the obvious smaller Groq
option (`gpt-oss-20b`) shares the same Harmony-response-format parsing
bug already hit live with `gpt-oss-120b` (see the model table above), and
introducing a second unverified model was judged not worth the risk.
Pass an explicit `fast_llm` once a smaller model is verified reliable to
actually realize the latency/cost benefit.

**3. Real token-by-token streaming (`ui/chat_page.py`).** The graph is compiled
with a checkpointer and `interrupt_before=["generate"]`, so it runs
routing/retrieval/grading/tool-gathering automatically and then pauses
right before the final generation call instead of running it
synchronously. `ui/chat_page.py` then builds the exact same prompt the `generate`
node would have (via `build_direct_prompt`/`build_generation_prompt` --
extracted from that node as the single source of truth, not
reimplemented) and streams it directly via `llm.stream()`, so the answer
appears incrementally instead of the UI waiting for the entire response
before showing anything. Verified end-to-end with a stub LLM confirming
the graph pauses with an empty `answer` field and the correct next node,
then that the extracted prompt-building functions produce output
consistent with what the non-streaming graph path sends.

**Done (updated from an earlier "not done, deliberately decided
against"): real embedding model swap.** The original call was to skip
this, reasoning that TF-IDF's ~1ms/query was hard to beat given the
project was already at 100% recall/MRR and a sandboxed environment
supposedly couldn't reach huggingface.co anyway. That network
assumption was never actually tested until later — once it was, both
huggingface.co and pypi.org turned out to be reachable (the real
blocker was Windows/AVG Antivirus's HTTPS inspection breaking Python's
TLS trust, not network egress — see "Windows/AVG SSL note" below), so
the original recommendation (a real Qwen3-Embedding model) was no
longer blocked and got implemented. Measured live on this machine:
`Qwen3-Embedding-0.6B` on CPU costs **~0.94s per query embed** versus
TF-IDF's ~1ms — a real, honest latency cost, not a free upgrade — but
retrieval quality holds at the same 100% recall/MRR on the test set
(re-run and confirmed after the swap, see Evaluation below) while now
capturing genuine semantic similarity instead of co-occurrence
statistics, which matters most on corpora or queries this small test
set doesn't stress. Sub-1s is an acceptable cost inside a chat turn
that already makes several LLM calls.

**Windows/AVG SSL note.** `SSLCertVerificationError: Basic Constraints
of CA cert not marked critical` was reproduced live against
huggingface.co and pypi.org both — caused by AVG Antivirus's HTTPS
traffic inspection re-signing outbound TLS with its own root cert, which
Python doesn't trust by default even though the OS does. Fixed via
`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` env vars (see `.env.example`) for
Python's own HTTP clients, and `system-certs = true` in
`pyproject.toml`'s `[tool.uv]` section for `uv`'s separate (Rust) TLS
stack, which doesn't read those Python env vars at all. Both settings
are no-ops on a machine without AVG.

**Follow-up bug, found after the above shipped**: the app still crashed
on a fresh run with `RuntimeError: Cannot send a request, as the client
has been closed` from inside `SentenceTransformer(...)`, even with the
model already fully cached locally and the SSL vars correctly set.
Root cause: sentence-transformers/huggingface_hub does an online
metadata check by default on every load, cache or no cache, and that
network round-trip through AVG's proxy is what's actually fragile — the
error above is a confusing secondary failure from huggingface_hub's
httpx-based retry wrapper reusing a client a prior SSL failure had
already torn down, masking the real one. Setting `HF_HUB_OFFLINE=1`
inside `Qwen3Embeddings.__init__` did NOT fix it: huggingface_hub reads
that env var into a module-level constant at import time, and
`sentence_transformers` is already imported by the time `__init__` runs,
so the later env var write has no effect. Fixed at the root instead by
passing `local_files_only=True` directly to `SentenceTransformer(...)` —
a genuine per-call argument huggingface_hub checks dynamically, not a
snapshotted one — which skips the network call entirely when the model
is already cached (confirmed live: loads in <1s with zero network
calls, even with no SSL vars set at all), falling back to a real fetch
only if that raises (cache genuinely missing).

## Evaluation & testing

Three layers, matching the assignment's "explain test case build to
assure quality" requirement at increasing depth:

```bash
uv sync   # pytest (dev dependency group) is included automatically

# 1. Unit tests -- no API key needed, no network beyond the local DB/index
uv run pytest tests/ -v

# 2. Retrieval-only metrics (precision@k, recall@k, MRR) -- no API key needed
uv run python eval/run_eval.py

# 3. + tool-routing accuracy, end-to-end generation checks, multi-turn memory test
uv run python eval/run_eval.py --full

# 4. + faithfulness and answer relevancy scoring (RAGAS-style, LLM-judge)
uv run python eval/run_eval.py --ragas

# 5. Traditional vs. agentic head-to-head (see the comparison section below)
uv run python eval/compare_modes.py
```

Every run of `run_eval.py` or `compare_modes.py` writes its results to
`eval/results/*.json` -- the app's **Quality Metrics** tab reads these
files directly, so there's no manual step to transcribe numbers into the
UI (see the app description above).

### 1. Unit tests (`tests/`, 130 tests)
Pure-function tests requiring no LLM: RRF fusion math in isolation, the
specs-DB SQL safety guardrails (injection/multi-statement/non-SELECT all
verified blocked), the exact-match→LIKE fallback that fixed a live bug
(see below), LSA embedding shape/normalization/similarity properties
(`LSAEmbeddings` is no longer the live default -- see Model selection
above -- but stays in `src/retrieval/embeddings.py` as a zero-download fallback,
and its tests still pass unmodified), and the router/grading response
parsers (extracted from the graph's node closures specifically to make
them testable independently of a live LLM).

### 2-3. Retrieval + end-to-end (`eval/test_cases.py`, 16 cases + 1 memory test)
- **single_hop / multi_hop** — answer in one doc vs. requires synthesizing across several
- **adversarial** — phrased to avoid lexical overlap with the source document (tests whether dense retrieval / the correction loop compensates for BM25's blind spot)
- **out_of_scope** — should be declined, not hallucinated
- **no_retrieval** — conversational input, should skip all tools
- **specs_lookup** — must route to the specs DB, not docs
- **web_required** — must route to web search (skipped gracefully if `TAVILY_API_KEY` isn't set)
- **multi_tool** — requires combining `docs` + `specs` in one answer
- a separate two-turn **memory** test verifying a follow-up question ("how does that compare to...") correctly resolves via `chat_history`

Retrieval-only results (k=5): **100% recall, 100% MRR** across all docs-tool
categories, including adversarial phrasing and the multi-tool case's docs half.

### 4. RAGAS-style generation metrics (`eval/llm_judge_eval.py`)
Faithfulness (are the answer's claims actually supported by retrieved
context?) and answer relevancy (does the answer address the question
asked?), implemented as direct LLM-judge calls rather than via the
`ragas` package. **Why not `ragas` directly**: it was installed and
tested during development, but `ragas==0.4.3` imports
`langchain_community.chat_models.vertexai`, a submodule removed from
current `langchain-community` (which is being actively sunset) --
`ModuleNotFoundError` reproduced directly, not a hypothetical
compatibility concern. Since RAGAS's metrics are themselves just
structured LLM-judge prompts, reimplementing the two used here avoids the
dependency risk while measuring the same thing with the Groq LLM already
in the pipeline.

### Bugs found and fixed during testing (real examples, not hypothetical)
1. **Specs DB exact-match bug**: `WHERE model = 'H100 SXM'` returned 0
   rows because the real stored value is `'NVIDIA H100 80GB SXM'`. Fixed
   with (a) showing the LLM real column values before it writes SQL, and
   (b) a word-level `LIKE`-fallback retry as defense-in-depth. Covered by
   `tests/test_specs_tool.py::TestLoosenExactMatchToLike`.
2. **Router hallucination bug**: a sourdough recipe question routed to
   `tools=['none']`, and the model answered fully from unverified memory
   with total confidence -- the exact failure mode this system exists to
   prevent, happening silently. Fixed by (a) tightening the router prompt
   so `none` is reserved for genuinely conversational input only, and (b)
   a second line of defense in the no-tools generation prompt that
   refuses factual questions instead of answering from memory even if
   routing misfires again.
3. **LSA embedding crash on tiny/homogeneous corpora**: `max_df=0.9`
   pruning can remove every shared term from a very small corpus,
   crashing instead of degrading. Fixed with a fallback retry without
   `max_df` pruning. Covered by
   `tests/test_embeddings.py::test_n_components_capped_by_corpus_size`.
4. **`gpt-oss-120b` Harmony-format parsing failure**: the original model
   choice intermittently threw `groq.BadRequestError:
   output_parse_failed` (empty `failed_generation`, unrelated to prompt
   content) -- a real, current, provider-side issue with how `gpt-oss`
   models' internal "Harmony" response format gets parsed through Groq's
   API, reproduced live during `--full` eval and confirmed against
   multiple open `langchain-ai/langchain` GitHub issues on this exact
   model/provider pairing. Fixed by switching the default model to
   `qwen/qwen3.6-27b`, which isn't a Harmony-format model and doesn't
   have this failure mode.
5. **`<think>` reasoning leaking into displayed output**: the very first
   live test of the app (a plain "hi" greeting) showed the model's
   internal `<think>...</think>` chain-of-thought as part of the visible
   answer, before the actual response. Routing and grading were
   unaffected (both use regex search for a pattern anywhere in the
   response text, which still finds it around a `<think>` block), but two
   places used raw response text directly: the final answer, and the
   rewritten search query in the corrective retry loop -- the latter is a
   functional bug, not just cosmetic, since a `<think>` block prepended
   to a retrieval query would badly degrade search quality. Fixed with
   `strip_thinking_tags()` (non-streaming) and `stream_without_thinking()`
   (streaming-safe, handles a tag split across multiple chunks) in
   `src/llm_utils.py`, covered by 26 tests including 8 parametrized
   chunk-size variations of the streaming filter.
6. **Reasoning burning the entire token budget on a structured
   micro-decision**: a real question comparing two GPU models NOT in the
   specs database ("Adakah gtx1050 lebih baik dari rtx5060") caused the
   text-to-SQL call to spend its *entire* response reasoning inside a
   `<think>` block about whether the model names might be typos or a
   "trick question" -- and got cut off before ever emitting a single line
   of SQL. `validate_sql` correctly rejected the resulting garbage (no
   data was hallucinated), but the deeper problem is that routing,
   reranking, grading, and SQL generation are structured micro-decisions
   that never needed open-ended chain-of-thought reasoning in the first
   place. Fixed at the root: `build_graph`'s `fast_llm` is now
   auto-configured (when not explicitly provided) via Groq's
   `reasoning_effort` parameter -- `'none'` for Qwen3 models (fully
   disables reasoning), `'low'` for gpt-oss models (their fastest
   supported level, since they don't support `'none'`) -- so these calls
   stop reasoning altogether rather than merely hoping a stray `<think>`
   block gets cleaned up after the fact. `strip_thinking_tags` was also
   applied to `specs_tool.py`'s SQL response (missed in the fix above) as
   defense-in-depth. Covered by 7 tests in `tests/test_model_tiering.py`
   plus 3 regression tests in `tests/test_specs_tool.py` reproducing the
   exact live scenario (including the cut-off, no-SQL-produced case).
7. **Citation granularity gap against benchmarked 2026 production
   research** (not a live bug -- found via research comparison, not
   testing): current guidance is explicit that *"the minimum bar for
   production RAG is source attribution at the claim level... not
   'according to company documents,' but 'according to the Q3 Financial
   Review, page 12.'"* Citations previously pointed to filename only
   (`[08_quantization_techniques.md]`), even though section-level
   metadata was already being sent to the LLM in its context and already
   stored per-chunk in the structured citation data used by the UI panel
   -- the data existed, the instruction just didn't ask the LLM to cite
   at that granularity. First fixed by asking the LLM to cite
   `[filename.md § section]` directly, then **superseded** by
   pre-assigned numbered citations (`[1]`, `[2]`, ...): sources are
   numbered in order *before* the LLM ever sees them, and the LLM is told
   to cite using only those numbers rather than formatting its own
   citation string -- the same "structured output over free-text
   generation" principle used for the router's tool selection and the
   reranker's scores, for the same reliability reason. The same source
   (same document+section, same specs model, same web URL) reuses its
   original number if cited again later in the answer. Numbering is
   sequential and continuous across docs → specs → web (no resets per
   tool). The citation panel in the UI is sorted by number to match, so
   `[3]` in the answer text goes straight to the third card, not a
   type-grouped section requiring a search. Covered by 7 tests in
   `tests/test_prompt_building.py`, including sequential-numbering,
   duplicate-source-reuse, and cross-tool-numbering-continuity cases.
8. **`st.chat_input` losing its sticky-bottom position inside `st.tabs`**:
   after adding the Session Dashboard and Quality Metrics tabs, the chat
   input rendered inline at the top of the Chat tab instead of pinned to
   the bottom of the page -- a confirmed, documented Streamlit behavior
   (chat_input loses its special fixed-position CSS when nested inside
   tabs/columns/containers), not something specific to this app. Fixed by
   moving the `st.chat_input()` call to the top level of `main()`,
   outside `st.tabs()` entirely, and passing its return value into
   `render_chat_tab()` as a parameter instead of calling it from inside
   the tab.
9. **Live reindex crashing silently on Windows** (found building the
   Documents tab's add/delete editor): `ingest.build_indexes()` used to
   `shutil.rmtree()` `data/chroma_db/` before rebuilding it from
   scratch. That's fine for a one-off offline script run, but `main()`
   calls `get_app()` unconditionally on every rerun regardless of which
   page is active, so a live Chroma/SQLite connection to that directory
   is already open by the time a user adds or deletes a document in the
   running app. Windows (unlike POSIX) refuses to delete a file another
   handle in the same process still has open, raising
   `PermissionError: [WinError 32] ... used by another process` --
   reproduced live, crashing the script run server-side with no visible
   error in the UI, so the add/delete appeared to succeed (the file
   write itself worked) while the search index silently never
   rebuilt. Clearing every cached reference and forcing `gc.collect()`
   before the rebuild did NOT fix it: chromadb keeps its own internal
   client registry keyed by persist directory specifically to avoid
   duplicate connections to the same path, so the file stayed open
   regardless of this module's own references. Fixed at the root by
   never deleting the directory at the filesystem level at all --
   `build_indexes()` now calls Chroma's own `delete_collection()` API to
   drop and recreate the collection's data through the already-open
   connection, which works whether or not another live connection to
   that same path exists. Verified end-to-end with a scripted
   add → view → delete → restore cycle against a running instance,
   confirming the corpus count and retrievable content update correctly
   at each step.

## Traditional RAG vs. agentic RAG — what this demonstrates

**Structural difference (verified by code and tests) vs. performance
difference (needs real measurement) are two separate claims — this
section keeps them separate rather than implying one proves the other.**

### Structural difference — proven by the code itself

The shipped app (`app.py`) is agentic-only — a single product, no mode
switching. `build_traditional_graph` in `src/agent/graph.py` exists as a
separate, internal comparison tool (not a user-facing feature) used
directly by `eval/compare_modes.py` below. It's a deliberately minimal,
textbook single-pass pipeline: retrieve top-k → rerank → generate once,
no matter what came back. `tests/test_traditional_graph.py` verifies
directly (not by inference) that this pipeline has no grading step, no
retry, no tool routing, and no memory in its state schema at all.

**Critical design choice**: traditional mode uses the exact same hybrid
retrieval (dense+BM25 RRF) and LLM reranking as agentic mode — not a
weaker retriever. This is deliberate. Retrieval quality and
agentic-vs-traditional are two separate engineering axes; conflating them
would make any observed difference ambiguous (is agentic actually better,
or did traditional just get worse retrieval?). Holding retrieval
identical isolates the comparison to the one thing actually being tested:
does a decision-making layer (route / grade / retry / remember) on top of
the *same* retrieval change the outcome.

### Performance difference — measure it, don't assume it

`eval/compare_modes.py` runs both modes on the same subset of the test
set (`single_hop`, `multi_hop`, `adversarial`, `out_of_scope` — the
categories traditional mode can fairly attempt, since it can't reach
specs/web/memory) and scores both with the same faithfulness/relevancy
judges used elsewhere in this project, plus correctness against expected
terms and (for `out_of_scope` cases specifically) whether each mode
correctly declined instead of hallucinating:

```bash
uv run python eval/compare_modes.py
```

This produces real per-question and aggregate numbers — mean
faithfulness, mean relevancy, correctness rate, correct-decline rate,
LLM-calls-per-query, and latency-per-query, for each mode. **Run this
and cite the actual output** rather than asserting agentic mode performs
better in the abstract; the honest claim is only as strong as this
script's numbers turn out to be.

| | Traditional RAG (`build_traditional_graph`, internal comparison tool) | Agentic RAG (this app) |
|---|---|---|
| Control flow | Fixed 3-node line: retrieve → rerank → generate | Graph with conditional branches, a retry cycle, and multi-tool fan-out |
| Query handling | Every input retrieved identically, docs only | Routes per-query across 3 tools (or none), using conversation memory |
| Bad retrieval — mechanism | No grading step exists in the graph (verified by test) | `grade_documents` node exists and can trigger `rewrite_query` (verified by test) |
| Bad retrieval — actual outcome | Measure with `compare_modes.py`, don't assume | Measure with `compare_modes.py`, don't assume |
| Multi-source | N/A — one fixed source | Combines heterogeneous sources (unstructured docs + structured DB + live web) in one answer when needed |
| Failure mode — mechanism | No refusal instruction in the generation prompt (verified by code) | Generation prompt explicitly instructs declining on insufficient evidence (verified by code) |
| Failure mode — actual rate | Correct-decline rate: measure with `compare_modes.py` | Correct-decline rate: measure with `compare_modes.py` |
| Retrieval quality | **Identical** — same `HybridRetriever`, same reranker | Identical — this is the controlled variable, not the comparison |

## Project structure

Reorganized from an earlier flat `app.py` (922 lines) + `src/*.py`
(8 files, largest 915 lines) layout into packages grouped by role, once
both files had grown to mix several genuinely separable concerns.
`app.py` is now a 20-line entry point; `graph.py`'s state/prompts/
parsing/LLM-factory pieces are each their own module, with the actual
LangGraph node closures/wiring left untouched in `agent/graph.py` (a
deliberately conservative split -- see the module docstrings for why
the closures weren't also hoisted to free functions). Every package's
`__init__.py` re-exports its public surface, so most call sites are
unchanged (`from agent import build_graph`, `from retrieval import
HybridRetriever`, etc.) -- only internals (e.g. `_build_default_fast_llm`)
need a submodule-qualified import.

```
agentic-rag/
├── app.py                    # thin entry point: sys.path setup, hands off to ui.app.main()
├── pyproject.toml / uv.lock  # uv-managed deps (requirements.txt kept in sync)
├── ui/                        # Streamlit UI, one module per page
│   ├── app.py                  # main(): st.set_page_config, orchestrates all pages
│   ├── styles.py                # CUSTOM_CSS + tool/citation icon & color maps
│   ├── config.py                # NAV_PAGES, model options, defaults
│   ├── resources.py             # get_retriever/get_app (st.cache_resource)
│   ├── sidebar.py                # nav radio + API key setup
│   ├── chat_page.py              # chat rendering, citations, reasoning trace, token streaming
│   ├── documents_page.py         # corpus view/add/delete + reindex, specs DB summary
│   ├── settings_page.py           # model selection, token-optimization (fast model), tool legend
│   ├── sources_page.py            # session-wide citation log
│   └── analytics_page.py          # session analytics + offline eval metrics
├── data/
│   ├── corpus/*.md            # 12 self-authored documents
│   ├── build_specs_db.py      # builds specs.db from sourced hardware specs
│   └── specs.db                # generated: GPU + SSD specs, real data
├── src/
│   ├── agent/                  # the orchestrating LangGraph agent
│   │   ├── state.py              # AgentState/TraditionalRAGState, retry/history constants
│   │   ├── llm_factory.py         # build_llm/build_fast_llm, reasoning-effort tuning
│   │   ├── parsing.py             # router/grader response parsers
│   │   ├── prompts.py             # build_direct_prompt/build_generation_prompt + citation numbering
│   │   └── graph.py               # build_graph (node closures, unchanged) + traditional graph
│   ├── retrieval/               # the `docs` tool
│   │   ├── embeddings.py          # Qwen3-Embedding-0.6B (live) + LSA/TF-IDF (unused fallback)
│   │   ├── ingest.py              # chunk -> embed -> Chroma + BM25 index
│   │   └── retriever.py           # hybrid RRF retriever + LLM reranker
│   ├── tools/                   # the `specs` and `web` tools
│   │   ├── specs_tool.py          # text-to-SQL + safety guardrails
│   │   └── web_tool.py            # Tavily wrapper
│   ├── llm_utils.py             # retry-with-backoff wrapper for all LLM calls
│   └── mcp_server.py             # optional: exposes the 3 tools over MCP for a second consumer
├── tests/                     # pytest unit tests (130 tests, no LLM/API key needed)
│   ├── test_retrieval.py       # RRF fusion math + batched reranking + end-to-end retrieval
│   ├── test_specs_tool.py      # SQL safety guardrails + LIKE-fallback regression
│   ├── test_graph_parsing.py   # router/grading response parsers + 2 live-bug regressions
│   ├── test_prompt_building.py # extracted generate() prompt functions (used by ui/chat_page.py streaming)
│   ├── test_embeddings.py      # LSA embedding correctness + edge cases (unused-fallback coverage)
│   ├── test_llm_utils.py       # retry-with-backoff + <think>-tag stripping (2 live bugs)
│   ├── test_model_tiering.py   # fast_llm reasoning_effort auto-configuration
│   └── test_traditional_graph.py  # verifies the traditional mode has no agentic mechanisms
└── eval/
    ├── test_cases.py           # 16 hand-labeled cases + 1 memory test
    ├── run_eval.py              # retrieval + routing + generation + memory + RAGAS-style eval
    ├── llm_judge_eval.py        # faithfulness + answer relevancy (self-contained, see README)
    ├── compare_modes.py         # traditional vs agentic head-to-head, real measured numbers
    └── results/                 # JSON output from the two scripts above -- read by ui/analytics_page.py's Quality Metrics tab
```


