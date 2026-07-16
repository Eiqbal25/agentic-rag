# Agentic RAG: Patterns and Architectures

Agentic RAG refers to retrieval-augmented generation systems in which an
LLM acts as a controller that makes decisions about *whether*, *what*, and
*how many times* to retrieve, rather than following one fixed
retrieve-then-generate pass. The agent has access to retrieval as a tool
among possibly several tools, and it reasons about intermediate results
before deciding on the next action.

## Common agentic RAG patterns

**Routing agent.** Before retrieving, the agent classifies the incoming
query and routes it to the appropriate source or strategy — for example,
distinguishing a question that needs the vector index from one that needs
a SQL database, a web search tool, or no retrieval at all because it is
conversational.

**Query rewriting / decomposition.** Complex or multi-hop questions are
broken into sub-questions that are each retrieved independently, and the
sub-answers are combined. For example, a question comparing two
techniques for reducing serving cost can be split into one sub-query per
technique, retrieved separately, then synthesized into a single
comparative answer.

**Corrective RAG (CRAG).** After retrieval, a grading step (often another
LLM call, sometimes a lightweight classifier) evaluates whether the
retrieved chunks are actually relevant to the query. If the chunks are
judged irrelevant or insufficient, the agent triggers a corrective action:
rewriting the query and re-retrieving, expanding the search to a
fallback source, or reducing reliance on retrieved context in the final
answer.

**Self-RAG.** The generation model is trained or prompted to emit
reflection tokens that assess (a) whether retrieval is needed at all for a
given segment, (b) whether retrieved passages are relevant, and (c)
whether the generated output is actually supported by the passages. This
turns retrieval and verification into an integrated part of generation
rather than a separate pipeline stage.

**Adaptive retrieval depth.** Instead of a fixed top-k, the agent
dynamically decides how many chunks or how many retrieval rounds are
needed based on the confidence of intermediate answers or a stopping
criterion evaluated at each iteration.

**Multi-agent RAG.** Separate agents specialize in different roles — a
retriever agent, a critic/grader agent, a synthesis agent — coordinating
through a shared state rather than a single monolithic prompt.

## Implementing agentic RAG as a graph

Frameworks such as LangGraph model agentic RAG as a directed graph of
nodes and conditional edges rather than a linear chain. Each node is a
function (often an LLM call or a retrieval call), and edges route the
control flow based on the output of a node — for instance, a "grade
documents" node can route either to "generate" (if documents are
relevant) or back to "rewrite query" (if not), creating an explicit loop
that a linear chain cannot express. This graph-based control flow is what
distinguishes agentic RAG implementations from traditional RAG chains at
the code level: traditional RAG is expressible as a straight-line
sequence of function calls, while agentic RAG requires conditional
branching and cycles.

## Trade-offs

Agentic RAG improves answer quality and reduces hallucination on
ambiguous or multi-hop queries, but at the cost of higher latency (more
LLM calls per user query) and higher operating cost. A well-designed
agentic system should therefore include a routing step that skips the
agentic loop entirely for simple queries, applying the more expensive
corrective loop only when the initial retrieval is judged weak.
