# Evaluating RAG Systems

Evaluating a RAG system requires separately assessing retrieval quality
and generation quality, since a good final answer requires both stages to
work correctly, and failures in one stage can mask or compound failures
in the other.

## Retrieval metrics

**Precision@k** — of the top-k retrieved chunks, what fraction are
actually relevant to the query. Measures how much irrelevant context is
being passed to the generator, which wastes context budget and can
distract the model.

**Recall@k** — of all chunks in the corpus that are actually relevant to
the query, what fraction were retrieved in the top-k. Measures whether
the retriever is missing evidence the generator would need to answer
correctly; low recall is a hard ceiling on answer quality no matter how
good the generator is, since a chunk that was never retrieved cannot be
used.

**Mean Reciprocal Rank (MRR)** — for queries with a single correct
answer chunk, the reciprocal of the rank at which that chunk first
appears, averaged across queries. Rewards retrievers that place the
correct chunk near the top of the ranked list, not just somewhere in the
top-k.

## Generation metrics (RAG-specific)

**Faithfulness / groundedness** — whether every claim in the generated
answer is actually supported by the retrieved context, typically assessed
by having an LLM (or human) check each claim in the answer against the
provided chunks. A response can be fluent and plausible while still being
unfaithful if it includes claims not present in the retrieved evidence —
this is the primary hallucination signal specific to RAG systems.

**Answer relevance** — whether the generated answer actually addresses
the user's question, independent of whether it is grounded in the
context. A response can be perfectly faithful to the retrieved chunks
while still failing to answer what was asked, if the retrieved chunks
themselves were off-topic.

**Context precision / context relevance** — evaluates whether the
retrieved chunks that were actually used for generation were relevant, as
distinct from raw retrieval precision, since a generator may selectively
ignore some retrieved chunks.

## Automated evaluation frameworks

Frameworks such as RAGAS compute these metrics using an LLM as a judge:
given a question, the retrieved contexts, and the generated answer, an
LLM is prompted to decompose the answer into individual claims and verify
each against the context (for faithfulness), and to assess whether the
answer addresses the question (for answer relevance), producing scores
without requiring human-labeled ground truth for every metric. Ground
truth answers or ground truth relevant chunks are still valuable where
available, since they allow direct computation of retrieval recall and
precision rather than relying entirely on LLM-judged proxies.

## Building a test set

A practical RAG test set pairs each question with (a) the expected
answer or key facts it must contain, and (b) the source chunk(s) that
should be retrieved to support that answer. This allows retrieval and
generation to be scored independently: a wrong final answer with correct
retrieved chunks points to a generation problem, while a wrong final
answer with irrelevant retrieved chunks points to a retrieval problem,
which is a distinction that overall end-to-end accuracy alone cannot
provide.
