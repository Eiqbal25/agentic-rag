# Chunking Strategies for RAG

Chunking is the process of splitting source documents into smaller units
before embedding and indexing. Chunk boundaries directly affect retrieval
quality: chunks that are too large dilute the embedding with irrelevant
content and waste context window space, while chunks that are too small
lose surrounding context needed to answer the query correctly.

## Fixed-size chunking

The simplest approach splits text into fixed-length windows (measured in
characters or tokens), often with an overlap (e.g., 10-20% of chunk size)
between consecutive chunks so that content near a boundary is not split
across two chunks with no shared context. Fixed-size chunking is fast and
requires no document structure, but it ignores semantic boundaries and can
cut a sentence or idea in half.

## Recursive character/structure-aware chunking

Splits text using a prioritized list of separators (e.g., paragraph
breaks, then sentence breaks, then word breaks), recursively falling back
to a finer-grained separator only when a chunk still exceeds the target
size. This tends to produce chunks that respect natural document
structure better than naive fixed-size splitting.

## Semantic chunking

Uses embedding similarity between consecutive sentences to detect topic
shifts: sentences are grouped into a chunk as long as consecutive sentence
embeddings remain similar, and a new chunk starts when similarity drops
below a threshold, indicating a topic boundary. This produces
variable-length chunks that align with actual semantic units at the cost
of additional embedding computation during the indexing phase.

## Document-structure-aware chunking

For structured documents (legal statutes with articles, contracts with
clauses, technical manuals with sections), chunking along existing
structural boundaries — one chunk per article, clause, or section —
generally outperforms generic text splitting, because the structural
units were authored to be self-contained units of meaning. This requires
document-specific parsing logic (e.g., regex or a parser for numbered
articles) rather than a generic splitter.

## Parent-child (small-to-big) chunking

Small chunks are used for the embedding/retrieval step (because small,
focused chunks embed more precisely and match queries better), but when a
small chunk is retrieved, its larger parent chunk or full source section
is what actually gets passed to the generator. This decouples retrieval
precision from generation context, addressing the tension between
"chunks small enough to retrieve accurately" and "chunks large enough to
contain the full answer context."

## Chunk size selection in practice

There is no universally correct chunk size; it depends on the embedding
model's effective context window, the granularity of facts in the source
documents, and the generator's context budget. Common starting points are
in the 200-500 token range for dense prose, with structure-aware or
parent-child strategies preferred whenever the source documents have
identifiable natural units, since arbitrary fixed-size splitting on
structured documents (like statutes or manuals) tends to produce chunks
that straddle unrelated provisions or steps.
