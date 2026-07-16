# Hybrid Retrieval and Reranking

Dense (embedding-based) retrieval and sparse (lexical) retrieval have
complementary strengths, which motivates combining them rather than
relying on either alone.

## Sparse retrieval: BM25

BM25 is a lexical ranking function built on term frequency and inverse
document frequency, scoring documents by how often query terms appear,
adjusted for document length and term rarity across the corpus. BM25 has
no notion of semantic similarity — it cannot match a query to a document
that expresses the same idea with different words — but it excels at
exact-match cases: proper nouns, identifiers, statute or article numbers,
acronyms, and rare technical terms, which dense embeddings sometimes blur
together because embedding models are trained to capture general semantic
similarity rather than exact token overlap.

## Dense retrieval

Dense retrieval, using embedding models as described in embedding model
documentation, captures semantic similarity — it can match a query to a
relevant document even when they share no vocabulary — but can
underperform on queries dominated by specific identifiers or rare terms
that the embedding model has not learned to weight heavily.

## Fusion methods

**Reciprocal Rank Fusion (RRF)** combines ranked lists from multiple
retrievers without requiring the retrievers' raw scores to be on the same
scale (which dense cosine similarity and BM25 scores are not). Each
document's fused score is the sum, across retrievers, of 1/(k + rank),
where rank is the document's position in that retriever's ranked list and
k is a small constant (commonly 60) that dampens the influence of very
high ranks. RRF can be weighted per retriever (e.g., giving dense
retrieval a higher weight than sparse retrieval, or vice versa) when one
retriever is known to be more reliable for a given corpus.

**Score normalization and linear combination** — an alternative to RRF
that normalizes each retriever's raw scores (e.g., min-max normalization)
before combining them with a weighted sum. This requires careful tuning
of normalization because dense and sparse score distributions differ
substantially, and is generally more fragile than RRF, which sidesteps
the scale problem entirely by operating on ranks rather than raw scores.

## Reranking

After an initial retrieval stage (dense, sparse, or fused) returns a
candidate set of, e.g., 20-50 chunks, a cross-encoder reranker re-scores
each query-chunk pair jointly and reorders the candidates, and only the
top few reranked chunks are passed to the generator. Reranking
consistently improves precision at the top of the ranked list because
cross-encoders model query-document interaction directly, at the cost of
one forward pass per candidate, which is why reranking is applied to a
small candidate set rather than the full corpus.

## When hybrid retrieval matters most

Hybrid retrieval provides the largest gains on corpora where queries mix
natural-language phrasing with exact identifiers — legal citations,
product model numbers, error codes, or named entities — since these are
precisely the cases where dense and sparse retrieval fail in different,
non-overlapping ways. On purely conversational, paraphrase-heavy query
sets, dense retrieval alone often captures most of the achievable recall,
and the incremental benefit of adding BM25 is smaller.
