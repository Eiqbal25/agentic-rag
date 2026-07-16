# Embedding Models for Retrieval

An embedding model maps text into a fixed-length dense vector such that
semantically similar text is placed close together under a distance
metric. Retrieval quality in a RAG system is bounded by embedding quality:
even a perfect generation model cannot produce a correct answer if the
retriever never surfaces the relevant chunk.

## Bi-encoders vs. cross-encoders

A **bi-encoder** embeds the query and each document independently into the
same vector space, so similarity is computed as a simple dot product or
cosine similarity between two pre-computed vectors. This allows documents
to be embedded once, offline, and enables sub-linear ANN search over
millions of vectors. Bi-encoders are what power the initial retrieval
stage in essentially all production RAG systems.

A **cross-encoder** takes the query and a candidate document together as a
single input and outputs a relevance score directly, allowing the model to
attend jointly over both texts. Cross-encoders are substantially more
accurate at judging relevance than bi-encoders because they can model
fine-grained interactions between query and document tokens, but they
cannot be pre-computed — every query-document pair requires a fresh
forward pass, which makes cross-encoders too slow to run over an entire
corpus. In practice, cross-encoders are used as a **reranking** stage: a
bi-encoder retrieves a candidate set (e.g., top-50), and a cross-encoder
re-scores and re-orders that smaller set before the top few are passed to
the generator.

## Model size and dimensionality trade-offs

Larger embedding models generally capture more nuanced semantics but
increase both indexing time and per-query latency. Smaller models (roughly
100–400 million parameters) such as BGE-small or MiniLM variants can be
run efficiently on CPU and are appropriate for prototypes and moderate
corpus sizes, while larger models (multi-billion parameter embedding
models) provide meaningfully better multilingual and domain-transfer
performance but typically require GPU inference to keep indexing and
query latency acceptable at scale.

## Multilingual and cross-lingual retrieval

When queries and documents are in different languages — for example, an
English legal query against a German-language statute corpus — the
embedding model must place semantically equivalent text from both
languages close together in vector space. Models trained with a
multilingual or cross-lingual objective (contrastive training across
parallel or comparable corpora in multiple languages) are required; a
monolingual embedding model will generally fail to bridge languages
because it has never learned to align them in the same vector space.
Cross-lingual retrieval quality is typically lower than monolingual
retrieval quality for the same corpus, because language mismatch adds a
second source of embedding noise on top of ordinary semantic ambiguity.

## Domain adaptation

General-purpose embedding models trained on web text often underperform
on specialized domains (legal, medical, technical/engineering) because
domain-specific vocabulary and phrasing are underrepresented in the
training distribution. Fine-tuning an embedding model on in-domain
query-document pairs, even a relatively small set (a few thousand
labeled pairs), typically improves retrieval recall substantially more
than switching to a larger general-purpose model.
