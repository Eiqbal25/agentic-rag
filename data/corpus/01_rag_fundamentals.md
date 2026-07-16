# Retrieval-Augmented Generation: Fundamentals

Retrieval-Augmented Generation (RAG) is an architecture that combines a
parametric language model with a non-parametric knowledge source, typically
a document index, to ground generated text in retrieved evidence. The core
motivation is that large language models encode knowledge implicitly in
their weights at training time, which makes that knowledge static, hard to
update, and prone to hallucination when the model is asked about facts
outside its training distribution or facts that have since changed.

A standard RAG pipeline has three stages:

1. **Indexing** — source documents are split into chunks (paragraphs,
   sections, or fixed-size windows), each chunk is converted into a dense
   vector using an embedding model, and the vectors are stored in a vector
   index (e.g., FAISS, Chroma, Qdrant, Milvus) alongside the original text
   and metadata such as source, page number, or section title.

2. **Retrieval** — at query time, the user's question is embedded with the
   same embedding model, and the index returns the top-k chunks whose
   vectors are closest to the query vector under a similarity metric
   (commonly cosine similarity or inner product).

3. **Generation** — the retrieved chunks are inserted into the LLM's
   context window alongside the original question, and the model is
   instructed to answer using only (or primarily) the supplied evidence.

## Why RAG over fine-tuning

Fine-tuning bakes knowledge into model weights, which is expensive to
update and does not naturally support citing sources. RAG keeps the
knowledge base external and swappable: updating an index by re-embedding a
changed document is orders of magnitude cheaper than retraining a model,
and because the generation step can be instructed to quote or reference
the exact chunk it used, RAG systems support traceability that fine-tuned
models cannot easily provide.

## Failure modes of naive (traditional) RAG

A single-pass "retrieve top-k, then generate" pipeline has several known
weaknesses:

- **Irrelevant retrieval** — if the query is ambiguous, poorly phrased, or
  the embedding model has weak coverage of the domain, the top-k chunks
  may not actually contain the answer, and the LLM has no mechanism to
  notice this or correct for it.
- **No query understanding** — traditional RAG treats every query
  identically, whether it needs a single fact, a multi-hop comparison
  across several documents, or no retrieval at all (e.g., "hello, how are
  you?").
- **Fixed retrieval depth** — the value of k is chosen once at design
  time, regardless of whether a given question needs 2 chunks or 20.
- **No verification step** — the generated answer is returned even if it
  is not actually supported by the retrieved context, which is a primary
  source of hallucination in RAG systems.

These limitations motivate agentic RAG, where an LLM-driven controller
makes retrieval decisions dynamically rather than following a fixed
pipeline.
