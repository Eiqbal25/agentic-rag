# Vector Databases for Retrieval Systems

A vector database stores high-dimensional embeddings and supports
approximate nearest neighbor (ANN) search, returning the vectors closest
to a query vector under a chosen distance metric without scanning the
entire dataset.

## Indexing algorithms

**Flat (brute-force) index** — computes exact distance to every stored
vector. Guarantees exact results but scales linearly with corpus size,
making it impractical beyond roughly a few hundred thousand vectors for
low-latency use cases.

**HNSW (Hierarchical Navigable Small World)** — builds a multi-layer graph
where each node connects to its approximate nearest neighbors; search
starts at a sparse top layer and descends, refining the candidate set at
each layer. HNSW offers strong recall-latency trade-offs and is the
default index type in Chroma, Qdrant, and Weaviate.

**IVF (Inverted File Index)** — partitions the vector space into clusters
(via k-means), and search is restricted to the clusters nearest the query,
trading some recall for large speedups. Often combined with product
quantization (IVF-PQ) to compress vectors and reduce memory footprint,
which matters when the index must fit on-device rather than in a
memory-heavy server.

**DiskANN / disk-based ANN** — indexes designed to serve nearest-neighbor
queries directly from SSD rather than requiring the full index to reside
in RAM, which is critical for billion-scale corpora where an all-RAM index
would be prohibitively expensive. These indexes are latency-sensitive to
storage read patterns: random 4KB reads at high queries-per-second are the
dominant cost, which is why NVMe SSDs with low tail latency are preferred
over SATA SSDs or spinning disks for production disk-based ANN.

## Popular vector database options

- **Chroma** — embedded, lightweight, good for prototyping and small-to-
  medium corpora; runs in-process or as a client-server deployment.
- **FAISS** — a library (not a full database) from Meta for efficient
  similarity search; widely used as the search backend inside larger
  systems rather than as a standalone service.
- **Qdrant** — production vector database with filtering, payload storage,
  and both cloud and self-hosted deployment.
- **Milvus** — designed for large-scale, distributed deployments with
  support for multiple index types and horizontal scaling.
- **pgvector** — a PostgreSQL extension that adds vector similarity search
  to a relational database, useful when the corpus already lives in
  Postgres and a separate vector store would add unnecessary
  operational overhead.

## Metadata filtering

Production retrieval rarely relies on vector similarity alone. Most vector
databases support attaching metadata (source document, date, section,
access permissions) to each vector and filtering the ANN search to only
consider vectors matching metadata constraints — for example, restricting
retrieval to documents the requesting user is authorized to see, or to
documents published after a given date. This hybrid of vector similarity
and structured filtering is standard in enterprise retrieval systems.
