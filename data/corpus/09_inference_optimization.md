# LLM Inference Optimization

Serving large language models efficiently requires optimizing both
latency (time to respond to a single request) and throughput (requests
served per unit time across many concurrent users), which are often in
tension with each other.

## KV cache

During autoregressive generation, each new token's attention computation
depends on the key and value projections of all previous tokens. Rather
than recomputing these projections at every step, they are cached (the
"KV cache") and reused, turning what would be quadratic recomputation
into a linear amount of new work per generated token. The KV cache grows
linearly with sequence length and batch size and can become the dominant
consumer of GPU memory during long-context or high-concurrency serving,
often exceeding the memory used by the model weights themselves at long
context lengths.

## Continuous batching

Naive batching waits for a full batch of requests to arrive and finishes
only when every sequence in the batch has completed, wasting compute on
already-finished sequences while they wait for the longest one. Continuous
(or dynamic) batching instead evicts completed sequences and admits new
requests into the batch at each generation step, keeping GPU utilization
high under variable request lengths and arrival times. This is one of the
largest throughput improvements available in production LLM serving
systems such as vLLM and TGI.

## PagedAttention

A memory management technique (introduced in vLLM) that manages the KV
cache in fixed-size blocks similar to virtual memory paging in operating
systems, rather than requiring each sequence's KV cache to occupy a single
contiguous memory region. This reduces memory fragmentation and allows
memory to be shared between sequences that share a common prefix (e.g.,
the same system prompt), substantially increasing the number of
concurrent sequences that fit in a given amount of GPU memory.

## Speculative decoding

A smaller, faster "draft" model proposes several candidate tokens ahead,
and the larger target model verifies them in a single forward pass,
accepting the draft tokens that match what the target model would have
generated and rejecting the rest. Because verifying multiple tokens in
one forward pass is cheaper than generating them one at a time, this can
reduce end-to-end latency without changing the target model's output
distribution, provided the draft model's proposals are accepted often
enough to offset the overhead of running two models.

## Hardware and storage considerations

At low batch sizes, LLM inference is typically bottlenecked by memory
bandwidth (moving weights from GPU memory to compute units) rather than
raw compute (FLOPs), which is why quantization and efficient KV cache
management yield outsized throughput gains. For on-premise deployments
serving models too large to fit entirely in GPU memory, weights or KV
cache pages may need to be offloaded to system memory or fast NVMe
storage; in that regime, storage read latency and bandwidth become a
direct component of inference latency, making low-latency NVMe storage a
meaningful factor in on-prem LLM serving performance, not just in
training or fine-tuning workloads.
