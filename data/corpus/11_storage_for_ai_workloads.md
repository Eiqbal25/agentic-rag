# Storage Architectures for AI Workloads

AI training and fine-tuning workloads place demands on storage systems
that differ from traditional enterprise storage patterns, which has
driven the emergence of storage architectures purpose-built for AI.

## Why GPU memory capacity is a constraint

GPU memory (VRAM) is the fastest tier available to a training or
fine-tuning job, but it is also the most limited and expensive. As model
sizes grow, the combined memory required for model weights, optimizer
states, gradients, and activations frequently exceeds what a single GPU
or even a multi-GPU server can hold, forcing a choice between (a) using
more GPUs purely to add memory capacity, which is costly, or (b)
offloading some of that state to a lower tier of memory or storage that
is slower but far cheaper per gigabyte.

## SSD-based memory extension for LLM training

One approach to this constraint is to use high-performance NVMe SSD
storage as an extension of GPU memory: optimizer states, gradients, or
inactive model layers that are not needed at every instant of the
training step can be offloaded to NVMe storage and streamed back in when
needed, rather than requiring them to permanently occupy GPU VRAM. This
allows fine-tuning of models that would not otherwise fit in the
available GPU memory, at the cost of some additional latency each time
offloaded state must be read back, which is why the read/write bandwidth
and latency characteristics of the SSD tier directly determine how much
training-throughput penalty this offloading approach incurs. The closer
NVMe read/write performance gets to GPU-memory-adjacent speeds, the
smaller the throughput penalty from offloading, which is why enterprise
SSD design for AI workloads focuses heavily on sustained random-read
throughput and low tail latency rather than only peak sequential
bandwidth.

## Enterprise SSD design considerations for AI

**Sustained throughput vs. burst throughput** — training workloads
involve long, sustained read/write patterns (streaming large datasets,
periodic checkpoint writes) rather than short bursts, so SSDs intended
for AI infrastructure are evaluated on sustained throughput under
continuous load, which can differ substantially from advertised peak
burst specifications.

**Endurance (write durability)** — frequent checkpoint writes and
optimizer-state offloading generate substantially more write traffic
than typical enterprise storage workloads, making drive endurance
(measured in total bytes written over the drive's lifetime, or drive
writes per day) an important selection criterion for AI training storage,
not just for capacity or raw speed.

**Data center integration** — enterprise SSDs for AI infrastructure are
typically deployed alongside IC design and controller-level engineering
so that the storage stack (controller firmware, SSD, and the software
layer coordinating offload between GPU memory and SSD) is co-optimized
end to end, rather than treating the SSD as an interchangeable commodity
component.

## Implication for cost-efficient on-prem LLM infrastructure

Combining a moderate number of GPUs with a well-engineered NVMe storage
tier for memory extension can make fine-tuning of larger models
economically accessible on hardware that would otherwise be undersized
for the task, shifting the cost trade-off from "buy enough GPU memory to
hold everything" toward "buy enough fast storage to extend a smaller GPU
memory footprint efficiently." This is particularly relevant for
organizations that want on-premise fine-tuning capability without the
capital cost of the largest, highest-memory GPU configurations.
