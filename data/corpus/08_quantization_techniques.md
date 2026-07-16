# Quantization Techniques for LLM Inference and Training

Quantization reduces the numerical precision used to represent model
weights (and sometimes activations), shrinking memory footprint and often
increasing throughput, at the cost of some accuracy degradation.

## Precision formats

Models are typically trained in 16-bit floating point (FP16 or BF16).
Quantization moves weights to lower-precision integer or floating-point
formats: INT8 (8-bit integer), INT4 (4-bit integer), and specialized
float formats like FP8 designed for hardware that supports native
low-precision matrix multiplication. Each halving of bit-width roughly
halves the memory required to store the weights, so a model quantized
from 16-bit to 4-bit requires roughly a quarter of the original memory to
load.

## Post-training quantization (PTQ)

Weights are quantized after training is complete, without further
gradient updates. Simple PTQ methods quantize each weight independently
(round-to-nearest), which can cause meaningful accuracy loss at very low
bit-widths. More sophisticated PTQ methods such as GPTQ and AWQ use a
small calibration dataset to determine quantization parameters that
minimize the resulting output error, layer by layer, which substantially
improves accuracy retention at 4-bit precision compared to naive
rounding. AWQ specifically identifies and preserves higher precision for
the subset of weights that have the largest impact on output activations
("salient" weights), rather than treating all weights uniformly.

## Quantization-aware training (QAT)

Quantization effects are simulated during training itself (via a
fake-quantize forward pass with straight-through gradient estimation), so
the model learns weights that are more robust to the eventual precision
reduction. QAT generally achieves better accuracy at a given bit-width
than PTQ, but requires access to training infrastructure and compute,
whereas PTQ can be applied to an already-trained model with comparatively
little compute.

## Effects on inference

Lower-precision weights reduce the memory bandwidth needed to move
weights from memory to compute units during each forward pass, which is
often the dominant bottleneck in LLM inference (a regime described as
memory-bandwidth-bound rather than compute-bound, especially at low batch
sizes). This means quantization frequently improves inference throughput
and latency by a larger margin than the raw compute savings alone would
suggest, because it directly reduces the memory traffic bottleneck.
Quantizing only weights while keeping activations in higher precision
(weight-only quantization) is common because activations are more
sensitive to precision loss and change per-input, making them harder to
quantize robustly than the fixed weight tensors.

## Trade-offs

Below roughly 4-bit precision, accuracy degradation becomes more
pronounced and task-dependent — some tasks (open-ended generation) degrade
gracefully, while tasks requiring precise numerical or logical reasoning
tend to be more sensitive to aggressive quantization. Choosing a
quantization scheme therefore requires evaluating the target model on
representative downstream tasks at the intended bit-width rather than
relying on generic benchmark numbers alone.
