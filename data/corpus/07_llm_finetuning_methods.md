# LLM Fine-Tuning Methods

Fine-tuning adapts a pretrained language model to a target task or domain
by continuing training on a smaller, task-specific dataset. Several
strategies trade off compute cost, memory footprint, and adaptation
quality.

## Full fine-tuning

Every parameter of the model is updated during training. This gives the
most flexibility to adapt model behavior but requires storing optimizer
states (for Adam-family optimizers, typically 2x the parameter count in
additional memory for momentum and variance terms) alongside gradients
and activations, meaning full fine-tuning of a large model can require
several times the GPU memory needed just to run inference on that model.
For a model with tens of billions of parameters, full fine-tuning
typically requires multiple high-memory GPUs working in parallel.

## Parameter-efficient fine-tuning (PEFT)

**LoRA (Low-Rank Adaptation)** freezes the pretrained weight matrices and
injects small trainable low-rank matrices alongside them; only these
low-rank matrices are updated during training. Because the number of
trainable parameters is a small fraction of the full model (often under
1%), LoRA dramatically reduces the memory needed for optimizer states and
gradients, and the resulting adapter weights are small enough to store and
swap independently of the base model — a single base model can serve
multiple LoRA adapters for different tasks or clients.

**QLoRA** combines LoRA with quantization of the frozen base model
(typically to 4-bit precision), further reducing the memory required to
even load the base model before training begins. This makes it possible
to fine-tune models with tens of billions of parameters on a single
consumer or workstation-class GPU, at some cost to numerical precision
during the forward and backward passes.

**Prefix/prompt tuning** prepends a small number of trainable "virtual
token" embeddings to the input and trains only those, leaving the entire
base model frozen. This is even more parameter-efficient than LoRA but
generally provides less adaptation capacity for complex tasks.

## Instruction tuning and RLHF

Instruction tuning fine-tunes a base (next-token-prediction) model on
examples of instructions paired with desired responses, converting a raw
language model into one that follows user directives. Reinforcement
Learning from Human Feedback (RLHF) further aligns model outputs with
human preferences by training a reward model on human-ranked outputs and
then optimizing the language model's policy against that reward model,
typically using an algorithm such as PPO or a simpler preference
optimization method such as DPO (Direct Preference Optimization), which
optimizes directly on preference pairs without training a separate
reward model.

## Choosing a fine-tuning strategy

Full fine-tuning is justified when the target domain is far from the
base model's training distribution and maximum adaptation quality is
required, and sufficient multi-GPU compute is available. PEFT methods
such as LoRA and QLoRA are preferred when compute or memory is
constrained, when multiple task-specific adapters need to be maintained
against a shared base model, or when fast iteration is more valuable than
squeezing out the last increment of task performance. In practice, LoRA
and QLoRA are the default choice for most on-premise or resource-
constrained fine-tuning workloads, since they make fine-tuning large
models feasible on hardware that could not otherwise support it.
