"""
Hand-built evaluation test set. Each case pairs a question with:
  - expected_sources: source file(s) that SHOULD be retrieved (docs tool only)
  - expected_tools: which of {docs, specs, web} the router SHOULD select
  - answer_must_contain: key facts/terms the generated answer should include
  - category: tags the retrieval/routing difficulty this case is probing

Categories:
  - single_hop: answer lives in one document, direct lexical + semantic match
  - multi_hop: answer requires synthesizing across 2+ documents
  - adversarial: query phrased to be lexically misleading (tests whether
    dense retrieval and the grading/correction loop compensate for BM25
    surface-level mismatch, or vice versa)
  - out_of_scope: question the corpus cannot answer; tests that the system
    says so rather than hallucinating
  - no_retrieval: conversational input that should skip all tools entirely
  - specs_lookup: requires the structured hardware specs database, not docs
  - web_required: needs live/current info outside docs and specs (requires
    TAVILY_API_KEY to actually execute; routing can still be checked without it)
  - multi_tool: requires combining two tools in a single query
  - memory_followup: a two-turn case testing whether chat_history is used
    correctly to resolve a follow-up question
"""

TEST_CASES = [
    {
        "id": "q1",
        "question": "What is the difference between a bi-encoder and a cross-encoder?",
        "expected_sources": ["04_embedding_models.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["bi-encoder", "cross-encoder"],
        "category": "single_hop",
    },
    {
        "id": "q2",
        "question": "How does Reciprocal Rank Fusion combine dense and sparse retrieval results?",
        "expected_sources": ["06_hybrid_retrieval.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["rank", "fusion"],
        "category": "single_hop",
    },
    {
        "id": "q3",
        "question": "What is QLoRA and how does it reduce memory requirements for fine-tuning?",
        "expected_sources": ["07_llm_finetuning_methods.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["qlora", "quant"],
        "category": "single_hop",
    },
    {
        "id": "q4",
        "question": "Why does continuous batching improve LLM serving throughput?",
        "expected_sources": ["09_inference_optimization.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["batch"],
        "category": "single_hop",
    },
    {
        "id": "q5",
        "question": "What is corrective RAG and when does it trigger a re-retrieval?",
        "expected_sources": ["02_agentic_rag_patterns.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["correct", "grad"],
        "category": "single_hop",
    },
    {
        "id": "q6",
        "question": (
            "How do quantization and KV-cache management both help with the "
            "memory-bandwidth bottleneck in LLM inference?"
        ),
        "expected_sources": ["08_quantization_techniques.md", "09_inference_optimization.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["bandwidth"],
        "category": "multi_hop",
    },
    {
        "id": "q7",
        "question": (
            "How does storage performance affect both LLM training (checkpointing) "
            "and inference (offloading) in on-premise deployments?"
        ),
        "expected_sources": [
            "10_on_prem_ai_infrastructure.md",
            "11_storage_for_ai_workloads.md",
            "09_inference_optimization.md",
        ],
        "expected_tools": ["docs"],
        "answer_must_contain": ["storage"],
        "category": "multi_hop",
    },
    {
        "id": "q8",
        "question": "What tokens does a self-reflective retrieval system emit to judge its own passages?",
        "expected_sources": ["02_agentic_rag_patterns.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["self-rag", "reflect"],
        "category": "adversarial",  # phrased around "self-reflective" not "Self-RAG"
    },
    {
        "id": "q9",
        "question": "What indexing algorithm builds a multi-layer navigable graph for nearest-neighbor search?",
        "expected_sources": ["03_vector_databases.md"],
        "expected_tools": ["docs"],
        "answer_must_contain": ["hnsw"],
        "category": "adversarial",  # describes HNSW without naming it
    },
    {
        "id": "q10",
        "question": "What's the capital of France?",
        "expected_sources": [],
        "expected_tools": [],
        "answer_must_contain": [],
        "category": "out_of_scope",
    },
    {
        "id": "q11",
        "question": "Does this corpus cover instructions for baking sourdough bread?",
        "expected_sources": [],
        "expected_tools": [],
        "answer_must_contain": [],
        "category": "out_of_scope",
    },
    {
        "id": "q12",
        "question": "Hey, how's it going?",
        "expected_sources": [],
        "expected_tools": [],
        "answer_must_contain": [],
        "category": "no_retrieval",
    },
    {
        "id": "q13",
        "question": "How much memory bandwidth does the H100 SXM have?",
        "expected_sources": [],
        "expected_tools": ["specs"],
        "answer_must_contain": ["3350", "3,350", "3.35"],
        "category": "specs_lookup",
    },
    {
        "id": "q14",
        "question": "Which enterprise SSD in the specs database has the highest write endurance (DWPD)?",
        "expected_sources": [],
        "expected_tools": ["specs"],
        "answer_must_contain": ["solidigm", "d7-p5810"],
        "category": "specs_lookup",
    },
    {
        "id": "q15",
        "question": "What Groq model pricing is currently listed on their website today?",
        "expected_sources": [],
        "expected_tools": ["web"],
        "answer_must_contain": [],  # live data, can't hardcode an expected answer
        "category": "web_required",
    },
    {
        "id": "q16",
        "question": (
            "Compare what our documents say about quantization's memory savings "
            "to the actual VRAM specs of the A100 versus H100 in the specs database."
        ),
        "expected_sources": ["08_quantization_techniques.md"],
        "expected_tools": ["docs", "specs"],
        "answer_must_contain": ["a100", "h100"],
        "category": "multi_tool",
    },
]

# Two-turn memory test, evaluated separately (needs sequential graph
# invocations sharing chat_history, not a single-shot check).
MEMORY_TEST_CASE = {
    "id": "mem1",
    "turn1_question": "What is the H100's memory bandwidth?",
    "turn2_question": "How does that compare to the A100?",
    # turn 2 should resolve "that" via chat_history and route to specs again,
    # answering with both models' bandwidth for comparison
    "turn2_expected_tools": ["specs"],
    "turn2_answer_must_contain": ["a100", "h100"],
}
