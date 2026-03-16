# AVP Benchmarks

> **+14.1pp on code generation vs text (p=0.004) · 14-78% fewer tokens · 1.2-4x faster** – 6 benchmarks, 5 models, 2 families. Cross-model rosetta across 4 model pairs.

---

## Accuracy

Same-model latent transfer matches or improves accuracy on structured tasks. Tested on NVIDIA A100, n=100-500 per benchmark.

### Code Generation

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **HumanEval** (Qwen 7B, n=164) | 58.5% | **67.1%** | 53.0% |
| **HumanEval** (Llama 3B, n=164) | 50.6% | **54.3%** | 44.5% |

Latent vs text (Qwen 7B): p=0.004. Validated across 4 seeds at T=0.01 (70.0%±0.3% latent vs 57.6%±0.3% text). Replicated on Llama 3B.

### Math Reasoning

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **GSM8K** (Qwen 7B, n=200) | 91.0% | 90.5% | 87.0% |
| **GSM8K** (Llama 3B, n=200) | 74.5% | 76.0% | 79.0% |

### Bug Fixing

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **DebugBench** (Qwen 7B, n=100) | 50.0% | 51.0% | 49.0% |
| **DebugBench** (Llama 3B, n=100) | 31.0% | 30.0% | 31.0% |

### Comprehension

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **HotpotQA** (Qwen 7B, n=200) | 51.5% | 52.5% | 50.5% |

All modes within noise. Latent's advantage here is purely efficiency.

---

## Efficiency

Token savings are structural – pre-computed KV-cache replaces re-processed text. Savings hold across every benchmark, every model.

| Agents | Benchmark | Token Savings | Speedup |
|--------|-----------|---------------|---------|
| 2 | GSM8K, DebugBench | 46-56% | 1.5-3x |
| 2 | HumanEval | 14% | 1.2x |
| 3 | Fan-out | 56-60% | 1.5x |
| 4 | GSM8K chain | 73-78% | 2-4x |

HumanEval token savings are lower because prompts are short (~182 tokens avg) and the latent reviewer generates longer, more complete code solutions (+53% more output tokens).

Text prompts grow **O(n²)** with agent count. Latent stays **O(n)**.

---

## Cross-Model (Rosetta Stone) – Experimental

> **Experimental.** Cross-model projection requires `cross_model=True`. Accuracy varies by task type – works well on structured tasks (math, code), degrades on comprehension and bug fixing.

Different models communicate via vocabulary-mediated projection. Zero training – uses existing embedding matrices.

### Rosetta Accuracy

| Source → Target | GSM8K (n=200) | HumanEval (n=164) | DebugBench (n=100) |
|-----------------|---------------|-------------------|--------------------|
| Qwen 7B → Qwen 3B | 82.5% | 66.5% | – |
| Qwen 7B → Llama 3B | 77.0% | 47.0% | 34.0% |
| Llama 3B → Qwen 7B | 90.0% | 79.3% | 45.0% |
| Qwen 7B → Qwen 1.5B | 58.5% | 42.1% | 26.0% |

Target model solo baselines: Qwen 7B = 91.0% / 58.5% / 50.0%. Qwen 3B = 82.5% / 61.0%. Llama 3B = 76.0% / 50.6% / 31.0%. Qwen 1.5B = 62.0%.

Accuracy is bounded by the target model's own capability. Advisory quality gate included for prompts >300 tokens where projection degrades.

### Rosetta vs Text Cross-Model

| Direction | Benchmark | Rosetta | Text | Delta |
|-----------|-----------|---------|------|-------|
| Qwen 7B → Llama 3B | GSM8K | 77.0% | **86.5%** | text +9.5pp |
| Llama 3B → Qwen 7B | GSM8K | **90.0%** | 82.0% | rosetta +8.0pp |
| Qwen 7B → Llama 3B | HumanEval | 47.0% | **57.9%** | text +10.9pp |
| Llama 3B → Qwen 7B | HumanEval | **79.3%** | 61.6% | rosetta +17.7pp |
| Qwen 7B → Llama 3B | DebugBench | 34.0% | **44.0%** | text +10.0pp |
| Llama 3B → Qwen 7B | DebugBench | **45.0%** | 40.0% | rosetta +5.0pp |

Direction matters: rosetta beats text when the stronger model is the solver. Text wins when the weaker model is the solver.

---

## Competition Math

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **MATH** (Qwen 7B, n=500) | 67.8% | 66.8% | 66.6% |

All three modes are statistically identical (p=1.0 latent vs text). Earlier runs at 512 max tokens showed a false text advantage due to solver truncation – with proper token budget (2048), the gap disappears.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Latent steps | 20 (validated: 10 ≈ 20 > 40 > 80) |
| Temperature | 0.7 |
| Max new tokens | 512 (MATH: 2048) |
| Seed | 42 |
| Hardware | NVIDIA A100 80GB |

---

## Limitations

- **Self-hosted only** – requires direct KV-cache access (cloud APIs don't expose this)
- **Single-embedding bottleneck** – cross-model transfers one vector; fails on long comprehension tasks
- **Multi-hop coherence** – KV-cache across 4+ sequential hops loses signal
- **1B-7B models tested** – larger models may behave differently

---

## Reproduce

```bash
pip install "avp" datasets

python -m benchmarks.humaneval.run_humaneval \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode all --max-samples 50 --latent-steps 20 --seed 42
```

All benchmark code: [`benchmarks/`](https://github.com/VectorArc/avp-python/tree/main/benchmarks). Llama models require [HF access](https://huggingface.co/meta-llama) and `HF_TOKEN`.
