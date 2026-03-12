# AVP Benchmarks

> **+14.1pp on code generation vs text (p=0.004) · 14-78% fewer tokens · 1.2-4x faster** — 7 benchmarks, 5 models, 2 families.

---

## Accuracy

Same-model latent transfer matches or improves accuracy on structured tasks. Tested on NVIDIA A100, n=100-500 per benchmark.

### Code Generation

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **HumanEval** (Qwen 7B, n=164) | 58.5% | **67.1%** | 53.0% |

Latent vs text: p=0.004. Text chains introduce formatting noise that disrupts code structure.

### Math Reasoning

| | Direct | Latent (AVP) | Text |
|---|--------|--------------|------|
| **GSM8K** (Qwen 7B, n=200) | 91.0% | 90.5% | 87.0% |
| **GSM8K** (Llama 3B, n=50) | 76.0% | 76.0% | 74.0% |

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

Token savings are structural — pre-computed KV-cache replaces re-processed text. Savings hold across every benchmark, every model.

| Agents | Benchmark | Token Savings | Speedup |
|--------|-----------|---------------|---------|
| 2 | GSM8K, DebugBench, MATH | 46-56% | 1.5-3x |
| 2 | HumanEval | 14% | 1.2x |
| 3 | Fan-out | 56-60% | 1.5x |
| 4 | GSM8K chain | 73-78% | 2-4x |

HumanEval token savings are lower because prompts are short (~182 tokens avg) and the latent reviewer generates longer, more complete code solutions (+53% more output tokens).

Text prompts grow **O(n²)** with agent count. Latent stays **O(n)**.

---

## Cross-Model (Rosetta Stone) — Experimental

> **Experimental.** Cross-model projection requires `cross_model=True`. Accuracy varies by task type — works well on structured tasks (math, code), degrades on comprehension.

Different models communicate via vocabulary-mediated projection. Zero training — uses existing embedding matrices. Wire size: 3-7 KB.

| Source → Target | GSM8K (n=200) | HumanEval (n=164) |
|-----------------|---------------|-------------------|
| Qwen 7B → Llama 3B | **77.0%** | **47.0%** |
| Llama 3B → Qwen 7B | **90.0%** | **79.3%** |
| Qwen 7B → Qwen 1.5B | 58.5% | 42.1% |

Accuracy is bounded by the target model's own capability. Advisory quality gate included for prompts >300 tokens where projection degrades.

---

## Where Text Wins

Latent transfer doesn't help every task:

| | Direct | Latent (AVP) | Text | Why text wins |
|---|--------|--------------|------|---------------|
| **MATH** (Qwen 7B, n=500) | 43.2% | 45.0% | **59.4%** | Solver needs to read explicit step-by-step reasoning |

When the downstream agent needs the upstream agent's explicit reasoning chain (step-by-step math solutions, proof structures), text mode wins. The decision rule: if Agent B needs to *read* Agent A's output, use text mode.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Latent steps | 20 (validated: 10 ≈ 20 > 40 > 80) |
| Temperature | 0.7 |
| Max new tokens | 512 |
| Seed | 42 |
| Hardware | NVIDIA A100 80GB |

---

## Limitations

- **Self-hosted only** — requires direct KV-cache access (cloud APIs don't expose this)
- **Single-embedding bottleneck** — cross-model transfers one vector; fails on long comprehension tasks
- **Multi-hop coherence** — KV-cache across 4+ sequential hops loses signal
- **1B-7B models tested** — larger models may behave differently

---

## Reproduce

```bash
pip install "avp" datasets

python -m benchmarks.humaneval.run_humaneval \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --mode all --max-samples 50 --latent-steps 20 --seed 42
```

All benchmark code: [`benchmarks/`](https://github.com/VectorArc/avp-python/tree/main/benchmarks). Llama models require [HF access](https://huggingface.co/meta-llama) and `HF_TOKEN`.
