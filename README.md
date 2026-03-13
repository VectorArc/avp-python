# AVP – Agents Share Thoughts, Not Text

[![PyPI](https://img.shields.io/pypi/v/avp.svg)](https://pypi.org/project/avp/)
[![CI](https://github.com/VectorArc/avp-python/actions/workflows/ci.yml/badge.svg)](https://github.com/VectorArc/avp-python/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Spec](https://img.shields.io/badge/spec-v0.3-blue.svg)](https://github.com/VectorArc/avp-spec)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VectorArc/avp-python/blob/main/notebooks/avp_quick_start.ipynb)

When LLM agents hand off work as text, the next agent re-processes everything from scratch. AVP (Agent Vector Protocol) transfers the actual computation – KV-cache, hidden states, attention – so the receiving agent picks up where the sender left off. Zero tokens between agents, 2-3x faster pipelines, same or better accuracy. Built on [LatentMAS](https://arxiv.org/abs/2511.20639), extended with cross-model vocabulary-mediated projection (novel – zero training, works across model families).

```bash
pip install avp
```

> **Requires self-hosted models on GPUs.** AVP accesses model internals (KV-cache, hidden states) that cloud APIs don't expose. If you call OpenAI, Anthropic, or Google endpoints, AVP can't help. Good fit: multi-agent pipelines on HuggingFace Transformers with local or datacenter GPUs.

## Quick Start

```python
from avp import HuggingFaceConnector

connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Agent A thinks (builds KV-cache, no text output)
context = connector.think("Analyze this math problem: 24 * 17 + 3", steps=20)

# Agent B generates using Agent A's KV-cache
answer = connector.generate("Solve step by step: 24 * 17 + 3", context=context)
```

## Results

**Direct** = single model, no pipeline. **Latent** = AVP transfer. **Text Chain** = standard text handoff between agents.

| | Direct | Latent (AVP) | Text Chain |
|---|--------|--------------|------------|
| **HumanEval** (Qwen 7B, n=164) | 58.5% | **67.1%** | 53.0% |
| **GSM8K** (Qwen 7B, n=200) | 91.0% | 90.5% | 87.0% |
| **DebugBench** (Qwen 7B, n=100) | 50.0% | 51.0% | 49.0% |
| **GSM8K** (Llama 3B, n=200) | 74.5% | 76.0% | 79.0% |

HumanEval: +12.4pp vs text across 4 seeds (p=0.004). GSM8K and DebugBench: neutral across all modes, but the pipeline runs 3x faster (7.6s vs 22.8s end-to-end on DebugBench). Llama 3B: text wins on GSM8K – latent overhead has more impact on smaller models. All benchmarks used `steps=20` on NVIDIA A100.

**Trade-off:** 20 latent steps add ~0.9s fixed cost on A100. Breaks even when Agent A would otherwise produce ~22+ tokens of text.

**Cross-model (zero training, 6 KB on the wire):**

| Source | Target | GSM8K | HumanEval |
|--------|--------|-------|-----------|
| Qwen 7B | Llama 3B | 77.0% | 47.0% |
| Llama 3B | Qwen 7B | **90.0%** | **79.3%** |

Cross-model accuracy depends on the target – a weaker model's reasoning can push a stronger solver past its text-chain baseline (Llama 3B → Qwen 7B: 90.0% vs 87.0% text), but the reverse direction underperforms text. The projection is vocabulary-mediated – no learned parameters, no training data, works across model families.

Full results: **[Benchmarks](docs/BENCHMARKS.md)** – 7 benchmarks, 5 models, 2 families, reproducible.

## How It Works

```mermaid
graph LR
    subgraph text["Text Handoff"]
        direction LR
        A1["Agent A generates text"] -->|"serialize, re-tokenize"| B1["Agent B re-processes from scratch"]
    end

    subgraph avp["AVP Transfer"]
        direction LR
        A2["Agent A builds KV-cache"] -->|"binary transfer"| B2["Agent B continues from cached state"]
    end

    style text fill:#fff3f3,stroke:#d44,stroke-width:2px
    style avp fill:#f3fff3,stroke:#4a4,stroke-width:2px
```

Three modes, auto-negotiated via handshake:

| Mode | When | Payload |
|------|------|---------|
| **Latent** | Same model | Full KV-cache |
| **Cross-model** | Different model or family | Single projected hidden state (~6 KB) |
| **JSON fallback** | No compatible projection path | Plain text |

## Works With

Replace `llm.invoke()` with `avp.generate()`. Your framework sees text in, text out. All integrations use HuggingFace Transformers as the inference engine.

| Framework | Integration point |
|-----------|-------------------|
| **HuggingFace** | Full latent pipeline (KV-cache + hidden states) |
| **LangGraph** | Graph node replaces LLM call |
| **CrewAI** | `BaseLLM.call()` override |
| **PydanticAI** | `FunctionModel` callback |
| **LlamaIndex** | `CustomLLM.complete()` override |
| **A2A / MCP** | Complementary – AVP handles tensor transfer, they handle routing |
| **vLLM** | Text-only generation; latent transfer on roadmap |

See **[Framework Integration Guide](docs/FRAMEWORK_INTEGRATION.md)** for working examples.

<details>
<summary><strong>Cross-model transfer</strong></summary>

```python
from avp import HuggingFaceConnector

researcher = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
solver = HuggingFaceConnector.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Solve step by step: 24 * 17 + 3"
context = researcher.think(prompt, steps=20)
answer = solver.generate(prompt, context=context, source=researcher, cross_model=True)
```

> **Experimental.** Cross-model accuracy varies by task – works well on structured tasks (math, code), may degrade on comprehension. See [Benchmarks](docs/BENCHMARKS.md).

Calibration is automatic and one-time per model pair (~0.5-2s), cached to `~/.avp/maps/`.

</details>

<details>
<summary><strong>Easy API (one-liners)</strong></summary>

```python
import avp

# think + generate in one call
answer = avp.generate("Solve: 24 * 17 + 3", model="Qwen/Qwen2.5-7B-Instruct")

# cross-model (experimental)
answer = avp.generate("Solve: 24 * 17 + 3",
                       model="meta-llama/Llama-3.2-3B-Instruct",
                       source_model="Qwen/Qwen2.5-7B-Instruct",
                       cross_model=True)
```

</details>

<details>
<summary><strong>Cross-process transfer</strong></summary>

```python
# Process A: serialize
wire_bytes = context.to_bytes(session_id="s1", source_agent_id="agent-a")

# Process B: restore and generate
from avp import AVPContext, HuggingFaceConnector
connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
restored = AVPContext.from_bytes(wire_bytes, device="cuda")
answer = connector.generate(prompt, context=restored)
```

</details>

## Roadmap

- vLLM latent transfer
- Bidirectional latent communication (both agents share thinking, not just one)
- CacheGen-style KV-cache compression (3-4x reduction)

## Documentation

- **[AVP Specification](https://github.com/VectorArc/avp-spec)** – Binary format, handshake, transport
- **[Benchmarks](docs/BENCHMARKS.md)** – 7 benchmarks, 5 models, 2 families
- **[Framework Integration](docs/FRAMEWORK_INTEGRATION.md)** – LangGraph, CrewAI, PydanticAI, LlamaIndex
- **[Examples](examples/)** – Quickstart, cross-model, and agent demos
- **[CHANGELOG](CHANGELOG.md)**

## License

Apache 2.0 – see [LICENSE](LICENSE)
