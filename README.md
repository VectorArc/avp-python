# Agent Vector Protocol (AVP) — KV-Cache Transfer for Multi-Agent LLMs

[![PyPI](https://img.shields.io/pypi/v/avp.svg)](https://pypi.org/project/avp/)
[![CI](https://github.com/VectorArc/avp-python/actions/workflows/ci.yml/badge.svg)](https://github.com/VectorArc/avp-python/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Spec](https://img.shields.io/badge/spec-v0.2-blue.svg)](https://github.com/VectorArc/avp-spec)

**Transfer KV-cache between LLM agents instead of regenerating text. Same multi-agent pipeline, 51-78% fewer tokens, 1.5-5x faster. Cross-model projection with zero training.**

```bash
pip install avp
```

## Who This Is For

> **Self-hosted models on GPUs.** AVP needs access to model internals (KV-cache, hidden states) that cloud APIs don't expose. If you use OpenAI, Anthropic, or Google APIs — AVP can't help you.

**Good fit:** Multi-agent pipelines on self-hosted models (vLLM, HuggingFace Transformers) with datacenter or same-machine connectivity.

**Not a fit:** Cloud API models, single-agent apps, edge/mobile, cross-internet links (<1 Gbps).

## Install

**Requires:** Python 3.9+, PyTorch >= 2.0. For vLLM integration: vLLM >= 0.15.

```bash
# Core SDK (codec, handshake, session, fallback)
pip install avp

# With latent communication (realignment, KV-cache, HuggingFace connector)
pip install "avp[latent]"

# With HTTP/2 transport server
pip install "avp[server]"

# Everything including dev tools
pip install "avp[all]"
```

**From source:**

```bash
git clone https://github.com/VectorArc/avp-python.git
cd avp-python
pip install -e ".[all]"
```

## Quick Start

**Layer 0 — JSON messaging (no GPU, no model download):**

```python
import avp

msg = avp.pack("Analyze this math problem: 24 * 17 + 3")
text = avp.unpack(msg)  # → "Analyze this math problem: 24 * 17 + 3"

wire = msg.to_bytes()   # send over any transport
```

**Layer 1 — Add model identity (downloads config only, ~1 KB):**

```python
msg = avp.pack("Analyze this", model="Qwen/Qwen2.5-7B-Instruct")
# msg.identity contains model family, dimensions, hash
# Receiving agent can check compatibility before GPU work
```

**Layer 2 — Latent transfer (requires GPU + model):**

```python
msg = avp.pack("Analyze this math problem: 24 * 17 + 3",
               model="Qwen/Qwen2.5-1.5B-Instruct", think_steps=20)
answer = avp.unpack(msg, model="Qwen/Qwen2.5-1.5B-Instruct")
```

## Key Results

| Metric | Value |
|--------|-------|
| Token savings vs text chains | **51-78%** across 4 benchmarks, 5 models |
| Speed improvement | **1.5-5x** faster (model and task dependent) |
| Cross-model (zero training) | **72%** GSM8K accuracy, Qwen 7B to Llama 3B, **6 KB** wire |
| Models validated | Qwen2.5 (1.5B, 7B), DeepSeek-R1 (1.5B), Llama 3.2 (1B, 3B) |
| Hardware | A100 (cloud), RTX 3070 Ti (local) |

> **Same-model:** Latent matches direct accuracy on Qwen 7B (85%) and beats text by 10pp. **Cross-model:** Zero-training vocabulary projection hits solver ceiling on structured tasks (math: 72%), but fails on comprehension (HotpotQA: 8%). See [full results](docs/BENCHMARKS.md).

## How It Works

```mermaid
graph LR
    subgraph text["Text Chain (today)"]
        direction LR
        A1["Agent A<br/>generates text"] -->|"serialize to text<br/>re-tokenize everything"| B1["Agent B<br/>re-processes from scratch"]
    end

    subgraph avp["AVP Latent Transfer"]
        direction LR
        A2["Agent A<br/>generates KV-cache"] -->|"binary transfer<br/>28-130 MB"| B2["Agent B<br/>picks up where A left off"]
    end

    style text fill:#fff3f3,stroke:#d44,stroke-width:2px
    style avp fill:#f3fff3,stroke:#4a4,stroke-width:2px
```

Every multi-agent framework today — LangChain, CrewAI, AutoGen, OpenAI Swarm — copies text between agents. Each agent re-tokenizes and re-processes everything prior agents already computed. Our benchmarks show **47-53% of all tokens in text chains are redundant re-processing**. (See [Works With](#works-with) for integration examples.)

AVP eliminates this by transferring the KV-cache (the computed attention states) directly. The receiving agent reads prior reasoning from attention states instead of re-computing it from text.

AVP defines a binary format, handshake, and codec — not the transport. It works alongside any agent framework or protocol.

```
┌──────────────────────────────────────────────────────────────┐
│  Your Orchestrator (LangGraph / CrewAI / PydanticAI / any)    │
│                                                              │
│  Agent A                          Agent B                    │
│    │                                ▲                        │
│    │  connector.think() ──►         │  connector.generate()  │
│    │  AVPContext                     │  with context=...      │
│    │                                │                        │
│    │    context.to_bytes()          │  AVPContext.from_bytes()│
│    ▼                                │                        │
│  ┌────────────────────────────────────────────┐              │
│  │  AVP (this library)                        │              │
│  │  • Handshake — resolves LATENT/JSON mode   │              │
│  │  • Codec — serialize/deserialize KV-cache  │              │
│  │  • Session — TTL, thread safety            │              │
│  └────────────────────────────────────────────┘              │
│         │                                                    │
│    Transport: HTTP/2, gRPC, shared memory, file, any         │
└──────────────────────────────────────────────────────────────┘
```

**Three communication modes, auto-negotiated via handshake:**

| Mode | When | What Happens |
|------|------|--------------|
| **Latent** | Same model | KV-cache + hidden state transfer, zero re-processing |
| **Cross-model** | Same or different family (e.g. Qwen 7B to Llama 3B) | Vocabulary-mediated projection, zero training needed |
| **JSON fallback** | No compatible projection path | Standard text, auto-negotiated |

**Transport-agnostic:** HTTP/2 (reference), gRPC, A2A, MCP, WebSockets, shared memory. AVP handles the latent communication layer — not discovery, routing, or orchestration.

## Connector API

For full control over model loading, device placement, and context serialization:

**High-level API (5 lines):**

```python
from avp import HuggingFaceConnector

connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Agent A: latent reasoning (no text output, builds KV-cache)
context = connector.think("Analyze this math problem: 24 * 17 + 3", steps=20)

# Agent B: generate with Agent A's context
answer = connector.generate("Now compute the final answer.", context=context)
```

**Cross-process transfer:**

```python
# Process A: serialize context
wire_bytes = context.to_bytes(session_id="s1", source_agent_id="agent-a")

# Process B: restore and generate
from avp import AVPContext, HuggingFaceConnector
connector = HuggingFaceConnector.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
restored = AVPContext.from_bytes(wire_bytes, device="cuda")
answer = connector.generate("Solve it.", context=restored)
```

**Check model compatibility:**

```python
from avp import extract_model_identity, CompatibilityResolver

local = extract_model_identity(model_a)
remote = extract_model_identity(model_b)
session = CompatibilityResolver.resolve(local, remote)
# session.mode → LATENT (same model) or JSON (different)
```

## Production: vLLM Integration

vLLM can't expose per-step hidden states, so latent transfer happens at the engine level via a KV connector plugin — transparent to your application code:

```bash
# Launch vLLM with AVP KV connector
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --kv-connector AVPKVConnectorV1Dynamic \
    --kv-connector-module-path avp.connectors.vllm_kv_connector
```

```python
# Application code stays simple — KV transfer happens behind the scenes
from avp import VLLMConnector

connector = VLLMConnector(model_id="Qwen/Qwen2.5-7B-Instruct")
answer = connector.generate("Analyze and solve: 24 * 17 + 3")
```

The `AVPKVConnectorV1Dynamic` plugin saves/loads KV-cache between vLLM instances via a file-based store, so agents on the same machine share computed attention states without re-processing.

## API Reference

### Easy API (start here)

| Import | What It Does |
|--------|-------------|
| `pack(content, *, model=, think_steps=)` | Pack text for transfer. Layer 0: JSON. Layer 1: + model identity. Layer 2: + latent context. Returns `PackedMessage`. |
| `unpack(data, *, model=)` | Unpack any AVP format to text. With `model=`, generates a response using latent context. |
| `PackedMessage` | Result of `pack()`. `str(msg)` for text, `msg.to_bytes()` for wire format, `.identity` for model info, `.context` for latent data. |

### Connector API (advanced)

| Import | What It Does |
|--------|-------------|
| `HuggingFaceConnector` | Main connector. `think()` builds KV-cache (returns `AVPContext`), `generate()` produces text. `from_pretrained()` for easy setup. |
| `VLLMConnector` | Production connector. `generate()` returns text. Latent transfer happens at engine level via KV connector plugin. |
| `AVPContext` | Wraps KV-cache + model metadata. Pass between `think()` and `generate()`, or serialize with `to_bytes()` / `from_bytes()` for cross-process transfer. |

### Protocol Layer

| Import | What It Does |
|--------|-------------|
| `encode` / `decode` | Binary codec for hidden states, KV-cache, and hybrid payloads. |
| `extract_model_identity` | Extract `ModelIdentity` (family, dimensions, hash) from a HuggingFace model. |
| `CompatibilityResolver.resolve()` | Handshake: compares two `ModelIdentity` objects, returns LATENT, HYBRID, or JSON mode. |
| `SessionManager` | Manage communication sessions with TTL and thread safety. |
| `AVPClient` / `AVPAsyncClient` | HTTP/2 client (sync and async) for sending AVP messages over the network. |
| `create_app` | Create a FastAPI server that receives AVP messages. |

### Cross-Model (Rosetta Stone)

| Import | What It Does |
|--------|-------------|
| `calibrate` | Build a projection map (`AVPMap`) between two models for cross-model transfer. Auto-detects same-family (vocab mediated) vs cross-family (vocab overlap). |
| `vocabulary_mediated_projection` | Project hidden states via shared vocabulary (same tokenizer). |
| `vocab_overlap_projection` | Project hidden states via overlapping BPE tokens (different tokenizers). |
| `validate_projection` | Quality gate: cosine similarity (fast) + pseudo-perplexity (thorough). Returns LATENT/HYBRID/JSON recommendation. |
| `save_map` / `load_map` / `find_map` | Persist and retrieve `.avp-map` files for reuse. |

### Error Types

All errors inherit from `AVPError`. Key types: `IncompatibleModelsError`, `HandshakeError`, `DecodeError`, `ShapeMismatchError`, `RealignmentError`, `SessionExpiredError`, `EngineNotAvailableError`, `FallbackRequested`.

## Roadmap

- Multi-embedding cross-model transfer (addressing single-embedding bottleneck on comprehension tasks)
- Compact hidden state mode (same-model, ~60x smaller wire than full KV-cache)
- CacheGen-style compression (3-4x KV-cache wire size reduction)
- vLLM serving throughput benchmarks

## Works With

### Agent Frameworks

AVP works *with* your orchestration framework, not instead of it. Your framework handles routing, state, and agent lifecycle. AVP handles the communication primitive between agents.

The integration pattern is the same across all six tested frameworks:

```python
# Agent A's output → AVP → Agent B's input
packed = avp.pack(agent_a_output, model="Qwen/Qwen2.5-7B-Instruct")
agent_b_input = str(packed)  # works as plain text in any framework
```

| Framework | Layer 0/1 Friction | Integration Point |
|-----------|-------------------|-------------------|
| **PydanticAI** | Zero — plain strings | `FunctionModel` callback |
| **LangGraph** | Low — wrap in `AIMessage` | Graph node function |
| **CrewAI** | Zero — plain strings | `BaseLLM.call()` override |
| **LlamaIndex** | Zero — plain strings | `CustomLLM.complete()` override |
| **OpenAI Agents SDK** | Low — custom `Model` class | `Model.get_response()` override |
| **Google ADK** | Low — async generator | `BaseLlm.generate_content_async()` override |

> **Layer 2 (latent transfer)** works in all six but requires a side-channel `dict` for KV-cache context — no framework natively supports binary tensor data between agents.

### Infrastructure & Protocols

- **[vLLM](https://github.com/vllm-project/vllm)** — KVConnectorBase_V1 plugin for production serving
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — Full hidden state and KV-cache access
- **[A2A](https://github.com/google/A2A)** — Transport binding via `multipart/related` with binary payloads
- **[MCP](https://github.com/modelcontextprotocol)** — Complementary: MCP handles tools and context, AVP handles tensor transfer

## Key Concepts

| Term | What It Means |
|------|---------------|
| **KV-cache** | During text generation, each transformer layer computes key and value vectors for the attention mechanism. These are cached so they don't need to be recomputed for each new token. AVP transfers this cache between agents so the receiving agent doesn't recompute what the sender already processed. |
| **Hidden states** | The internal vector representations at each transformer layer — the model's "understanding" of the input at that point in the network. Richer than text because they carry information that gets lost when converting to tokens. |
| **Latent transfer** | Sending KV-cache or hidden states (the "latent" internal representations) instead of converting to text and back. Avoids the lossy text bottleneck. |
| **Realignment** | Normalizing hidden states before injecting them into another model instance, so they match the expected input distribution. Required because hidden state magnitudes can drift. |
| **Tied weights** | When a model reuses the same weight matrix for both input embeddings and output projection (common in smaller models like Qwen <=3B, Llama 3.2 <=3B). Requires a special softmax-based projection instead of simple normalization. |
| **Vocabulary-mediated projection** | Cross-model transfer technique: convert source hidden states to token probabilities using the source model's output head, then reconstruct target-compatible representations using the target model's input embeddings. Works across families — when tokenizers differ, AVP projects through overlapping vocabulary tokens (~85% overlap for Qwen/Llama). |
| **PagedAttention** | vLLM's memory management for KV-cache — stores cache in non-contiguous pages. AVP's `page_convert` module handles conversion between paged and contiguous formats. |

## Documentation

- **[AVP Specification](https://github.com/VectorArc/avp-spec)** — Binary format, handshake, transport, security, test vectors
- **[Benchmark Results](docs/BENCHMARKS.md)** — Full results: 4 benchmarks, 5 models, same-model + cross-model
- **[Examples](examples/)** — Quickstart, agent demo, mixed-model demo, pack/unpack
- **[Contributing](CONTRIBUTING.md)** — Dev setup, tests, code style

## Research Foundation

AVP builds on [LatentMAS: Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) (Gen-Verse, 2025), which demonstrated same-model latent communication via hidden state transfer and KV-cache sharing. AVP productionizes this into a transport-agnostic binary protocol with cross-model support, compression, and engine connectors.

## License

Apache 2.0 — see [LICENSE](LICENSE)
