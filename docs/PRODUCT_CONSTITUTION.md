# AVP Product Constitution

**Version:** 1.0
**Date:** March 20, 2026
**Status:** Active -- every PR, feature, and benchmark gets evaluated against this document.

---

## Why This Document Exists

AVP is a multi-agent communication protocol. Not a single-model optimization tool, not an inference accelerator, not a KV-cache utility library. Every line of code in this project exists to make *communication between agents* faster, cheaper, or more accurate.

We wrote this after the vLLM integration drifted into single-model latent thinking (overwrite pattern) that didn't support multi-agent transfer. The code worked -- it even showed a +4pp accuracy gain on GSM8K. But it violated AVP's reason for existence. A feature that makes one model think better, without enabling Agent A to transfer that thinking to Agent B, belongs in a different project.

This document is the filter. When someone asks "should we build X?" -- check it against these values, claims, and anti-patterns first.

---

## 1. Core Values

### V1: Multi-Agent First

Every feature must serve communication between two or more agents. A capability that only benefits a single model in isolation is out of scope, regardless of how impressive the results are. If a feature *also* benefits single-model workflows, that's fine -- but the multi-agent use case must be the design driver.

**Test:** Remove the second agent from the scenario. If the feature still makes sense, it doesn't belong in AVP.

### V2: Zero Inter-Agent Text

AVP exists to eliminate the text serialization bottleneck between agents. Same-model agents share KV-cache. Cross-model agents share projected hidden states. The protocol should never require Agent A to decode to text and Agent B to re-encode it. When text is the only viable path (incompatible models, no projection available), AVP falls back to JSON explicitly -- it doesn't pretend the fallback is the product.

**Test:** Does this feature reduce, maintain, or increase the number of text tokens exchanged between agents? If it increases them, it's moving the wrong direction.

### V3: Measured Claims, Disclosed Tradeoffs

Every claim AVP makes is backed by a reproducible benchmark with sample sizes, p-values where applicable, and stated limitations. When something doesn't work (cross-model comprehension, latent step ceiling, solver ceiling), we say so in the same paragraph where we report what does work. We never announce honesty -- we just practice it.

**Test:** Can someone reproduce this claim with the instructions we provide? Have we stated the tradeoff in the same breath?

### V4: Transport and Framework Agnostic

AVP defines the binary format, handshake, and codec. It does not own the transport layer, the orchestration logic, or the agent framework. It works alongside LangGraph, CrewAI, A2A, MCP, gRPC, and anything else that can carry binary payloads. The framework sees text in, text out. AVP is the sidecar, never the chassis.

**Test:** Does this feature require AVP to become an orchestration layer, a serving framework, or a transport protocol? If yes, we're overstepping.

### V5: Reference Implementation Parity

The HuggingFace connector is the reference implementation with published benchmarks. Any new engine integration (vLLM, llama.cpp, future engines) must reproduce the same multi-agent communication patterns. A new connector that can't do `think()` on Agent A and `generate(context=...)` on Agent B hasn't shipped the thing that matters.

**Test:** Can this connector run the 2-agent GSM8K benchmark and produce results within noise of the HuggingFace reference? If not, it's not ready for the benchmark table.

---

## 2. Hard Claims (The Constitution)

These are the specific commitments AVP makes in its README, blog post, benchmarks doc, and core messaging. They are non-negotiable. If a change would violate any of these, it does not ship.

### HC1: Zero tokens between agents

**Claim:** In latent mode, zero text tokens are exchanged between agents. Agent A's computation transfers as KV-cache (same-model) or a projected hidden state (cross-model), not as serialized text.

**Measurement:** Count inter-agent text tokens in a 2-agent pipeline. Must be 0 for latent mode, ~6 KB for rosetta mode. Only JSON fallback mode uses text tokens.

**Current status:** MET. HuggingFace connector achieves this for same-model and cross-model. vLLM connector does not yet support latent transfer (text-only).

### HC2: 2-3x faster pipelines (end-to-end)

**Claim:** Latent pipelines run 2-3x faster than equivalent text chain pipelines, measured end-to-end (Agent A start to Agent B answer complete). Not 10x, not one sub-component. Amdahl's Law bounds us because Agent B's own decode is irreducible.

**Measurement:** Wall-clock time, end-to-end, on published benchmarks. GSM8K 2.0x, DebugBench 3.0x, HumanEval 1.2x, HotpotQA 5.8x. Tradeoff: 20 latent steps = 0.9s fixed cost (7B, A100). Breakeven at ~22 tokens of Agent A output.

**Current status:** MET on HuggingFace connector (A100). Not yet measured on vLLM.

### HC3: Same or better accuracy

**Claim:** Latent transfer produces same or better accuracy compared to text chain on every published same-model benchmark. HumanEval: +14.1pp (p=0.004, 4 seeds, 2 model families). GSM8K, DebugBench, MATH, HotpotQA: neutral (no statistically significant difference). No same-model benchmark shows degradation.

**Measurement:** Accuracy on published benchmarks with stated n, seed, temperature, and hardware. New engine integrations must reproduce within noise of HuggingFace reference numbers.

**Current status:** MET on HuggingFace connector. vLLM single-model "overwrite" showed +4pp GSM8K but used semantically wrong pattern (no inter-step information flow). Not valid for multi-agent claim.

### HC4: Cross-model with zero training

**Claim:** Cross-model transfer works between model families (Qwen, Llama) with zero learned parameters, using vocabulary-mediated projection. The projection code is ~100 lines. Wire payload is ~6 KB.

**Measurement:** Cross-model benchmark accuracy vs published numbers. Qwen 7B to Llama 3B: 77.0% GSM8K, 47.0% HumanEval. Llama 3B to Qwen 7B: 90.0% GSM8K, 79.3% HumanEval. Must also disclose: cross-model comprehension fails (HotpotQA 7.5% EM).

**Current status:** MET on HuggingFace connector.

### HC5: Transport-agnostic

**Claim:** AVP defines binary format, handshake, and codec -- not transport. Works with HTTP/2, A2A, MCP, gRPC, WebSockets, or anything that carries binary payloads.

**Measurement:** The spec and SDK must not hardcode a transport. `AVPContext.to_bytes()` / `from_bytes()` must work independently of any transport layer.

**Current status:** MET. Reference HTTP/2 transport exists; core protocol has no transport dependency.

### HC6: Handshake auto-negotiation

**Claim:** The handshake automatically selects the best communication mode: same-model (full KV-cache), cross-model (rosetta projection), or JSON fallback. No manual configuration of mode selection.

**Measurement:** `CompatibilityResolver` must correctly route through rules 1-6 (hash match, structural match, shared tokenizer, avp-map file, vocab overlap, JSON fallback) for any pair of models.

**Current status:** MET. 6-rule resolution chain implemented and tested.

### HC7: Token cost scales O(n) not O(n^2)

**Claim:** In text pipelines, each agent reads all previous agents' output, so token cost grows O(n^2) with agent count. Latent transfer stays O(n) because each agent receives a fixed-size payload.

**Measurement:** 4-agent GSM8K chain: 73-78% token savings. Verify token counts scale linearly with agent count in latent mode.

**Current status:** MET. 4-agent benchmark published.

---

## 3. Use Cases (Ranked by Priority)

### Tier 1: Must work today

These are the use cases we've published benchmarks for and committed to publicly.

| # | Use Case | Status | Evidence |
|---|----------|--------|----------|
| 1 | **2-agent same-model pipeline** -- Agent A thinks, Agent B generates from A's computation | SHIPPED | GSM8K, HumanEval, DebugBench, MATH, HotpotQA benchmarks |
| 2 | **2-agent cross-model pipeline** -- Agent A (model X) thinks, Agent B (model Y) generates via rosetta projection | SHIPPED (experimental) | Qwen-Llama cross-family benchmarks |
| 3 | **N-agent same-model chain** -- 3+ agents in sequence, each building on prior computation | SHIPPED | 4-agent GSM8K (73-78% token savings) |
| 4 | **Cross-process transfer** -- Agent A and Agent B in different processes, same machine | SHIPPED | `AVPContext.to_bytes()` / `from_bytes()` |

### Tier 2: Must work before v1.0

These are on the roadmap and referenced in public docs.

| # | Use Case | Status | Dependency |
|---|----------|--------|------------|
| 5 | **vLLM multi-agent latent transfer** -- Agent A thinks via vLLM, Agent B generates via vLLM, using AVP KV connector | IN PROGRESS | vLLM extend pattern, KV extraction from paged buffer |
| 6 | **Bidirectional latent communication** -- Agent A sends computation to B, B sends computation back to A | PLANNED | Tier 1/hidden state return or Tier 2/delta KV return |
| 7 | **Cross-node transfer** -- Agents on different machines, KV-cache serialized over network | PARTIAL | `to_bytes()`/`from_bytes()` works; no optimized transport yet |

### Tier 3: Valuable, not committed

| # | Use Case | Status | Notes |
|---|----------|--------|-------|
| 8 | **Fan-out multi-agent** -- one agent broadcasts computation to N downstream agents | BENCHMARKED | 3-agent fan-out benchmark exists |
| 9 | **llama.cpp / Ollama latent transfer** | RESEARCHED | llama.cpp has C primitives; Ollama blocks on no internal API |
| 10 | **Mixed-engine pipeline** -- Agent A on HuggingFace, Agent B on vLLM | NOT STARTED | Requires `AVPContext` interop between connectors |
| 11 | **CacheGen-style KV compression** -- 3-4x reduction for network transfer | NOT STARTED | zstd gives 1-7%; need specialized compression |

---

## 4. Anti-Patterns

Things AVP is NOT. If you find yourself building any of these, stop and reconsider.

### AP1: Single-model thinking without transfer

Building latent thinking capabilities that only benefit one model instance, with no mechanism for Agent B to receive the enriched computation. This is what happened with the vLLM overwrite pattern -- it made one model think better but the thinking couldn't flow to another agent.

**Why it's wrong:** It turns AVP into an inference optimization library. There are better tools for that (speculative decoding, prompt caching, continuous batching). AVP's value is inter-agent, not intra-model.

### AP2: Orchestration framework

Building agent routing, task decomposition, workflow graphs, or role assignment. LangGraph, CrewAI, AutoGen, and A2A own this layer. AVP is the wire between agents, not the brain that decides which agent runs.

**Why it's wrong:** Competing with orchestration frameworks fragments our focus and puts us in a market with well-funded incumbents. Our moat is the protocol layer, not the orchestration layer.

### AP3: Cloud API compatibility layer

Wrapping OpenAI, Anthropic, or Google endpoints to simulate latent transfer. These APIs don't expose KV-cache or hidden states. There is no way to make latent transfer work through them.

**Why it's wrong:** It would require faking the core mechanism (KV-cache transfer) with something that isn't it (text summarization, embeddings API, etc.). This violates HC1 (zero tokens between agents) and misleads users.

### AP4: General-purpose KV-cache library

Building KV-cache utilities (serialization, compression, caching, eviction) that aren't specifically serving multi-agent transfer. If the tool is useful to someone running a single model with no agents, it's probably out of scope.

**Why it's wrong:** There are existing KV-cache management tools in vLLM, SGLang, and inference frameworks. We build the protocol for transferring KV-cache between agents, not the cache itself.

### AP5: Accuracy amplification claims

Claiming that AVP makes models smarter. The HumanEval result (+14.1pp) is real and reproducible, but the mechanism is "avoids text-chain degradation," not "amplifies reasoning." The solver ceiling is real -- a 7B researcher doesn't make a 3B solver smarter.

**Why it's wrong:** Overpromising accuracy gains sets up users for disappointment. Most benchmarks are neutral. The value proposition is efficiency (2-3x faster, O(n) scaling) with accuracy preservation, plus a genuine gain on code generation.

### AP6: Benchmarks without multi-agent communication

Running benchmarks that test single-model sequential generation, not inter-agent communication. This is why ClassEval was removed (March 12) -- it measured one model generating a class sequentially, not agents collaborating.

**Why it's wrong:** Benchmark results are commitments. Publishing a number that doesn't reflect multi-agent communication dilutes the benchmark suite and misleads users about what AVP does.

---

## 5. Decision Framework

Before building any feature, answer these questions in order. A "no" at any step is a hard stop.

### Gate 1: Multi-Agent Relevance

> **Does this feature enable, improve, or support communication between two or more agents?**

If no: Don't build it. It doesn't matter how technically impressive it is.

If yes: Proceed.

### Gate 2: Hard Claim Compliance

> **Does this feature violate any of the 7 hard claims (HC1-HC7)?**

Check each:
- HC1: Does it introduce text tokens between agents where there were none?
- HC2: Does it slow down the end-to-end pipeline?
- HC3: Does it degrade accuracy on any published benchmark?
- HC4: Does it require training for cross-model transfer?
- HC5: Does it hardcode a transport?
- HC6: Does it bypass or break auto-negotiation?
- HC7: Does it change token cost scaling?

If any violation: Don't ship until the violation is resolved.

If no violations: Proceed.

### Gate 3: Reference Implementation Parity

> **If this is a new connector or engine integration: can it run the 2-agent benchmark suite and match HuggingFace reference numbers within noise?**

If no: It's not ready for the benchmark table or public docs. Ship as "experimental" with clear disclaimers.

If yes (or not applicable): Proceed.

### Gate 4: Tradeoff Disclosure

> **Have we identified and documented the tradeoffs this feature introduces?**

Every feature has costs -- latency, memory, complexity, compatibility constraints. If we can't articulate the tradeoff, we don't understand the feature well enough to ship it.

If no: Document the tradeoffs first.

If yes: Proceed.

### Gate 5: Scope Check

> **Does this feature pull AVP toward any anti-pattern (AP1-AP6)?**

If yes: Redesign to avoid the anti-pattern, or don't build it.

If no: Build it.

---

## Applying This to the vLLM Integration

The vLLM integration that triggered this document serves as a worked example.

**What was built:** A model plugin (`AVPLatentQwen2ForCausalLM`) that runs latent thinking steps during prefill using an overwrite pattern. Single vLLM instance, single model, no transfer to a second agent.

**Gate 1 (Multi-Agent Relevance):** FAIL. Remove Agent B from the scenario and the feature still works. It's single-model latent thinking, not multi-agent communication.

**Gate 5 (Scope Check):** FAIL. This is AP1 (single-model thinking without transfer).

**What should have been built:** The latent thinking steps are valuable *if and only if* the enriched KV-cache can be transferred to Agent B. The correct vLLM integration path is:

1. Agent A submits a request. vLLM runs latent steps (extend pattern, not overwrite). The enriched KV-cache is saved to the KV store.
2. Agent B submits a separate request. vLLM loads Agent A's KV-cache from the store. Agent B generates from Agent A's computation.

The model plugin code isn't wasted -- the projection logic, registration, and profiling detection are all reusable. But the feature isn't complete until step 2 works, and it must be the extend pattern (each latent step sees all prior steps) to match the HuggingFace reference behavior.

---

## Maintaining This Document

This constitution is versioned. Changes require:
1. A clear rationale for why the change is needed
2. Verification that the change doesn't contradict published claims
3. Update of any affected hard claims, use cases, or anti-patterns

The hard claims (Section 2) can only be relaxed if:
- The claim was wrong (we reported incorrect data)
- The claim is being replaced by a strictly stronger claim
- The feature behind the claim is being deprecated with adequate notice

They cannot be relaxed because a new feature would be easier to build without them.
