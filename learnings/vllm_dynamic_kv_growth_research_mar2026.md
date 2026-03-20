# Dynamic KV-Cache Growth in Production LLM Serving — Research Report

**Date**: March 20, 2026
**Scope**: How to implement the "extend" pattern (causal chain of N latent steps growing the KV-cache) inside vLLM, without forking vLLM.
**Verified against**: vLLM 0.17.x source code, Huginn vLLM plugin, COCONUT inference loop, syncdoth/latent-coprocessor, vLLM KV connector infrastructure.
**Status**: Complete. Six approaches assessed. One recommended.

---

## Executive Summary

The fundamental tension: AVP's HuggingFace `think()` grows the KV-cache by N entries (one per latent step) where each step causally attends to all prior steps. vLLM's scheduler pre-allocates a fixed number of blocks before `forward()` runs — the model cannot request additional blocks mid-forward. Of six approaches investigated, **sequential prompt_embeds requests with prefix caching** is the only path that (a) matches HF extend semantics exactly, (b) uses only stable vLLM APIs, (c) requires no vLLM fork or model plugin, and (d) survives vLLM version updates. The Huginn plugin provides a second viable path for in-model recurrence but with critical KV-cache compromises.

---

## Key Findings (Ranked by Importance)

### 1. [CRITICAL] Huginn's vLLM Plugin Uses OVERWRITE, Not Extend

**Finding**: The Huginn vLLM plugin (`raven_vllm.py`, 898 lines) implements its recurrent core loop **entirely within a single forward() call**, running N iterations of the recurrent block on the same set of positions. Each iteration overwrites the hidden states at the same KV-cache positions — it does NOT grow the KV-cache.

**Evidence**: Direct source code inspection (cloned `seal-rg/recurrent-pretraining`):

```python
# raven_vllm.py:375-412 (RavenModel.forward)
def forward(self, input_ids, positions, ...):
    input_embeds = self.embed_tokens(input_ids)
    # Prelude layers (positions fixed)
    for i in range(self.config.n_layers_in_prelude):
        input_embeds = self.layers[i](input_embeds, freqs_cis)

    # Recurrent core: N iterations, SAME positions, SAME KV slots
    hidden_states = self.initialize_state(input_embeds)
    for recurrent_step in range(self.config.mean_recurrence):
        hidden_states, _ = self.adapter(torch.cat([hidden_states, input_embeds], dim=-1))
        for i in range(self.config.n_layers_in_recurrent_block):
            hidden_states = self.layers[self.config.n_layers_in_prelude + i](hidden_states, freqs_cis)

    # Coda layers
    hidden_states = self.ln_f(hidden_states)
    for i in range(self.config.n_layers_in_coda):
        hidden_states = layer(hidden_states, freqs_cis)
    return self.ln_f(hidden_states)
```

**Key insight**: The recurrent core shares **only 4 layers** (indices 2-5) across all iterations. Prelude (layers 0-1) and coda (layers 6-7) are run once. The 4 recurrent layers' KV entries are overwritten 32 times, but each iteration sees the same positions. This is NOT the causal chain that AVP's HF extend pattern creates — it's a fundamentally different architecture designed for a trained recurrent model.

**The Huginn paper's "modulo cycling" claim**: The paper describes KV modulo cycling for decode, not for the recurrent core during prefill. During prefill, all positions are computed simultaneously — there is no sequential KV growth. The recurrence happens at the hidden-state level, not the KV-cache level.

**Implication for AVP**: Huginn's architecture cannot be directly adapted for AVP's extend pattern. AVP needs each latent step to ADD a new KV position (so step 10 can attend to steps 1-9). Huginn overwrites the same positions. The model plugin approach in AVP's `vllm_model_plugin.py` (overwrite pattern) is actually architecturally similar to Huginn, which explains why the overwrite gave +4pp on GSM8K — but it doesn't match the HF reference behavior.

**Evidence quality**: Battle-tested (direct source code). This finding invalidates the prior assessment that "Huginn proves extend works in vLLM."

---

### 2. [CRITICAL] Sequential Requests with prompt_embeds + Prefix Caching Is the Viable Extend Path

**Finding**: The only way to achieve true extend semantics (KV-cache grows by 1 entry per step, each step attends to all prior) in vLLM 0.17 without forking is to submit each latent step as a **separate request** where:

1. Step 0: Prefill with original prompt → extract hidden state → project to embedding
2. Step 1: Submit `prompt_embeds = [original_embeds + projected_step_0]` → prefix cache hits for original prompt, only 1 new position computed → extract hidden state
3. Step N: Submit `prompt_embeds = [original_embeds + projected_0 + ... + projected_{N-1}]` → prefix cache hits for all prior, 1 new position → extract hidden state
4. Final: Submit `prompt_embeds = [full_accumulated]` → generate text

Each step naturally allocates 1 new block position via the scheduler. Prefix caching (PR #27219, merged Oct 2025) ensures the model only computes the 1 new position — the prior L+k positions come from cache.

**Evidence**:
- PR #27219 demonstrates **8x speedup** (3,178 → 26,165 tokens/sec) with prefix caching on prompt_embeds (Llama 3.2-1B, A100)
- `tensor_data()` helper hashes embedding tensors as `memoryview` for cache matching
- Block-level hash comparison: identical embedding prefixes share cache blocks
- Scheduler's `allocate_slots()` naturally handles growing token counts

**Critical gap — hidden state extraction**: There is no merged API to extract hidden states from a vLLM generate call. RFC #18176 has been open for 10+ months. Workarounds:
- **Speculators VllmHiddenStatesGenerator**: Patches forward via WorkerExtension, but V0 only and prefill-only
- **Custom model plugin**: Override `forward()` to capture hidden states in a side channel
- **Post-generation re-prefill**: Run generation with max_tokens=1, then re-prefill the full sequence to extract hidden states (RFC #18176's proposed approach — doubles compute)
- **HiddenStatesProcessor (RFC #12249)**: Merged, but read-only and instance-level (not per-request)

**Estimated overhead per step**: ~2-5ms (scheduler round-trip + 1-token forward pass). For N=10: 20-50ms total, vs ~0.9s for full HF think loop on 7B.

**Engineering complexity**: Medium. Need to solve hidden state extraction, but all other pieces are stable APIs.

**Risk on vLLM updates**: LOW. Uses only `prompt_embeds` (stable since Sep 2025) and prefix caching (stable since Oct 2025). No model plugin, no monkey-patching, no internal API dependency.

**Evidence quality**: Well-established (individual pieces production-tested; combined pattern untested).

---

### 3. [HIGH] COCONUT's Extend Pattern — Direct Source Code Evidence

**Finding**: COCONUT (Facebook Research) implements the EXACT extend pattern AVP needs, with KV-cache growth at each latent step. The key insight from the source code:

```python
# coconut.py:62-103 — The latent loop
for pass_idx in range(max_n_latents):
    if kv_cache is None:
        # First pass: full prefill
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, 0:first_latent_pos, :],
            attention_mask=attention_mask[:, 0:first_latent_pos],
            position_ids=position_ids[:, 0:first_latent_pos],
            output_hidden_states=True,
        )
    else:
        # Subsequent passes: reuse KV, compute only NEW position
        past_key_values = [(k[:,:,:pos,:], v[:,:,:pos,:]) for k,v in kv_cache]
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, pos:pos+1, :],  # SINGLE new embedding
            attention_mask=attention_mask[:, :pos+1],       # Attend to ALL prior
            position_ids=position_ids[:, pos:pos+1],       # New position
            past_key_values=past_key_values,                # Reuse cached KV
            output_hidden_states=True,
        )

    # Feed hidden state back: replace next position's embedding
    hidden_states = outputs.hidden_states[-1]
    kv_cache = outputs.past_key_values
    inputs_embeds[batch_idx][next_latent_pos] = hidden_states[batch_idx, current_pos - offset, :]
```

**Key mechanisms**:
1. Each latent step processes EXACTLY 1 new position via `inputs_embeds[:, pos:pos+1, :]`
2. The KV-cache GROWS naturally — `past_key_values` from HuggingFace includes all prior steps
3. The hidden state at position P replaces the embedding at position P+1 (the next latent token)
4. `position_ids` are sequential — latent token at position P gets position_id P

This is directly analogous to what AVP's sequential prompt_embeds approach would achieve in vLLM. The difference: COCONUT uses HuggingFace's `past_key_values` for KV reuse, while the vLLM approach uses prefix caching for the same purpose.

**Evidence quality**: Battle-tested (published ICML paper, official Facebook implementation).

---

### 4. [HIGH] The Async KV Path (WAITING_FOR_REMOTE_KVS) Cannot Solve Dynamic Growth

**Finding**: The vLLM async KV path was designed for disaggregated prefill/decode (P/D) — one engine computes KV, another engine loads it for generation. It is NOT a mechanism for dynamic KV growth within a single request.

**How it actually works**:
1. Scheduler calls `connector.get_num_new_matched_tokens(request, num_computed_tokens)` → returns `(N, True)` where N is the number of tokens available externally and True means async
2. Request enters `WAITING_FOR_REMOTE_KVS` state — scheduler skips it
3. KV data arrives asynchronously (via RDMA/NIXL/shared storage)
4. `_update_waiting_for_remote_kv()` promotes request back to scheduling with `num_computed_tokens` set to the external token count
5. Scheduler allocates blocks and schedules the request for remaining tokens

**Why it can't help AVP**:
- The number of external tokens must be known UP FRONT when `get_num_new_matched_tokens` is called
- There is no mechanism to say "this request needs N MORE tokens than it currently has"
- The async path is for loading PRE-COMPUTED KV from another source, not for computing new KV iteratively
- Once a request exits WAITING_FOR_REMOTE_KVS, it proceeds normally — no re-entry

**Creative hack considered**: Could we use the connector to claim N+10 external tokens (original prompt + 10 latent), then compute the 10 latent KV entries in `start_load_kv()` using the model's forward pass? **No** — `start_load_kv()` runs in the worker but doesn't have access to the model's forward pass in a way that produces proper attention (it only has the KV buffer, not the model). And the KV entries must be computed causally (each seeing all prior), which requires N sequential forward passes that can't run inside a single `start_load_kv()` call.

**Evidence quality**: Battle-tested (confirmed via scheduler source code and NIXL connector implementation).

---

### 5. [HIGH] Speculative Decoding's Extra Slots Cannot Be Repurposed

**Finding**: EAGLE speculative decoding pre-allocates extra KV-cache slots via `extra_slots_per_request`. However, this mechanism is tightly coupled to the spec-decode pipeline.

**How EAGLE allocates extra slots**:
```python
# eagle.py
self.extra_slots_per_request = (
    1 if not self.parallel_drafting else self.num_speculative_tokens
)
```

The scheduler allocates `num_lookahead_tokens` extra blocks when spec decode is active. These slots are used by the EAGLE proposer for draft tokens, then either accepted (kept) or rejected (overwritten by target model).

**Why it can't help AVP**:
- `extra_slots_per_request` is set globally, not per-request
- The extra slots are managed by the spec decode framework's propose-verify cycle
- `prompt_embeds` is EXPLICITLY INCOMPATIBLE with speculative decoding (RFC #22124): "Speculative decoding should be forced to be disabled when prompt embeds is enabled"
- Even if we could allocate extra slots, we'd need to fill them with causally-computed KV entries, which requires N sequential forward passes — the same problem as before

**PADDING_SLOT_ID pattern**: EAGLE uses `PADDING_SLOT_ID` for positions beyond max_model_len. This is a masking mechanism, not a growth mechanism.

**Evidence quality**: Battle-tested (direct source code).

---

### 6. [MEDIUM] Meaningful Placeholder Embeddings — What the Literature Says

**Finding**: The Pause Tokens paper (arXiv 2310.02226) tested learnable pause tokens appended to prompts. Key findings relevant to AVP:

- Pause token embeddings are **learned during fine-tuning** — there is no "neutral" embedding that works out of the box
- Using periods `'...'` as filler produced **no gains** — confirming that arbitrary text tokens don't provide useful compute
- **Appending is better than prepending** pause tokens
- There is an **optimal count** for each task — too many pause tokens degrade performance
- The paper does NOT test zero embeddings, mean embeddings, or BOS embeddings as placeholders

**AVP's `<|endoftext|>` experience**: The dummy padding approach (using `<|endoftext|>` tokens to pre-allocate positions) corrupted output. This is expected — `<|endoftext|>` is a strong signal that generation should stop, and the model's attention to those KV entries actively harms generation.

**What would make a "neutral" placeholder**:
- **Zero vector**: Would produce zero attention scores (pre-softmax), effectively masked out. But the KV entries from a zero-input forward pass are NOT zero — the model's bias terms and layer norms produce non-trivial outputs. Untested in practice.
- **Mean embedding**: The mean of all token embeddings. In theory, a "maximally uninformative" input. No published results.
- **BOS embedding**: Already in the sequence. Adding copies would create positional confusion.
- **Random noise at embedding scale**: Some evidence from Huginn's `initialize_state()` which uses `trunc_normal_(std=sqrt(2/5*d))` — but this is for a model trained to expect random initialization.

**Bottom line**: There is no known "neutral" embedding that, when processed by a standard transformer, produces KV entries that are harmless to subsequent generation. The entire premise of placeholder tokens for pre-allocation is flawed for standard (non-pause-trained) models.

**Evidence quality**: Well-established (Pause Tokens paper) + Anecdotal (AVP's `<|endoftext|>` failure).

---

### 7. [MEDIUM] Micro-Requests (N+1 Shared-KV Requests) — Theoretically Sound, Practically Blocked

**Finding**: The idea of running each latent step as a separate "micro-request" that shares KV blocks with the main request is architecturally clean but blocked by vLLM's design.

**How it would work in theory**:
1. Main request: prefill prompt → P blocks allocated, KV computed
2. Micro-request 1: shares P blocks + 1 new block. Forward pass with projected embedding. Produces 1 new KV entry.
3. Micro-request 2: shares P+1 blocks + 1 new. Forward with projected_step_1. Produces 1 new KV.
4. After N micro-requests: main request has P+N blocks of KV. Generate from this.

**Why it's blocked**:
- **No block sharing API**: vLLM's block allocator assigns blocks per-request. There is no mechanism for request B to reference request A's blocks. The KV connector can INJECT data into allocated blocks, but can't share physical blocks.
- **Prefix caching is the equivalent**: When micro-request 2 submits the same prefix as micro-request 1, prefix caching automatically reuses the KV — but through hash matching, not explicit sharing.
- **Result ownership**: After a request finishes, its blocks are returned to the pool (unless prefix-cached). There's no mechanism to "transfer" blocks from one request to another.

**Insight**: This approach collapses into the "sequential requests with prefix caching" approach (Finding #2). The prefix caching mechanism IS the block-sharing mechanism — just implemented through content-addressable hashing rather than explicit references.

**Evidence quality**: Well-established (vLLM block allocator source code).

---

## Anti-Patterns & Documented Failures

### Anti-Pattern 1: Dummy Token Pre-Allocation

**What was tried**: Pre-pad the prompt with N dummy tokens (`<|endoftext|>`, zeros, or pad tokens) to force the scheduler to allocate N extra blocks, then overwrite the KV entries during the latent loop.

**Why it failed for AVP**: `<|endoftext|>` tokens inject strong "stop" signals into the KV-cache. Even when the KV entries at those positions are overwritten during latent steps, the model has already computed attention to the original dummy entries during prefill, and those attention patterns propagate through residual connections. The output degrades to repetitive garbage ("This is a great way to get in a lot of the most of the most...").

**How to avoid**: Don't use semantically meaningful tokens as placeholders. If pre-allocation is needed, the placeholder must be processed by the model to produce genuinely neutral KV entries — and no such placeholder exists for standard pre-trained models (see Finding #6).

### Anti-Pattern 2: Assuming Huginn's Architecture Applies to Standard Models

**What was tried**: Using Huginn as a template for "latent thinking in vLLM works."

**Why it's misleading**: Huginn is a purpose-trained depth-recurrent model. Its recurrent core (4 shared layers) is designed to refine hidden states through iteration. Standard transformer models (Qwen, Llama) have unique per-layer weights and were never trained for recurrent execution. The fact that Huginn's vLLM plugin works proves that vLLM CAN handle recurrent computation — but it does NOT prove that standard models produce useful computation through the same pattern.

**How to avoid**: Always distinguish between "the infrastructure supports X" and "standard models benefit from X." AVP's HF extend pattern works on standard models (90.5% GSM8K) not because of recurrence, but because each step's hidden-state → embedding → forward creates a genuinely new computation with fresh per-layer attention.

### Anti-Pattern 3: Trying to Modify Block Allocation from Inside forward()

**What was tried**: Various community attempts to allocate additional KV-cache blocks from within the model's `forward()` method.

**Why it fails**: vLLM's architecture is strictly separated: the scheduler allocates blocks BEFORE `forward()` runs. The model runner's `_prepare_inputs()` converts block allocations into slot mappings. By the time `forward()` executes, the slot mappings are finalized (and potentially compiled into CUDA graphs). There is no API to request more blocks from inside `forward()`.

**How to avoid**: Accept that KV-cache growth must happen between scheduler steps (i.e., between requests), not within a single forward pass. This is why the sequential request approach works — each request triggers a new scheduler step with a new block allocation.

### Anti-Pattern 4: Forward Hooks for Hidden State Extraction

**What was tried**: Using PyTorch `register_forward_hook()` to capture hidden states from vLLM models.

**Why it fails in production**: CUDA graphs (the default for decode) replay cached GPU operations. Hooks registered on Python modules are NOT replayed — they fire during graph recording but NOT during graph replay. The only workaround is `enforce_eager=True`, which costs 30-50% throughput.

**How to avoid**: Use vLLM's official hidden state extraction mechanisms (HiddenStatesProcessor, or custom model plugin with side channel). For AVP's sequential approach, the hidden state can be extracted by the custom model plugin during the 1-token forward pass (which is a prefill, not a decode — CUDA graphs may not apply).

---

## Detailed Assessment: The Sequential prompt_embeds Approach

### Architecture

```
Agent A's "think" (N=10):
  Request 0: prefill("Solve: 24*17+3") → KV[0..L], extract hidden[L]
  Request 1: prompt_embeds=[embed[0..L] + proj(hidden[L])] → prefix cache hit L, compute 1 new pos → KV[0..L+1], extract hidden[L+1]
  Request 2: prompt_embeds=[embed[0..L] + proj(hidden[L]) + proj(hidden[L+1])] → prefix cache hit L+1, compute 1 → KV[0..L+2]
  ...
  Request 10: → KV[0..L+10]
  Save KV[0..L+10] to store

Agent B's "generate":
  Request: same prompt, KV connector reports L+10 external tokens
  Scheduler allocates blocks, start_load_kv injects stored KV
  Agent B generates from Agent A's enriched KV
```

### What Matches HF Extend Semantics

| Property | HF extend | Sequential prompt_embeds |
|----------|-----------|------------------------|
| KV grows by 1 per step | Yes | Yes (new position each request) |
| Step N sees all prior steps | Yes (via past_key_values) | Yes (via prefix cache) |
| Causal attention | Yes | Yes (standard attention mask) |
| Hidden state from last layer | Yes | Needs extraction mechanism |
| Position IDs sequential | Yes | Yes (positions 0..L+N) |
| RoPE correct | Yes | Yes (positions assigned by scheduler) |

### Open Engineering Problems

1. **Hidden state extraction**: No merged API. Options ranked by viability:
   a. **Custom model plugin (minimal)**: Override `forward()` to store `hidden_states[-1][:, -1, :]` in a module-level dict. Only needed during think() requests, not generation. The latent step requests are prefill (1 new token), so CUDA graphs may not apply.
   b. **Post-generation re-prefill**: Generate with max_tokens=0 or 1, then re-prefill to extract hidden states. Doubles compute. Not viable for N=10.
   c. **Speculators**: V0 only. Not viable for vLLM 0.17+.

2. **Embedding extraction for first step**: Need the model's input embeddings for the original prompt to construct `prompt_embeds` for step 1. Options:
   a. Use vLLM's tokenizer + model's embedding table (accessible via model plugin)
   b. Pre-compute embeddings using a separate HF model load (unacceptable memory)
   c. Use vLLM's `embed_input_ids()` if available on the model class

3. **Latency**: N sequential scheduler round-trips add ~20-50ms for N=10. Acceptable — the HF reference takes ~900ms for the same.

4. **Prefix cache eviction**: Under memory pressure, prefix cache blocks can be evicted. This would cause step N+1 to recompute all prior positions. Mitigation: pin cache blocks (no API for this), or accept the performance regression.

5. **Multi-agent transfer**: The enriched KV (L+10 positions) must be saved to a KV store for Agent B. The existing KV connector infrastructure handles this — `save_kv_layer` already extracts per-request KV from the paged buffer.

### What to Keep from Current Implementation

- **KV connector** (`vllm_kv_connector.py`): FileKVStore, save/load/inject infrastructure — all reusable
- **Model plugin registration** (`vllm_model_plugin.py`): Entry point, plugin system — reusable, but the overwrite latent loop is replaced
- **Projection logic**: `_setup_projection_from_weights`, `_project_hidden_state` — reusable for the hidden-state → embedding projection step
- **Slot mapping**: `_compute_slot_mapping`, `_extract_request_kv`, `_inject_request_kv` — reusable for KV save/load

### What Changes

- **The latent loop moves OUTSIDE forward()**: Instead of N iterations inside a single forward(), there are N sequential vLLM requests
- **Model plugin becomes minimal**: Only needed for hidden state extraction, not for the latent loop itself
- **New orchestration layer**: A controller that sequences the N requests, extracts hidden states, projects embeddings, and constructs the next request's prompt_embeds

---

## Gaps & Uncertainties

1. **Prefix caching hash stability for growing embeddings**: PR #27219 hashes embeddings as raw bytes. If the embedding tensor for step N is constructed differently than for step N+1 (e.g., different memory layout, different floating point rounding), the prefix cache may miss. Needs empirical validation.

2. **CUDA graph behavior for 1-token prefill requests**: The sequential approach submits requests where only 1 new position is computed (prefix cache handles the rest). If vLLM treats this as a "prefill" (query_len=1 but first time seeing this sequence), CUDA graphs may not be used. If it treats it as extending a cached sequence, CUDA graphs may apply. The behavior for this edge case is undocumented.

3. **Hidden state extraction WITHOUT model plugin**: If we can solve this without a model plugin (e.g., RFC #18176 merges, or using the HiddenStatesProcessor with a per-request flag), the entire approach becomes plugin-free — using only stable public APIs. Worth monitoring.

4. **Prompt_embeds precision**: Open PR #37170 addresses "prompt_embeds precision divergence with MTP speculative decoding." If there are precision issues with prompt_embeds in general, this could affect the quality of projected embeddings. Needs monitoring.

5. **KVCacheSpec registry (RFC #36668)**: If this merges and allows custom KV cache layouts, a future approach could register a KV cache type that supports dynamic growth. This would be the clean architectural solution but is far from merging.

6. **Whether the sequential approach matches HF accuracy**: The causal chain structure is identical (each step sees all prior), but implementation differences (prefix caching vs HuggingFace past_key_values, numerical precision) could affect results. Must be validated with GSM8K n=200 benchmark.

---

## Recommendation

**Implement the sequential prompt_embeds approach** with a minimal model plugin for hidden state extraction.

**Phase 1** (~3 days):
1. Minimal model plugin: override `forward()` to store last hidden state in a module-level dict when a flag is set. No latent loop — just extraction.
2. Orchestration function: `vllm_think(prompt, model, steps=10)` that submits N sequential requests, each growing the prompt_embeds by 1 projected embedding.
3. Validate prefix caching works correctly with growing embeddings.

**Phase 2** (~2 days):
4. KV extraction: After the final latent step, extract the full KV (L+N positions) using the existing KV connector infrastructure.
5. KV injection: Agent B's request loads the enriched KV from the store.
6. End-to-end test: 2-agent GSM8K benchmark via vLLM.

**Phase 3** (~1 day):
7. Benchmark against HF reference: must match 90.5% GSM8K within noise.
8. Document the approach and update BENCHMARKS.md.

**Do NOT pursue**:
- Huginn-pattern recurrence inside forward() — wrong architecture for standard models' extend pattern
- Async KV path hacking — the API doesn't support dynamic growth
- Speculative decoding repurposing — explicitly incompatible with prompt_embeds
- Dummy token pre-allocation — empirically proven to corrupt output

---

## References

### Source Code Inspected
- `seal-rg/recurrent-pretraining/vllm/raven_vllm.py` — Huginn vLLM plugin (898 lines)
- `facebookresearch/coconut/coconut/coconut.py` — COCONUT inference loop (263 lines)
- `syncdoth/latent-coprocessor/train.py` — DeepMind Deliberation reproduction
- `vllm/v1/core/sched/scheduler.py` — vLLM scheduler, async KV path
- `vllm/v1/core/kv_cache_manager.py` — Block allocation, external tokens
- `vllm/v1/spec_decode/eagle.py` — EAGLE extra slot allocation
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — KVConnectorBase_V1
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` — NIXL connector (production)

### vLLM PRs and RFCs
- PR #27219: Prefix caching with prompt_embeds (merged Oct 2025, **8x speedup**)
- PR #24278: prompt_embeds V1 implementation (merged Sep 2025)
- RFC #22124: Prompt Embeddings V1 (merged)
- RFC #18176: Return hidden states via API (**OPEN**, 10+ months)
- RFC #36668: KVCacheSpec registry (open, Mar 2026)
- PR #37170: prompt_embeds precision fix (open, Mar 2026)

### Papers
- Huginn: "Scaling up Test-Time Compute with Latent Reasoning" (arXiv 2502.05171)
- COCONUT: Continuous Chain-of-Thought (Facebook Research)
- Pause Tokens: "Think Before You Speak" (arXiv 2310.02226)
- DeepMind Deliberation in Latent Space (arXiv 2412.17747)
- CacheBlend (EuroSys 2025 Best Paper)
