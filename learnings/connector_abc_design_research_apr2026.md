# Connector / Provider ABC Design Research

**Date:** April 3, 2026
**Scope:** How 8 major ML/AI SDKs design their connector/provider/backend abstractions
**Relevance:** Informing AVP EngineConnector public API design

## Executive Summary

Across 8 frameworks (LiteLLM, LangChain, vLLM, HuggingFace Transformers, llama-cpp-python, OpenAI SDK, smolagents, PydanticAI), three patterns dominate: (1) the minimal abstract contract is always 1-3 methods, never more; (2) model metadata/capabilities are handled separately from the generation interface via a data object (not methods on the connector); (3) tokenization is universally treated as optional with fallbacks. AVP's current design (1 abstract method) is well-aligned with industry practice, but it lacks a structured metadata/capabilities layer that every mature framework provides.

---

## Framework-by-Framework Analysis

### 1. smolagents (HuggingFace) — Most Relevant Comparison

**Minimal contract:** 1 method
- `generate(messages, stop_sequences, response_format, tools_to_call_from, **kwargs) -> ChatMessage`

**Model metadata:**
- `model_id: str | None` — set in __init__, used for capability checks
- `supports_stop_parameter` — property that checks model compatibility
- No architecture properties (hidden_size, etc.) exposed at all

**Tokenization:** Handled by concrete classes (TransformersModel stores `self.tokenizer`, VLLMModel calls `get_tokenizer(model_id)`). Not part of ABC.

**Engine-specific features:** `cleanup()` on VLLMModel, `make_stopping_criteria()` on TransformersModel. Not abstracted.

**Hierarchy:** `Model` → `ApiModel` (adds rate limiting, retry, client) → `OpenAIModel`, `InferenceClientModel`, etc. Local models (`TransformersModel`, `VLLMModel`) extend `Model` directly.

**Key insight:** smolagents treats models as opaque text-in/text-out boxes. Zero architecture introspection. This works because smolagents only does agent orchestration — it never needs hidden_size or KV-cache access.

### 2. LangChain — Most Mature Abstraction

**Minimal contract:** 2 methods
- `_generate(messages, stop, run_manager, **kwargs) -> ChatResult` (abstract)
- `_llm_type` property (abstract)

**Optional overrides (with defaults):**
- `_stream()` — streaming (falls back to invoke)
- `_agenerate()` — async (falls back to running sync in executor)
- `_astream()` — async streaming
- `_resolve_model_profile()` — returns `ModelProfile | None`

**Model metadata via ModelProfile (TypedDict, total=False):**
```python
class ModelProfile(TypedDict, total=False):
    name: str; status: str; release_date: str
    max_input_tokens: int; max_output_tokens: int
    text_inputs: bool; image_inputs: bool; audio_inputs: bool
    tool_calling: bool; tool_choice: bool; structured_output: bool
    reasoning_output: bool; temperature: bool
    # ... ~25 total fields
```

**Tokenization:**
- `get_token_ids(text) -> list[int]` — on BaseLanguageModel, defaults to GPT-2 tokenizer
- `get_num_tokens(text) -> int` — delegates to get_token_ids
- `custom_get_token_ids: Callable | None` — injectable custom tokenizer
- Providers override for accuracy (ChatOpenAI uses tiktoken)

**Engine-specific:** `bind_tools()`, `with_structured_output()` — optional methods with `NotImplementedError` defaults. ChatOpenAI adds tiktoken-based counting, tool binding, structured output modes.

**Key insight:** The ModelProfile TypedDict pattern is LangChain's most important design. It separates "what can this model do" from "call this model." All fields optional. Provider packages populate it. The profile is a **data object**, not methods on the model.

### 3. LiteLLM — Most Comprehensive Provider Abstraction

**Minimal contract:** 6 abstract methods (heaviest of all frameworks)
- `get_supported_openai_params(model) -> list`
- `map_openai_params(non_default_params, optional_params, model, drop_params) -> dict`
- `validate_environment(headers, model, messages, ...) -> dict`
- `transform_request(model, messages, optional_params, ...) -> dict`
- `transform_response(model, raw_response, ...) -> ModelResponse`
- `get_error_class(error_message, status_code, headers) -> BaseLLMException`

**Model metadata via ModelInfo (TypedDict, Required fields + optional):**
```python
class ModelInfoBase(TypedDict, total=False):
    max_tokens: int; max_input_tokens: int; max_output_tokens: int
    input_cost_per_token: float; output_cost_per_token: float
    supports_system_messages: bool; supports_response_schema: bool
    supports_vision: bool; supports_function_calling: bool
    supports_prompt_caching: bool; supports_reasoning: bool
    litellm_provider: str; mode: str; tpm: int; rpm: int
    # ... 50+ fields including pricing, rate limits, capabilities
```

**Routing:** String-based model routing (`"openai/gpt-4"`, `"anthropic/claude-3"`) with `get_llm_provider()` parser.

**Tokenization:** `token_counter(model, text, messages) -> int` — top-level function, not on provider. Falls back when counter is disabled.

**Key insight:** LiteLLM's 6-method ABC is the **anti-pattern** — it's the most complained-about framework for provider implementation complexity. However, their model metadata approach (JSON file with 50+ fields per model, `get_model_info()` lookup) is the most comprehensive capability registry.

### 4. PydanticAI — Best Modern Design

**Minimal contract:** 3 items
- `request(messages, model_settings, model_request_parameters) -> ModelResponse` (async, abstract)
- `model_name` property (abstract)
- `system` property (abstract, returns provider name for OTEL)

**Optional overrides:**
- `request_stream()` — streaming (raises NotImplementedError)
- `count_tokens()` — token counting (raises NotImplementedError)
- `supported_builtin_tools()` — tool capability declaration (returns empty frozenset)

**Model metadata via ModelProfile (dataclass):**
```python
@dataclass
class ModelProfile:
    supports_tools: bool = True
    supports_json_schema_output: bool = ...
    supports_json_object_output: bool = ...
    supports_image_output: bool = ...
    supports_thinking: bool = ...
    thinking_always_enabled: bool = ...
    default_structured_output_mode: str = 'tool'
    json_schema_transformer: ... = None
    supported_builtin_tools: frozenset = ...
```

**Model settings via ModelSettings (TypedDict, total=False):**
```python
class ModelSettings(TypedDict, total=False):
    max_tokens: int; temperature: float; top_p: float
    seed: int; presence_penalty: float; frequency_penalty: float
    logit_bias: dict; stop_sequences: list[str]
    thinking: ThinkingLevel; parallel_tool_calls: bool
    timeout: float; extra_headers: dict; extra_body: object
```

**Key insight:** PydanticAI's cleanest separation: `ModelProfile` = what the model CAN do (static, set once), `ModelSettings` = how to USE the model (per-request). The `prepare_request()` pipeline merges both. This 3-layer architecture (ABC + Profile + Settings) is the most thoughtful design.

### 5. vLLM — Most Detailed Architecture Introspection

**Model metadata via ModelConfig (dataclass with getters):**
```python
# Getters (require ParallelConfig for distributed)
get_hidden_size() -> int
get_vocab_size() -> int
get_head_size() -> int
get_num_attention_heads(parallel_config) -> int
get_total_num_attention_heads() -> int
get_total_num_kv_heads() -> int
get_num_layers(parallel_config) -> int
get_total_num_hidden_layers() -> int
get_num_experts() -> int
get_sliding_window() -> int | None

# Properties (no args needed)
architectures -> list[str]
is_moe -> bool
is_encoder_decoder -> bool
is_hybrid -> bool
is_multimodal_model -> bool
is_quantized -> bool
uses_alibi -> bool
uses_mrope -> bool
use_mla -> bool
```

**Tokenizer:** `get_tokenizer()` on LLM class. Returns tokenizer instance.

**Key insight:** vLLM exposes architecture details because it NEEDS them for memory planning, tensor parallelism, and attention kernel selection. The getter pattern (`get_hidden_size()` not `hidden_size` property) is deliberate — some values depend on parallelism config. AVP needs architecture introspection for cross-model communication, so vLLM's level of detail is the right reference.

### 6. HuggingFace Transformers — Most Direct Access

**Model properties via PreTrainedModel:**
- `model.config` — PretrainedConfig instance with all architecture params
- `model.config.hidden_size`, `.num_attention_heads`, `.num_hidden_layers`, `.vocab_size`
- `model.config.max_position_embeddings` — context length
- `model.config.num_key_value_heads` — GQA heads (model-specific, not on base)
- `model.get_input_embeddings()` → nn.Embedding
- `model.get_output_embeddings()` → nn.Linear (lm_head)
- `model.num_parameters()` → int
- `model.dtype`, `model.device`

**Hidden state access:**
```python
# Via output_hidden_states=True in config or forward() kwargs
outputs = model(input_ids, output_hidden_states=True)
outputs.hidden_states  # tuple[Tensor, ...], one per layer
outputs.last_hidden_state  # Tensor [batch, seq, hidden]
outputs.past_key_values  # Cache object (DynamicCache)
```

**Cache API (DynamicCache):**
```python
cache.update(key_states, value_states, layer_idx) -> (K, V)
cache.get_seq_length(layer_idx) -> int
cache.reset()
cache.crop(max_length)
cache.reorder_cache(beam_idx)
# Per-layer: cache.layers[idx].keys, cache.layers[idx].values
```

**Pipeline (high-level):** Stores `self.model` and `self.tokenizer`. No architecture properties exposed directly — users access `pipeline.model.config`.

**Key insight:** HF is the "expose everything" approach. Works because the model object IS the thing. For AVP, we can't expose `model.config` directly (engine-agnostic), but we should expose the SAME information through a structured interface.

### 7. llama-cpp-python — Best GGUF Property API

**Model properties (all exposed as methods on Llama class):**
```python
n_ctx() -> int      # Context window size
n_embd() -> int     # Embedding dimension (= hidden_size)
n_vocab() -> int    # Vocabulary size
n_layer() -> int    # Number of layers (via metadata)
token_eos() -> int  # EOS token ID
token_bos() -> int  # BOS token ID
```

**Tokenization:**
```python
tokenize(text, add_bos, special) -> list[int]
detokenize(tokens, prev_tokens, special) -> bytes
```

**State management (KV-cache):**
```python
save_state() -> LlamaState    # Captures KV-cache + logits + RNG
load_state(state) -> None      # Restores full state
# LlamaState: input_ids, scores, n_tokens, llama_state (bytes), seed
```

**Embedding extraction:**
```python
embed(input, normalize, truncate, return_count)  # Via llama_get_embeddings()
```

**Key insight:** llama-cpp-python's API is the simplest and most direct. `n_embd()`, `n_vocab()`, `n_ctx()` — that's it. No config objects, no getters with parameters. AVP's `ModelIdentity` already captures similar info but it's only used for handshake, not exposed as connector properties.

### 8. OpenAI SDK — Minimal Metadata

**Model object:**
```python
class Model:
    id: str          # "gpt-4"
    created: int     # Unix timestamp
    object: str      # "model"
    owned_by: str    # "openai"
```

**No architecture properties.** No hidden_size, no context window, no capability flags. Everything is in documentation, not the API.

**Key insight:** The OpenAI SDK is deliberately opaque. This is the right choice for a HOSTED API where internals are irrelevant. Not the right model for AVP (we need architecture details for cross-model communication).

---

## Cross-Framework Pattern Analysis

### Pattern 1: Minimal Abstract Contract

| Framework | Abstract Methods | Evidence Quality |
|-----------|-----------------|------------------|
| smolagents | 1 (generate) | Battle-tested |
| LangChain | 2 (_generate + _llm_type) | Battle-tested |
| PydanticAI | 3 (request + model_name + system) | Well-established |
| **AVP** | **1 (get_model_identity)** | — |
| LiteLLM | 6 | Battle-tested but criticized |

**Consensus:** 1-3 abstract methods. LiteLLM's 6 is universally considered too heavy. AVP's 1 is appropriate.

### Pattern 2: Capability/Metadata Separation

Every mature framework separates "call the model" from "describe the model":

| Framework | Generation Interface | Metadata Object | Capability Object |
|-----------|---------------------|-----------------|-------------------|
| LangChain | `_generate()` | — | `ModelProfile` (TypedDict) |
| PydanticAI | `request()` | — | `ModelProfile` (dataclass) + `ModelSettings` (TypedDict) |
| LiteLLM | `transform_*()` | `ModelInfo` (TypedDict, 50+ fields) | `supports_*()` functions |
| vLLM | engine internals | `ModelConfig` (dataclass + getters) | `is_moe`, `is_multimodal`, etc. |
| HuggingFace | `model()` / `generate()` | `PretrainedConfig` (class) | `output_hidden_states` flag |
| llama-cpp-python | `generate()` | `n_embd()`, `n_ctx()`, etc. | — |
| smolagents | `generate()` | `model_id` only | `supports_stop_parameter` |
| OpenAI | `chat.completions.create()` | `Model` (4 fields) | None in API |

**AVP current state:** `ModelIdentity` serves as metadata but it's only for handshake. No capability object. No properties on connector for architecture introspection.

### Pattern 3: Tokenization Handling

| Framework | Tokenization | Part of ABC? | Fallback |
|-----------|-------------|-------------|----------|
| LangChain | `get_token_ids()`, `get_num_tokens()` | On base class | GPT-2 tokenizer |
| PydanticAI | `count_tokens()` | Optional, raises NotImplementedError | None |
| LiteLLM | `token_counter()` | Standalone function | Disabled gracefully |
| smolagents | `self.tokenizer` on concrete class | No | N/A |
| vLLM | `get_tokenizer()` on LLM | No (engine method) | N/A |
| HuggingFace | `tokenizer` attribute | Pipeline stores it | Extracted from processor |
| llama-cpp-python | `tokenize()`, `detokenize()` | On Llama class | N/A |
| **AVP** | `tokenize()` | On ABC, raises NotImplementedError | None |

**Consensus:** Tokenization is never abstract. Always optional with graceful degradation. AVP is aligned.

### Pattern 4: Hidden State / KV-Cache Access

This is where AVP is UNIQUE among all frameworks. No other connector/provider ABC exposes hidden states:

| Framework | Hidden State Access | KV-Cache Access |
|-----------|-------------------|-----------------|
| HuggingFace | `output_hidden_states=True` → tuple per layer | `DynamicCache` class with full API |
| llama-cpp-python | Via `embed()` for final layer | `save_state()`/`load_state()` (opaque blob) |
| vLLM | RFC #18176 still open (10+ months) | KVConnectorBase_V1 plugin only |
| LangChain | None | None |
| LiteLLM | None | None |
| PydanticAI | None | None |
| smolagents | None | None |
| OpenAI | None | None |

**AVP is the only framework that puts hidden state access on the connector ABC.** This is a novel requirement driven by latent communication. No precedent to follow — must be designed from first principles.

### Pattern 5: Engine-Specific vs Universal Features

| Framework | Approach |
|-----------|----------|
| LangChain | Universal ABC + engine-specific methods (bind_tools on subclass) |
| PydanticAI | Universal ABC + ModelProfile capability flags + `prepare_request()` gating |
| LiteLLM | Universal OpenAI-format request + provider transforms |
| smolagents | Universal generate() + engine-specific cleanup/streaming |
| vLLM | Engine-specific everything (single engine) |

**Consensus:** Keep the ABC universal. Engine-specific features live on subclasses or are gated by capability flags. Never force engine-specific methods into the ABC.

---

## Anti-Patterns and Documented Failures

### 1. LiteLLM's Heavy ABC (6 abstract methods)

**What they tried:** Requiring every provider to implement 6 methods including request transformation, response parsing, error mapping, and parameter validation.

**Why it fails:** Provider authors complain about the burden. Many providers have near-identical implementations for 4 of the 6 methods. The boilerplate discourages community contributions. New providers take days instead of hours.

**Lesson for AVP:** Keep abstract methods to 1-2. Move everything else to optional overrides with sensible defaults.

### 2. LangChain's BaseLanguageModel Dual Hierarchy

**What they tried:** Separate `BaseLLM` (text completion) and `BaseChatModel` (chat) with a shared `BaseLanguageModel` parent. Three levels of inheritance.

**Why it fails:** Every modern model is chat-based. The LLM/ChatModel split is now vestigial. Provider implementers are confused about which to extend. The `generate_prompt()` abstract method on `BaseLanguageModel` is rarely implemented directly.

**Lesson for AVP:** Don't split the ABC by era or paradigm. One connector base, capability-flagged.

### 3. Exposing Raw Config Objects

**What they tried:** HuggingFace exposes `model.config` directly — a mutable PretrainedConfig with 100+ attributes varying by model family.

**Why it fails for wrappers:** Consumers can't rely on any specific attribute existing. `config.num_key_value_heads` exists on Llama but not GPT-2. `config.head_dim` exists on some but not others. Every consumer must do `getattr(config, 'num_key_value_heads', None)` with fallback logic.

**Why it works for HF:** HF IS the config. They define it.

**Lesson for AVP:** Don't expose engine-specific config objects. Define a fixed schema (`ModelIdentity` or similar) with guaranteed fields. Map engine configs to it inside the connector.

### 4. smolagents Missing Architecture Introspection

**What they tried:** Treating models as pure text-in/text-out boxes with no architecture awareness.

**Why it's limiting:** smolagents can't do token counting, can't estimate costs, can't select models based on capabilities. The `supports_stop_parameter` hack (checking model_id against a hardcoded list) is brittle.

**Lesson for AVP:** AVP explicitly needs architecture info (hidden_dim for projection, num_layers for injection point, vocab for overlap). Don't follow smolagents here.

### 5. OpenAI's Opaque Model Object

**What they tried:** 4-field Model object (id, created, object, owned_by). No capabilities, no architecture.

**Why it works for them:** Hosted API, internals are irrelevant to callers.

**Why it fails for AVP:** AVP needs architecture dimensions for cross-model communication. Can't negotiate latent transfer without knowing hidden_dim, num_layers, tokenizer_hash.

---

## Findings Ranked by Relevance to AVP

### Critical: Separate Metadata from Generation Interface

**Rank:** Critical
**Evidence:** Battle-tested (LangChain, PydanticAI, LiteLLM, vLLM all do this)
**Finding:** Model capabilities and architecture info should be a data object, not methods scattered across the connector.
**Common mistake:** Mixing "describe" methods with "do" methods on the same class, leading to bloated ABCs.
**Recommendation:** Create a `ModelInfo` (or extend `ModelIdentity`) dataclass with fixed fields for architecture (hidden_dim, num_layers, vocab_size, max_context_len, head_dim, num_kv_heads) plus capability flags (can_think, supports_cross_model, supports_streaming). Return it from a single method or property. Keep `generate()` and `think()` separate.
**AVP-specific:** `ModelIdentity` already has most architecture fields. The gap is (a) it's not exposed as a connector property, (b) it lacks vocab_size and max_context_len, (c) it has no capability flags.

### Critical: 1-3 Abstract Methods Maximum

**Rank:** Critical
**Evidence:** Battle-tested across all 8 frameworks. LiteLLM's 6-method ABC is the documented negative case.
**Finding:** The minimal contract should be 1-3 abstract methods. Everything else must have defaults.
**Common mistake:** Adding abstract methods for features that only some engines support.
**Recommendation:** AVP's current 1-method ABC (`get_model_identity`) is correct. If adding more, limit to 2-3 total (e.g., `get_model_identity` + `generate`). Keep `think()`, `extract_hidden_state()`, etc. as optional with concrete defaults.
**AVP-specific:** Consider making `generate()` abstract (every connector should implement it). That gives 2 abstract methods — still well within the consensus range.

### High: Capability Flags as Structured Data

**Rank:** High
**Evidence:** Well-established (LangChain ModelProfile, PydanticAI ModelProfile, LiteLLM supports_*)
**Finding:** Model capabilities should be queryable as structured data (bool flags, enums), not just "try it and catch NotImplementedError."
**Common mistake:** Using `hasattr()` or `try/except` to discover capabilities at runtime.
**Recommendation:** Add capability properties to the connector or to ModelIdentity:
```python
@property
def can_think(self) -> bool       # Already exists
@property
def can_stream(self) -> bool      # New
@property
def can_cross_model(self) -> bool # New
@property
def can_tokenize(self) -> bool    # New
```
Or, following LangChain/PydanticAI, put them in a `ModelProfile` dataclass.
**AVP-specific:** `can_think` already exists. Extend the pattern.

### High: Architecture Properties as Simple Properties (not getters)

**Rank:** High
**Evidence:** llama-cpp-python's `n_embd()`, `n_vocab()`, `n_ctx()` are the simplest. vLLM's `get_hidden_size()` with ParallelConfig parameter is overengineered for AVP's needs.
**Finding:** Architecture dimensions should be simple properties, not methods with parameters.
**Common mistake:** Over-parameterizing getters when the values are fixed at model load time.
**Recommendation:** Expose architecture info as properties on the connector:
```python
@property
def hidden_dim(self) -> int      # delegates to ModelIdentity
@property
def num_layers(self) -> int
@property
def vocab_size(self) -> int
@property
def context_length(self) -> int  # NEW - not in ModelIdentity today
```
These can delegate to `get_model_identity()` internally.
**Trade-off:** Adding properties increases the connector surface area. Alternative: just make `identity: ModelIdentity` a cached property and access `connector.identity.hidden_dim`.

### High: Tokenization as Optional with Fallback

**Rank:** High
**Evidence:** Well-established. LangChain falls back to GPT-2. PydanticAI raises NotImplementedError. LiteLLM has a standalone function.
**Finding:** Tokenization should never be abstract. Provide a `tokenize()` method with a `NotImplementedError` default, and a `can_tokenize` flag.
**Current state in AVP:** Already correct. `tokenize()` raises NotImplementedError.
**Recommendation:** Add `can_tokenize` property (return True in connectors that implement it).

### Medium: Three-Layer Architecture (ABC + Profile + Settings)

**Rank:** Medium
**Evidence:** Emerging consensus (PydanticAI is the cleanest example)
**Finding:** The most flexible pattern separates: (1) ABC contract (what you MUST implement), (2) Profile/Capabilities (what the model CAN do, static), (3) Settings (how to USE the model, per-request).
**Common mistake:** Mixing per-request settings (temperature, max_tokens) with static capabilities (supports_tools, hidden_dim).
**Recommendation:** AVP already has this partially:
- ABC: `EngineConnector` (what you implement)
- Identity/Profile: `ModelIdentity` (what the model IS — needs expansion)
- Settings: kwargs on `think()`/`generate()` (how to use it — could be formalized)

The missing piece is the middle layer. `ModelIdentity` covers architecture but not capabilities.

### Medium: Connector Factory Pattern Consistency

**Rank:** Medium
**Evidence:** Well-established across frameworks
**Finding:** Every framework with local model support has a factory method: HF's `from_pretrained()`, llama-cpp-python's `from_pretrained()`, AVP's `from_pretrained()`/`from_ollama()`.
**Current state:** AVP has engine-specific factories. This is fine — factories SHOULD be engine-specific (they take engine-specific args).
**Recommendation:** No change needed. Document that factory methods are not part of the ABC.

### Low: Async as Optional Override

**Rank:** Low
**Evidence:** Well-established (LangChain runs sync in executor by default; PydanticAI is async-first)
**Finding:** If the primary interface is sync, provide async as an optional override that defaults to `asyncio.to_thread(sync_version)`.
**Current state in AVP:** No async support.
**Recommendation:** When adding async, follow LangChain's pattern: provide default `athink()`/`agenerate()` that wrap sync methods in an executor.

### Informational: Hidden State Access is Novel Territory

**Rank:** Informational
**Evidence:** No framework except HuggingFace directly exposes hidden states. HF does it via `output_hidden_states=True` flag + ModelOutput dataclass. llama-cpp-python exposes final-layer embeddings only. vLLM has no merged solution after 10+ months.
**Finding:** AVP's `extract_hidden_state()` and `inject_and_generate()` on the connector ABC have NO precedent in any other framework. This is novel API design.
**Implication:** Can't copy an existing pattern. Must design from first principles. The current approach (optional methods with NotImplementedError defaults) is the safest choice.

---

## Specific Recommendations for AVP EngineConnector

### What to Keep (Already Correct)
1. Single abstract method (`get_model_identity`)
2. `can_think` property pattern
3. `think()`/`generate()` as optional high-level methods with defaults
4. `extract_hidden_state()`/`inject_and_generate()` as optional low-level methods
5. `tokenize()` as optional with NotImplementedError
6. Engine-specific factories NOT on the ABC
7. Extension policy (new methods always have defaults)

### What to Add
1. **`identity` cached property** — replaces `get_model_identity()` for property access:
   ```python
   @cached_property
   def identity(self) -> ModelIdentity: return self.get_model_identity()
   ```
2. **Expand ModelIdentity** with `vocab_size: int = 0` and `max_context_length: int = 0`
3. **Capability properties**: `can_stream`, `can_cross_model`, `can_tokenize`
4. **`get_embedding_weights()` should be a property** if it's always the same value

### What NOT to Do
1. Don't add more abstract methods
2. Don't expose engine-specific config objects (no `model.config`)
3. Don't add architecture introspection as methods with parameters (vLLM's `get_num_attention_heads(parallel_config)` pattern)
4. Don't split into BaseChatConnector / BaseCompletionConnector
5. Don't add a ModelSettings TypedDict yet — kwargs on think()/generate() work fine at current scale

---

## Gaps and Uncertainties

1. **No framework has solved "latent state access" as an abstraction.** AVP is pioneering. The right level of abstraction is unknown. Current approach (optional methods) is the safest but may need iteration.

2. **Streaming for latent communication** — no precedent. Streaming text generation is well-understood across all frameworks, but streaming KV-cache or hidden states has no established pattern.

3. **Cross-model capability negotiation** — LiteLLM's `get_supported_openai_params()` is the closest analog (which params does this provider support?), but it's for text API params, not for latent transfer modes.

4. **The "simple user" vs "advanced user" split** — PydanticAI's `prepare_request()` pipeline is the best example of hiding complexity while allowing customization. AVP's `easy.py` module fills this role. The connector ABC should serve advanced users; `avp.think()`/`avp.generate()` serves simple users.

---

## Recommended Reading

1. **PydanticAI models/__init__.py** — Cleanest modern ABC design. Model + ModelProfile + ModelSettings separation.
2. **LangChain model_profile.py** — ModelProfile TypedDict pattern for capability declaration.
3. **vLLM config/model.py** — Most comprehensive architecture introspection API (getter pattern).
4. **llama-cpp-python llama.py** — Simplest property-based architecture access (n_embd, n_ctx, n_vocab).
5. **LiteLLM BaseConfig** — What NOT to do (6 abstract methods) and what TO do (ModelInfo registry with 50+ fields).
