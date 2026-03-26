# Changelog

All notable changes to the AVP Python SDK are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

## [0.4.1] - 2026-03-25

### Added

- **Result objects** — `think()` returns `ThinkResult`, `generate()` returns `GenerateResult` (str subclass). No more Union return types. Metrics accessible via `result.metrics` instead of tuple unpacking.
- **`InspectResult`** — `avp.inspect()` returns a typed dataclass instead of `Dict[str, Any]`.
- **CRC32 payload checksum** — Optional integrity check on wire payloads (proto field 15). Encode always writes, decode verifies when present. Catches corruption and truncation.
- **`ConfigurationError`** — New error class for invalid arguments to `think()`/`generate()`. Subclass of `AVPError`, catchable by `except AVPError`.
- **`ProjectionError`** — New error class for cross-model projection failures. `RealignmentError` kept as alias.
- **`TensorLike`, `KVCache`** — Type aliases on `EngineConnector` for documentation.
- **`ContextStore.__contains__` and `__len__`** — `"key" in store` and `len(store)` now work.
- **Spec test vector validation** — 8 new tests cross-validate SDK against published spec hex baselines.

### Changed

- **`generate(prompt=)` replaces `generate(content=)`** — `content=` is deprecated with `DeprecationWarning`, will be removed in v2.0.
- **`EngineConnector` ABC simplified** — 1 abstract method (`get_model_identity`) instead of 6. All others have concrete defaults. Third-party connectors now need minimal boilerplate.
- **`AVPContext` is `kw_only`** — Positional construction no longer allowed.
- **`store` parameter typed** — `Optional[Any]` changed to `Optional[ContextStore]` in `generate()`.
- **`to_bytes()` dtype fix** — Reads actual dtype from KV-cache header instead of hardcoding FLOAT32.
- **Enum decode safety** — Unknown wire values for `DataType`, `PayloadType`, `CommunicationMode` now raise `DecodeError` with actionable message instead of crashing or silently corrupting.
- **Endianness enforced** — `embedding_to_bytes()` forces little-endian on big-endian hosts.
- **`inject_and_generate` default** — `max_new_tokens` aligned to 512 across all connectors (was 256 on some).

### Fixed

- **`OllamaConnector.get_model_identity()`** — Used wrong field names (`hidden_size` instead of `hidden_dim`, nonexistent `vocab_size`). Runtime crash.
- **`LlamaCppConnector.get_model_identity()`** — Same bug as Ollama.
- **Integrations stored `ThinkResult` instead of `AVPContext`** — LangChain, CrewAI, AutoGen now unwrap before storing.

### Removed

- **`AVPMetadata` compat properties** — `.embedding_dim`, `.data_type`, `.agent_id`, `.task_id` removed (never on PyPI, zero external users).
- **Dead code** — `_get_local_identity()` and cache (~37 lines), tuple guards in integrations.

## [0.4.0] - 2026-03-23

### Added

- **llama.cpp connector** – `LlamaCppConnector`: full think/generate latent pipeline on GGUF models via llama.cpp's embeddings API. Stop token fix, Jinja2 chat templates, memory leak fix (weakref.finalize), GGUF weight caching. GPU validated on A100. `pip install avp[llamacpp]`.
- **Ollama connector** – `OllamaConnector.from_ollama("qwen2.5:7b")`: resolves Ollama model names to GGUF blobs on disk, auto-unloads from Ollama server to free VRAM. Inherits full latent pipeline from LlamaCppConnector. `pip install avp[ollama]`.
- **vLLM latent communication** – Full integration: same-model KV transfer, cross-model rosetta, 4 model architectures (Qwen2, Llama, Mistral, Gemma), CUDA graph support, prefix caching, explicit store key API. GPU validated on A100. `pip install avp[vllm]`.
- **Framework integrations** – LangChain (`ChatAVP` BaseChatModel), CrewAI (`AVPLLM` BaseLLM), AutoGen (`AVPChatCompletionClient`). All support same-model latent + cross-model rosetta.
- **`[huggingface]` extra** – Discoverable alias for `[hf]`.
- **`[all]` extra** – Convenience bundle: hf + llamacpp + frameworks + transport (excludes vLLM).

### Changed

- **torch is now optional** – Projection math (rosetta/project.py, realign.py) rewritten in numpy. `pip install avp[ollama]` drops from ~3 GB to ~85 MB. torch only required for HuggingFace connector (`[hf]`) and vLLM plugin.
- **Base install is lightweight** – `pip install avp` installs only numpy, protobuf, zstandard (~25 MB). Engine-specific deps via extras.
- **Protocol version** bumped to 0.4.0.
- **Python requirement** raised from >=3.9 to >=3.10 (3.9 EOL October 2025).
- **transformers requirement** raised from >=4.36 to >=5.0 (4.x line is dead).
- **huggingface-hub requirement** raised from >=0.20 to >=1.0.
- **README rewritten** – 361 to 139 lines. Single install command, no redundant sections, SVG diagram, handshake negotiation prominent.
- **Framework Integration Guide rewritten** – Per-engine code examples for all 4 engines + 4 frameworks.
- **`calibrate()` simplified** – Signature reduced to `(source_model, target_model, source_tokenizer, target_tokenizer, device, auto_save)`. Raises `ValueError` for incompatible models instead of falling through to broken ridge regression.

### Removed

- **`pack()` / `unpack()` / `PackedMessage`** – Deprecated in v0.3.0, now removed. Use `think()` / `generate()`.
- **`PackMetrics` / `UnpackMetrics`** – Deprecated aliases, now removed. Use `ThinkMetrics` / `GenerateMetrics`.
- **`HandshakeMetrics`** – Exported but never instantiated. Removed.
- **`AVPAsyncClient`** – Exported but never used. Removed.
- **`encode_hidden_state()`** – Unused convenience wrapper. Removed.
- **Ridge regression / Procrustes projection** – `_ridge_regression()`, `_orthogonal_procrustes()`, `_extract_hidden_states()`, `DEFAULT_ANCHORS`. Failed at 0.004 cosine similarity, never benchmarked on real tasks. `RIDGE` and `PROCRUSTES` removed from `ProjectionMethod` enum.
- **`page_convert.py`** – PagedAttention conversion module, never imported.
- **`make_eval_callback()`** and ctypes tensor helpers in `_llamacpp_compat.py` – Old cb_eval approach replaced by embeddings API.

### Fixed

- **llama.cpp stop token** – Model emits `<|endoftext|>` (token 151643) not `<|im_end|>` on embeddings context. Fix: token-level + text-based stop detection. GSM8K: 0% to 68%.
- **vLLM CUDA graph compatibility** – Pass dummy `input_ids` during latent steps for graph capture.
- **vLLM projection performance** – Cache numpy weights in setup instead of copying 600 MB+ GPU to CPU per latent iteration.
- **Error messages** – All "pip install avp should include this dependency" updated to "pip install avp[hf]".
- **Version consistency** – All Modal benchmarks updated from `@engine_integration` to `@main`, transformers >=5.0, vLLM upper bound added.

## [0.3.2] - 2026-03-12

### Added

- **Colab quickstart notebook** – `notebooks/avp_quick_start.ipynb`. Runs on a free T4 GPU in ~8 minutes. Compares direct, latent (AVP), and text chain on 10 GSM8K problems.
- **Open in Colab badge** in README.

### Changed

- **Cross-model projection is now opt-in** – `source_model=` (easy API) and `source=` (connector API) now require `cross_model=True` for latent transfer. Without it, falls back to text-only generation with a `UserWarning` explaining how to opt in. Rosetta Stone projection is experimental – accuracy varies by task type (structured tasks work well, comprehension may degrade). Same-model latent transfer is unaffected.
- **`think()` and `generate()` can now use different prompts** – e.g., researcher prompt for `think()`, solver prompt for `generate()`. Previously returned empty output due to the `prompt_len` bug below.

### Fixed

- **Critical: `prompt_len` bug in `connector.generate()`** – `prompt_len` was computed after extending the attention mask with KV-cache entries, causing generated tokens to be sliced incorrectly. This made `generate()` with `context=` return empty or truncated output, especially with different prompts for `think()` and `generate()`.
- **Easy API cross-model path dropped user-provided `context=`** – always ran a fresh `think()` instead of using the caller's context.
- **Easy API cross-model path ignored `store`/`store_key`/`prior_key`** – ContextStore was not consulted or updated in the cross-model code path.

## [0.3.1] - 2026-03-08

### Fixed

- **Protobuf compatibility** – Removed gencode version check from `avp_pb2.py` that required protobuf >=6.31.1 at runtime. Now works with protobuf >=4.21 as declared in dependencies. Fixes installation on Google Colab and other environments with protobuf 4.x/5.x.

## [0.3.0] - 2026-03-07

### Added

- **`think()` / `generate()` API** – New primary API replacing `pack()`/`unpack()`. Zero-friction entry point: `avp.generate("Solve: 2+2", model="Qwen/Qwen2.5-7B-Instruct")`.
- **Cross-model `source=` parameter** – `connector.generate(prompt, context=ctx, source=other)` automatically calibrates and projects across models. Zero ceremony.
- **Easy API cross-model** – `avp.generate(prompt, model=target, source_model=source)` handles everything: model loading, handshake, projection.
- **`ContextStore`** – Thread-safe, TTL-backed store for `AVPContext` objects. Enables multi-turn latent conversations.
- **`avp.inspect(data)`** – Decode AVP binary header/metadata without loading models. Returns dict with version, flags, model_id, dimensions, etc.
- **Debug mode** – `debug=True` on `think()`/`generate()` surfaces `TransferDiagnostics`: norm trajectory, projection metrics, quality gate result, text baseline comparison.
- **Always-on warnings** – `RuntimeWarning` for empty output, NaN/Inf in hidden states. NaN early exit in latent loop.
- **Vocabulary-overlap projection** – Cross-family zero-parameter projection through shared BPE tokens (~85% overlap for Qwen/Llama). Strict generalization of vocab-mediated projection.
- **Per-transfer quality gate** – `assess_transfer(prompt_tokens)` recommends latent vs JSON based on prompt length. Advisory only. Default threshold: 300 tokens.
- **Projection validation** – Two-tier gate: cosine similarity (fast, ~1ms) + pseudo-perplexity (~30ms). `validate_projection()` for model-pair diagnostics.
- **`resolution_path` on `SessionInfo`** – Exposes which handshake rule matched: `hash_match`, `structural_match`, `shared_tokenizer`, `avp_map_file`, `vocab_overlap`, `json_fallback`.
- **`tokenizer_hash` on `ModelIdentity`** – SHA-256 of sorted tokenizer vocabulary. Enables automatic cross-model projection via shared tokenizer detection.
- **vLLM connector (experimental)** – `VLLMConnector` (SDK wrapper) + `AVPKVConnectorV1Dynamic` (KVConnectorBase_V1 plugin). Text generation and identity extraction work. KV-cache transfer plugin has not been validated end-to-end with a real vLLM engine – known issues with PagedAttention format conversion, CUDA graph compatibility, and concurrent request isolation. Use `HuggingFaceConnector` for production latent transfer.
- **`GenerateMetrics`** – Observability for `generate()`: think + generate durations, context/store flags, debug diagnostics.
- **`HandshakeMetrics`** – Resolution path, mode, avp_map_id, duration.
- **7 benchmark suites** – GSM8K (4-agent, 2-agent), HotpotQA, fan-out, MATH 2-agent, HumanEval, DebugBench. Cloud results on all.

### Changed

- **API rename**: `pack()` → `think()`, `unpack()` → `generate()`. Old names still work with deprecation warnings.
- **`PackMetrics`** is now an alias for `ThinkMetrics`.
- **Protocol version** bumped to 0.3.0.
- **`CommunicationMode` enum** – `LATENT = 0`, `JSON = 1`. Simplified from three values.
- **Flag bits renumbered** – `FLAG_COMPRESSED = 0x01`, `FLAG_HAS_MAP = 0x02`, `FLAG_KV_CACHE = 0x04`.
- **Handshake resolution** now checks vocabulary overlap (>= 100 shared tokens) before falling back to JSON.
- **Quality gate threshold** lowered from 512 to 300 tokens based on cross-benchmark validation.
- **Package extras** – torch and transformers are now required deps. `pip install avp` just works. `[vllm]` extra for production serving. Removed `[latent]`, `[hf]`, `[demo]`, `[all]`.

### Removed

- **Hybrid mode** – Wire format bundling latent + text fallback. Never consumed by any pipeline. `encode_hybrid()`, `FLAG_HYBRID`, `CommunicationMode.HYBRID`, `HybridPayload`, `HybridChunk`, `ChunkType` all removed.
- **Universal representation mode** – Learned cross-model adapters via `inputs_embeds`. Validated negative (0% same-model accuracy). `src/avp/universal/` deleted.
- **`FallbackRequest`** – Dataclass for requesting JSON fallback. Never used by any pipeline.
- **`FallbackRequested`** – Exception for fallback signaling. Never raised.
- **`bytes_to_embedding()`** – Utility function, never called.
- **`confidence_score`** – Metadata field, never set to non-zero. Removed from wire format.
- **v0.1.0 proto backward-compat fields** – `embedding_dim` (100), `data_type` (101), `agent_id` (102), `task_id` (103) removed from protobuf schema.

### Fixed

- **Tied-weight models** – Softmax projection (`project_to_embedding_space()`) instead of simple normalize. Fixes cosine similarity from ~0.24 to ~1.0.
- **Vocab size mismatch** – `vocabulary_mediated_projection()` truncates to shared prefix when embedding tables differ (e.g. Qwen 7B vs 1.5B: 152,064 vs 151,936).
- **Pseudo-perplexity alignment** – Compare `projected[i]` to `target_embed[token_ids[i+1]]` for next-token prediction. Cast `inputs_embeds` to model dtype.
- **KV-cache serialization** – Fix bfloat16 support and transformers 5.x compatibility.
- **Cross-platform** – Windows console encoding, MPS device detection, pre-Ampere GPU support.

## [0.2.3] - 2026-02-28

### Added

- Vocabulary-overlap projection for cross-family models (Qwen/Llama).
- Auto-discovery of vocabulary overlap in handshake.
- Configurable projection temperature with empirical default.
- Cross-model benchmark mode and cross-process demo.
- Rosetta mode for HotpotQA and fan-out benchmarks.

### Fixed

- Vocab size mismatch in vocabulary-mediated projection.
- New-user experience issues found during docs audit.

## [0.2.2] - 2026-02-25

### Added

- `generate()` API: combined think + store + generate in one call.
- `ContextStore` for multi-turn latent context management.
- Observability metrics (`ThinkMetrics`, `GenerateMetrics`, `HandshakeMetrics`).
- 3 new benchmarks (MATH-500, HumanEval, fan-out) with cloud runner.
- Confidence intervals and per-sample agreement analysis.

### Fixed

- MATH-500 answer normalization: strip LaTeX sizing commands.

## [0.2.1] - 2026-02-22

### Added

- Zero-friction `pack()` / `unpack()` easy API.
- Model validation, cleaner exports, example improvements.

### Fixed

- Cold-start developer experience: model size warnings, deprecation notices.

## [0.2.0] - 2026-02-20

### Added

- Rosetta Stone v2: vocabulary-mediated cross-model projection (zero learned parameters).
- Cross-model handshake with `tokenizer_hash` and `avp_map_id`.
- Projection quality validation (cosine similarity + pseudo-perplexity).
- Mixed-model demo with automatic LATENT/JSON handshake.
- KV-cache truncation modes (sequential, latent_only).
- GSM8K 4-agent benchmark harness.
- End-to-end integration tests.
- HTTP/2 transport (sync + async) with FastAPI server.
- Session management with TTL.
- zstd compression.

### Fixed

- Hidden state normalization before injection (LatentMAS port).
- KV-cache serialization for bfloat16.

## [0.1.0] - 2026-02-15

### Added

- Initial release: same-model latent communication.
- Binary codec: encode/decode hidden states, KV-cache, embeddings.
- Protobuf metadata with 12-byte header.
- Realignment for untied-weight models.
- HuggingFace Transformers connector.
