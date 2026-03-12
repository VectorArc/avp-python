# Changelog

All notable changes to the AVP Python SDK are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-03-08

### Fixed

- **Protobuf compatibility** — Removed gencode version check from `avp_pb2.py` that required protobuf >=6.31.1 at runtime. Now works with protobuf >=4.21 as declared in dependencies. Fixes installation on Google Colab and other environments with protobuf 4.x/5.x.

## [0.3.0] - 2026-03-07

### Added

- **`think()` / `generate()` API** — New primary API replacing `pack()`/`unpack()`. Zero-friction entry point: `avp.generate("Solve: 2+2", model="Qwen/Qwen2.5-7B-Instruct")`.
- **Cross-model `source=` parameter** — `connector.generate(prompt, context=ctx, source=other)` automatically calibrates and projects across models. Zero ceremony.
- **Easy API cross-model** — `avp.generate(prompt, model=target, source_model=source)` handles everything: model loading, handshake, projection.
- **`ContextStore`** — Thread-safe, TTL-backed store for `AVPContext` objects. Enables multi-turn latent conversations.
- **`avp.inspect(data)`** — Decode AVP binary header/metadata without loading models. Returns dict with version, flags, model_id, dimensions, etc.
- **Debug mode** — `debug=True` on `think()`/`generate()` surfaces `TransferDiagnostics`: norm trajectory, projection metrics, quality gate result, text baseline comparison.
- **Always-on warnings** — `RuntimeWarning` for empty output, NaN/Inf in hidden states. NaN early exit in latent loop.
- **Vocabulary-overlap projection** — Cross-family zero-parameter projection through shared BPE tokens (~85% overlap for Qwen/Llama). Strict generalization of vocab-mediated projection.
- **Per-transfer quality gate** — `assess_transfer(prompt_tokens)` recommends latent vs JSON based on prompt length. Advisory only. Default threshold: 300 tokens.
- **Projection validation** — Two-tier gate: cosine similarity (fast, ~1ms) + pseudo-perplexity (~30ms). `validate_projection()` for model-pair diagnostics.
- **`resolution_path` on `SessionInfo`** — Exposes which handshake rule matched: `hash_match`, `structural_match`, `shared_tokenizer`, `avp_map_file`, `vocab_overlap`, `json_fallback`.
- **`tokenizer_hash` on `ModelIdentity`** — SHA-256 of sorted tokenizer vocabulary. Enables automatic cross-model projection via shared tokenizer detection.
- **vLLM connector (experimental)** — `VLLMConnector` (SDK wrapper) + `AVPKVConnectorV1Dynamic` (KVConnectorBase_V1 plugin). Text generation and identity extraction work. KV-cache transfer plugin has not been validated end-to-end with a real vLLM engine — known issues with PagedAttention format conversion, CUDA graph compatibility, and concurrent request isolation. Use `HuggingFaceConnector` for production latent transfer.
- **`GenerateMetrics`** — Observability for `generate()`: think + generate durations, context/store flags, debug diagnostics.
- **`HandshakeMetrics`** — Resolution path, mode, avp_map_id, duration.
- **7 benchmark suites** — GSM8K (4-agent, 2-agent), HotpotQA, fan-out, MATH 2-agent, HumanEval, DebugBench. Cloud results on all.

### Changed

- **API rename**: `pack()` → `think()`, `unpack()` → `generate()`. Old names still work with deprecation warnings.
- **`PackMetrics`** is now an alias for `ThinkMetrics`.
- **Protocol version** bumped to 0.3.0.
- **`CommunicationMode` enum** — `LATENT = 0`, `JSON = 1`. Simplified from three values.
- **Flag bits renumbered** — `FLAG_COMPRESSED = 0x01`, `FLAG_HAS_MAP = 0x02`, `FLAG_KV_CACHE = 0x04`.
- **Handshake resolution** now checks vocabulary overlap (>= 100 shared tokens) before falling back to JSON.
- **Quality gate threshold** lowered from 512 to 300 tokens based on cross-benchmark validation.
- **Package extras** — torch and transformers are now required deps. `pip install avp` just works. `[vllm]` extra for production serving. Removed `[latent]`, `[hf]`, `[demo]`, `[all]`.

### Removed

- **Hybrid mode** — Wire format bundling latent + text fallback. Never consumed by any pipeline. `encode_hybrid()`, `FLAG_HYBRID`, `CommunicationMode.HYBRID`, `HybridPayload`, `HybridChunk`, `ChunkType` all removed.
- **Universal representation mode** — Learned cross-model adapters via `inputs_embeds`. Validated negative (0% same-model accuracy). `src/avp/universal/` deleted.
- **`FallbackRequest`** — Dataclass for requesting JSON fallback. Never used by any pipeline.
- **`FallbackRequested`** — Exception for fallback signaling. Never raised.
- **`bytes_to_embedding()`** — Utility function, never called.
- **`confidence_score`** — Metadata field, never set to non-zero. Removed from wire format.
- **v0.1.0 proto backward-compat fields** — `embedding_dim` (100), `data_type` (101), `agent_id` (102), `task_id` (103) removed from protobuf schema.

### Fixed

- **Tied-weight models** — Softmax projection (`project_to_embedding_space()`) instead of simple normalize. Fixes cosine similarity from ~0.24 to ~1.0.
- **Vocab size mismatch** — `vocabulary_mediated_projection()` truncates to shared prefix when embedding tables differ (e.g. Qwen 7B vs 1.5B: 152,064 vs 151,936).
- **Pseudo-perplexity alignment** — Compare `projected[i]` to `target_embed[token_ids[i+1]]` for next-token prediction. Cast `inputs_embeds` to model dtype.
- **KV-cache serialization** — Fix bfloat16 support and transformers 5.x compatibility.
- **Cross-platform** — Windows console encoding, MPS device detection, pre-Ampere GPU support.

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
