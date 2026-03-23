# audit-docs

Audit all AVP documentation against the actual codebase. Spawns parallel subagents to cross-check every engine, framework, and core API section.

## When to use

Run before releases, after API changes, or whenever you suspect docs are stale.

## Instructions

Launch 5 parallel Explore subagents, each auditing one area. Each agent reads the relevant source files and compares them against the documentation. Report all inconsistencies.

### Agent 1: Core API & README

Read `README.md` and compare against:
- `src/avp/__init__.py` — verify all exports mentioned in README exist
- `src/avp/easy.py` — verify `avp.generate()` and `avp.think()` signatures match README examples
- `src/avp/context_store.py` — verify `ContextStore` API (`.store()`, `.get()`, `.cleanup_expired()`, `.active_count`)
- `pyproject.toml` — verify all install extras mentioned in README exist and deps match
- `src/avp/version.py` — verify version badges are current
- Check: do README benchmark numbers match `docs/BENCHMARKS.md`?
- Check: are all `[Works With]` table links valid?

### Agent 2: HuggingFace & Ollama engines

Read `docs/FRAMEWORK_INTEGRATION.md` HuggingFace and Ollama sections and compare against:
- `src/avp/connectors/huggingface.py` — class name, `from_pretrained()`, `think()`, `generate()` signatures, `source=`, `cross_model=` parameter names
- `src/avp/connectors/ollama.py` — `OllamaConnector`, `from_ollama()` signature, import path
- Verify every code example in the doc would actually run (correct imports, parameter names, return types)

### Agent 3: llama.cpp & vLLM engines

Read `docs/FRAMEWORK_INTEGRATION.md` llama.cpp and vLLM sections, plus `docs/VLLM_INTEGRATION.md`, and compare against:
- `src/avp/connectors/llamacpp.py` — `LlamaCppConnector`, `from_pretrained()` signature
- `src/avp/connectors/vllm_model_plugin.py` — verify all architecture class names (`AVPLatent*ForCausalLM`) exist
- `src/avp/connectors/vllm_kv_connector.py` — verify class name `AVPKVConnectorV1Dynamic`, config keys (`avp_latent_steps`, `avp_store_dir`), `prepare_latent_prompt()` function
- `src/avp/connectors/_vllm_compat.py` — version range check matches pyproject.toml
- Verify `KVTransferConfig` usage pattern is consistent between FRAMEWORK_INTEGRATION and VLLM_INTEGRATION
- Check limitation claims (architectures supported, TP/PP restrictions, CUDA graph status)

### Agent 4: Framework integrations

Read `docs/FRAMEWORK_INTEGRATION.md` LangChain, CrewAI, AutoGen, and LangGraph sections and compare against:
- `src/avp/integrations/langchain.py` — `ChatAVP` class, constructor params (`model`, `role`, `store`, `store_key`, `source_model`, `cross_model`)
- `src/avp/integrations/crewai.py` — `AVPLLM` class, constructor params
- `src/avp/integrations/autogen.py` — `AVPChatCompletionClient` class, constructor params
- `src/avp/easy.py` — verify `avp.generate(store=, store_key=, prior_key=)` signature for LangGraph pattern
- Verify every code example uses correct import paths and parameter names

### Agent 5: Benchmarks & spec

Read `docs/BENCHMARKS.md` and compare against:
- `README.md` — do the benchmark numbers match between README and BENCHMARKS.md?
- Verify benchmark count claim (header says "N benchmarks")
- Verify install command in Reproduce section uses correct extras
- Check `avp-spec/` README badges and version numbers if accessible
- Verify the "Limitations" section is still accurate

## Output

For each agent, report:
- **PASS** items (verified correct)
- **FAIL** items with exact file, line, what's wrong, and what it should be
- **WARN** items that are technically correct but could be clearer

Compile all results into a single summary table.
