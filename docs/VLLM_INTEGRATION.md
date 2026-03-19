# vLLM Integration

AVP provides two plugins for vLLM: a **model plugin** that adds latent thinking steps during prefill, and a **KV connector plugin** that saves/loads KV-cache between agents via a file-based store.

Both plugins are auto-discovered via the `vllm.general_plugins` entry point when `avp` is installed.

## Requirements

- `pip install avp[vllm]` (vLLM >= 0.15.0)
- CUDA GPU
- Currently supports **Qwen2** model family only

## Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="AVPKVConnectorV1Dynamic",
    kv_connector_module_path="avp.connectors.vllm_kv_connector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "avp_latent_steps": 20,     # number of latent thinking steps
        "avp_store_dir": "/tmp/avp_kv_store",  # KV file store location
    },
)

engine = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,  # recommended for initial use
    kv_transfer_config=ktc,
)

params = SamplingParams(max_tokens=256, temperature=0.7)
outputs = engine.generate(["Solve step by step: 24 * 17 + 3"], params)
print(outputs[0].outputs[0].text)
```

Or via CLI:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enforce-eager \
  --kv-connector AVPKVConnectorV1Dynamic \
  --kv-connector-module-path avp.connectors.vllm_kv_connector \
  --kv-connector-extra-config '{"avp_latent_steps": 20}'
```

## Configuration

All configuration is passed via `kv_connector_extra_config`:

| Key | Default | Description |
|-----|---------|-------------|
| `avp_latent_steps` | `20` | Number of latent thinking steps during prefill. Set to `0` to disable. |
| `avp_store_dir` | `/tmp/avp_kv_store` | Directory for file-based KV store. |

The `avp_latent_steps` value is bridged to the model plugin via the `AVP_LATENT_STEPS` environment variable.

## How It Works

### Model Plugin (latent thinking)

During prefill, the model plugin runs N additional forward passes after the initial prompt processing:

1. Extract last hidden state from the forward pass output
2. Project it back to embedding space (softmax projection for tied-weight models, realignment matrix for untied)
3. Feed the projected embedding as the next input at the same position (overwrite pattern)
4. Repeat N times

This builds reasoning state in the KV-cache without generating text tokens. The overwrite pattern means the KV-cache grows by N entries (one per step).

The model plugin is registered as `AVPLatentQwen2ForCausalLM` via the `vllm.general_plugins` entry point. It wraps vLLM's `Qwen2ForCausalLM` with the latent loop.

### KV Connector Plugin (transfer)

The KV connector saves and loads KV-cache layers to/from a file-based store:

- **Producer**: After Agent A's request completes, KV layers are saved per-layer as torch tensors.
- **Consumer**: Before Agent B's forward pass, matching KV data is loaded from the store.
- **Key derivation**: Store keys are derived from prompt token IDs (content-addressable). Two requests with the same prompt tokens will share KV data.

## Architecture

```
Single vLLM Instance (e.g., Qwen2.5-7B)
+----------------------------------------------------------+
|  Model Plugin (AVPLatentQwen2ForCausalLM)                |
|    forward() -> initial prefill -> N latent steps        |
|                                                           |
|  KV Connector Plugin (AVPKVConnectorV1Dynamic)           |
|    save_kv_layer() -> FileKVStore (per-layer .pt files)  |
|    start_load_kv() -> load from FileKVStore              |
+----------------------------------------------------------+

Request 1 (Agent A "think"):        Request 2 (Agent B "generate"):
  latent steps -> enriched KV          load enriched KV -> generate text
  save to FileKVStore                  from FileKVStore
```

## Limitations

- **Qwen2 only**: Other model families (Llama, Gemma, Mistral) are not yet supported. The model plugin uses a generic factory internally, so adding support is straightforward.
- **No tensor parallelism**: Latent steps are automatically disabled when TP > 1 (embedding table is sharded, making softmax projection incorrect).
- **No pipeline parallelism**: Latent steps are disabled when PP > 1.
- **File-based store only**: KV transfer uses local filesystem. Not suitable for multi-node deployments. Redis/shared memory backends are planned.
- **`enforce_eager=True` recommended**: CUDA graph support is experimental. The plugin returns `requires_piecewise_for_cudagraph() = True` but this has not been validated.
- **Not production-hardened**: The plugin passes 475 mock-based tests but has limited real vLLM validation. Use with caution.

## Verifying the Plugin

After starting vLLM with the plugin, check the logs for:

```
AVP latent model plugin registered: AVPLatentQwen2ForCausalLM (latent_steps=10)
AVPKVConnectorV1Dynamic initialized: store=/tmp/avp_kv_store, latent_steps=10
```

Latent step timing is logged at DEBUG level:

```
Latent thinking: 20 steps in 90.4ms (4.5ms/step)
```

## Running Integration Tests

```bash
# Requires CUDA GPU and vLLM installed
pip install avp[vllm]
pytest tests/test_vllm_integration.py -v -m requires_vllm
```
