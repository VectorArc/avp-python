# vLLM Integration

AVP provides two plugins for vLLM: a **model plugin** that adds latent thinking steps during prefill, and a **KV connector plugin** that enables multi-agent KV-cache transfer between requests.

Both plugins are auto-discovered via the `vllm.general_plugins` entry point when `avp` is installed.

## Requirements

- `pip install avp[vllm]` (vLLM >= 0.17.0)
- CUDA GPU
- Currently supports **Qwen2** model family (Llama, Mistral planned)

## Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="AVPKVConnectorV1Dynamic",
    kv_connector_module_path="avp.connectors.vllm_kv_connector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "avp_latent_steps": 20,
        "avp_store_dir": "/tmp/avp_kv_store",
    },
)

engine = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enforce_eager=True,
    kv_transfer_config=ktc,
    hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
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
  --kv-connector-extra-config '{"avp_latent_steps": 20}' \
  --hf-overrides '{"architectures": ["AVPLatentQwen2ForCausalLM"]}'
```

Alternatively, set `AVP_OVERRIDE_QWEN2=1` to override all Qwen2 model loads globally:

```bash
AVP_OVERRIDE_QWEN2=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enforce-eager \
  --kv-connector AVPKVConnectorV1Dynamic \
  --kv-connector-module-path avp.connectors.vllm_kv_connector \
  --kv-connector-extra-config '{"avp_latent_steps": 20}'
```

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `avp_latent_steps` | `20` | Number of latent thinking steps during prefill. Set to `0` to disable. |
| `avp_store_dir` | `/tmp/avp_kv_store` | Directory for file-based KV store. Also configurable via `AVP_KV_STORE_DIR` env var. |

## How It Works

### Model Plugin (latent thinking)

During prefill, the model plugin runs N additional forward passes using the overwrite pattern:

1. Extract last hidden state from the forward pass output
2. Project it back to embedding space (softmax projection for tied-weight models, realignment matrix for untied)
3. Feed the projected embedding as the next input at the same position
4. Repeat N times

Each step reads the previous step's KV entry at the overwrite position, creating a recurrence that accumulates information. The enriched KV entry persists through all decode tokens.

The latent loop handles **multi-request batches**: it identifies prefill requests via `query_start_loc`, projects all their hidden states together, and forwards as a single batched decode-like step. Decode-only requests pass through unmodified. No `max_num_seqs=1` restriction needed.

Validated: +4-8pp accuracy gain on GSM8K (Qwen 7B, A100) with ~17ms/step latency.

### KV Connector Plugin (multi-agent transfer)

The KV connector enables Agent A's computation to transfer to Agent B:

1. **Agent A's request finishes**: `save_kv_layer` extracts per-request KV entries from vLLM's paged buffer using the slot_mapping formula (`block_id * block_size + offset`). Saved per-layer to a file-based store.
2. **Agent B's request arrives**: `get_num_new_matched_tokens` reports stored token count. Scheduler allocates blocks and marks positions as externally computed.
3. **Before Agent B's forward**: `start_load_kv` injects stored KV into Agent B's allocated blocks.
4. **Agent B generates**: All decode tokens attend to Agent A's transferred KV as a prefix.

## Architecture

```
Agent A Request                        Agent B Request

Orchestration                          Orchestration
    |                                      |
    v                                      v
Model Plugin                           KV Connector
  latent steps (overwrite)               loads Agent A's KV from store
  enriches KV at last position           injects into allocated blocks
    |                                      |
    v                                      v
KV Connector                           Model Plugin
  extracts per-request KV                 (latent steps optional)
  saves to FileKVStore                    generates from Agent A's context
```

## Limitations

- **Qwen2 only**: Other model families not yet registered. The factory pattern supports adding them.
- **No tensor parallelism**: Latent steps disabled when TP > 1 (embedding table is sharded).
- **No pipeline parallelism**: Latent steps disabled when PP > 1.
- **`enforce_eager=True` recommended**: CUDA graph support not validated.
- **Overwrite pattern**: Each step overwrites the same KV position (1-position information bottleneck). The extend pattern (growing KV-cache) is not yet supported in vLLM due to block allocation constraints.
- **File-based store**: KV transfer uses local filesystem. Not suitable for multi-node.
- **Validated on vLLM 0.17.x only**: Internal API dependencies may break on other versions.

## Verifying the Plugin

Check logs for:

```
AVP latent model plugin registered (latent_steps=20, override_qwen2=False)
AVPKVConnectorV1Dynamic initialized: store=/tmp/avp_kv_store, block_size=16, latent_steps=20
```

Enable DEBUG logging for step timing:

```
Latent thinking: 4 reqs, 20 steps in 300.2ms (15.0ms/step)
Saved KV for abc123: 28 layers, 99 tokens
```

## Running Integration Tests

```bash
pip install avp[vllm]
pytest tests/test_vllm_integration.py -v -m requires_vllm
```
