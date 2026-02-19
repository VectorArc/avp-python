# AVP Python SDK

Python implementation of the [Agent Vector Protocol](https://github.com/VectorArc/avp-spec) -- a binary protocol for same-model latent communication between LLM agents. Same-model agents exchange KV-cache and hidden states directly, skipping autoregressive generation. Different-model agents fall back to JSON.

## Install

```bash
# Core SDK (codec, handshake, session, transport, fallback)
pip install -e ".[server]"

# With latent communication support (realignment, KV-cache, HuggingFace connector)
pip install -e ".[latent]"

# Everything including dev tools
pip install -e ".[all]"
```

## Dependencies

Core:
- `numpy>=1.24`
- `protobuf>=4.21`
- `zstandard>=0.21`
- `httpx[http2]>=0.25`

Server (`[server]`):
- `fastapi>=0.104`
- `uvicorn[standard]>=0.24`

Latent communication (`[latent]`):
- `torch>=2.0`
- `transformers>=4.36`

Dev/test (`[dev]`):
- `pytest>=7.0`, `pytest-asyncio>=0.21`, `ruff>=0.1`, `grpcio-tools>=1.59`
- Tests require `torch` and `transformers` to run the full suite

## Quick Start

```python
import numpy as np
import avp

# Encode a hidden state
hidden = np.random.randn(4096).astype(np.float32)
metadata = avp.AVPMetadata(
    model_id="meta-llama/Llama-2-7b",
    hidden_dim=4096,
    payload_type=avp.PayloadType.HIDDEN_STATE,
    dtype=avp.DataType.FLOAT32,
    tensor_shape=(4096,),
)
data = avp.encode(hidden.tobytes(), metadata)

# Decode
msg = avp.decode(data)
print(msg.metadata.model_id)   # "meta-llama/Llama-2-7b"
print(msg.metadata.hidden_dim) # 4096
```

## Architecture

### Handshake

Agents exchange model identity (family, hash, hidden_dim, num_layers, num_kv_heads, head_dim). The `CompatibilityResolver` determines the communication mode:

- **Latent** -- same model hash, or same family + matching structure. Agents communicate via KV-cache and hidden states.
- **JSON** -- incompatible models. Agents fall back to text.

```python
from avp import HelloMessage, CompatibilityResolver, ModelIdentity, extract_model_identity

local = extract_model_identity(my_model)
remote = ModelIdentity.from_dict(remote_hello["identity"])
session = CompatibilityResolver.resolve(local, remote)
# session.mode == CommunicationMode.LATENT or CommunicationMode.JSON
```

### Realignment

For same-model communication, hidden states need projection from output space to input embedding space. The SDK computes a realignment matrix from the model's embedding weights (ported from [LatentMAS](https://github.com/Gen-Verse/LatentMAS) research). Models with tied weights (`tie_word_embeddings=True`) skip this step.

Matrices are cached to `~/.avp/realign/{model_hash}.pt`.

### KV-Cache Transfer

Serialize and deserialize KV-cache between agents. Supports both legacy tuple format and HuggingFace `DynamicCache`.

### Transport

HTTP/2 endpoints via FastAPI:

| Endpoint | Purpose |
|----------|---------|
| `POST /avp/v2/handshake` | Model identity exchange |
| `POST /avp/v2/transmit` | Binary payload transfer |
| `POST /avp/v2/text` | JSON fallback messages |
| `GET /health` | Health check |

### Binary Format

```
Bytes 0-1:   Magic (0x4156 = "AV")
Byte 2:      Version (0x01)
Byte 3:      Flags (bit 0=compressed, bit 1=hybrid, bit 2=has_map, bit 3=kv_cache)
Bytes 4-7:   Payload length (uint32 LE)
Bytes 8-11:  Metadata length (uint32 LE)
Bytes 12..N: Protobuf-encoded metadata
Bytes N..:   Raw tensor bytes (optionally zstd-compressed)
```

## Tests

```bash
pip install -e ".[dev,latent]"
pytest tests/
```

All 117 tests require `torch` and `transformers` to be installed for the full suite to run.

## License

Apache 2.0
