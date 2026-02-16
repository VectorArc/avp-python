# AVP Python SDK

Python implementation of the [Agent Vector Protocol](https://github.com/VectorArc/avp-spec) — a binary protocol for efficient agent-to-agent vector communication.

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
import numpy as np
import avp

# Encode an embedding
embedding = np.random.randn(384).astype(np.float32)
data = avp.encode(embedding, model_id="all-MiniLM-L6-v2")

# Decode it back
msg = avp.decode(data)
print(msg.metadata.model_id)       # "all-MiniLM-L6-v2"
print(msg.embedding.shape)          # (384,)
print(f"Binary size: {len(data)} bytes")  # ~1,590 bytes vs ~5,000+ JSON
```

## Features

- Binary encode/decode of numpy embeddings with protobuf metadata
- Zstandard compression (fast/balanced/max levels)
- HTTP/2 transport client + FastAPI server factory
- 90%+ size reduction vs JSON for typical embeddings

## Examples

```bash
python examples/quickstart.py      # Encode/decode demo
python examples/agent_demo.py      # Two-agent communication
python benchmarks/run_benchmarks.py  # AVP vs JSON benchmarks
```

## Protocol

See the [AVP Specification](https://github.com/VectorArc/avp-spec) for full protocol details.

### Binary Format

```
Bytes 0-1:   Magic (0x4156 = "AV")
Byte 2:      Version (0x01)
Byte 3:      Flags (bit 0 = compressed)
Bytes 4-7:   Payload length (uint32 LE)
Bytes 8-11:  Metadata length (uint32 LE)
Bytes 12+N:  Protobuf metadata (N bytes)
Bytes 12+N+: Raw embedding bytes
```

## License

Apache 2.0
