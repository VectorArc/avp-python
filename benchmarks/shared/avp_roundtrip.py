"""AVP codec roundtrip wrapper for benchmarks."""

import time
import uuid
from typing import Any, Tuple


def avp_kv_roundtrip(
    past_kv: Any,
    session_id: str,
    src_agent: str,
    tgt_agent: str,
    model_name: str,
    identity: Any,
    device: str,
) -> Tuple[Any, float, int]:
    """Serialize KV-cache through the full AVP codec and deserialize back.

    Returns (restored_kv, codec_time_ms, wire_bytes).
    """
    from avp.codec import decode as avp_decode
    from avp.codec import encode_kv_cache
    from avp.kv_cache import (
        deserialize_kv_cache,
        legacy_to_dynamic_cache,
        serialize_kv_cache,
    )
    from avp.types import (
        AVPMetadata,
        CommunicationMode,
        DataType,
        PayloadType,
    )

    codec_t0 = time.perf_counter()

    kv_bytes, kv_header = serialize_kv_cache(past_kv)

    metadata = AVPMetadata(
        session_id=session_id,
        source_agent_id=src_agent,
        target_agent_id=tgt_agent,
        model_id=model_name,
        hidden_dim=identity.hidden_dim,
        num_layers=identity.num_layers,
        payload_type=PayloadType.KV_CACHE,
        dtype=DataType.FLOAT32,
        mode=CommunicationMode.LATENT,
    )

    wire_bytes = encode_kv_cache(kv_bytes, metadata)
    wire_size = len(wire_bytes)

    # Decode on "receiving" side
    avp_msg = avp_decode(wire_bytes)
    legacy_kv, _ = deserialize_kv_cache(avp_msg.payload, device=device)
    restored_kv = legacy_to_dynamic_cache(legacy_kv)

    codec_time_ms = (time.perf_counter() - codec_t0) * 1000

    return restored_kv, codec_time_ms, wire_size
