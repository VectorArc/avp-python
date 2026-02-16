#!/usr/bin/env python3
"""Quickstart: encode/decode an embedding and compare size to JSON."""

import numpy as np

import avp
from avp.utils import embedding_to_json


def main():
    print("=== AVP Quickstart ===\n")

    # Create a random embedding (simulating a 384-dim model output)
    embedding = np.random.randn(384).astype(np.float32)
    print(f"Embedding: {embedding.shape} {embedding.dtype}")

    # --- Encode as AVP binary ---
    avp_data = avp.encode(
        embedding,
        model_id="all-MiniLM-L6-v2",
        agent_id="quickstart-agent",
    )
    print(f"\nAVP binary size:        {len(avp_data):,} bytes")

    # --- Encode with compression ---
    avp_compressed = avp.encode(
        embedding,
        model_id="all-MiniLM-L6-v2",
        compression=avp.CompressionLevel.BALANCED,
    )
    print(f"AVP compressed size:    {len(avp_compressed):,} bytes")

    # --- JSON baseline ---
    json_data = embedding_to_json(embedding, {"model_id": "all-MiniLM-L6-v2"})
    print(f"JSON size:              {len(json_data):,} bytes")

    # --- Size comparison ---
    ratio = len(json_data) / len(avp_data)
    compressed_ratio = len(json_data) / len(avp_compressed)
    print(f"\nAVP is {ratio:.1f}x smaller than JSON")
    print(f"AVP+zstd is {compressed_ratio:.1f}x smaller than JSON")

    # --- Decode and verify ---
    msg = avp.decode(avp_data)
    print(f"\nDecoded: model={msg.metadata.model_id}, dim={msg.metadata.embedding_dim}")
    print(f"Roundtrip exact: {np.array_equal(embedding, msg.embedding)}")

    # --- Convenience API ---
    print("\n--- Convenience API ---")
    data = avp.encode_simple(embedding, model_id="test", compress=True)
    arr, meta = avp.decode_simple(data)
    print(f"encode_simple -> {len(data)} bytes")
    print(f"decode_simple -> shape={arr.shape}, meta={meta}")


if __name__ == "__main__":
    main()
