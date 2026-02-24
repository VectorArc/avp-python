#!/usr/bin/env python3
"""Quick benchmark: AVP vs JSON size and encode/decode speed."""

import time

import numpy as np

import avp
from avp.utils import embedding_to_bytes, embedding_to_json


def main():
    print("=== AVP vs JSON Quick Benchmark ===\n")

    dims = [384, 768, 1024, 4096]
    iterations = 1000

    print(f"{'Dim':>6} | {'AVP':>8} | {'AVP+zstd':>9} | {'JSON':>10} | "
          f"{'Ratio':>6} | {'Enc (µs)':>9} | {'Dec (µs)':>9}")
    print("-" * 75)

    for dim in dims:
        emb = np.random.randn(dim).astype(np.float32)
        payload = embedding_to_bytes(emb)
        metadata = avp.AVPMetadata(
            model_id="bench",
            hidden_dim=dim,
            payload_type=avp.PayloadType.HIDDEN_STATE,
            dtype=avp.DataType.FLOAT32,
            tensor_shape=emb.shape,
        )

        # Sizes
        avp_raw = avp.encode(payload, metadata)
        avp_zstd = avp.encode(payload, metadata, compression=avp.CompressionLevel.BALANCED)
        json_data = embedding_to_json(emb, {"model_id": "bench"})

        # Encode speed
        t0 = time.perf_counter()
        for _ in range(iterations):
            avp.encode(payload, metadata)
        encode_us = (time.perf_counter() - t0) / iterations * 1e6

        # Decode speed
        t0 = time.perf_counter()
        for _ in range(iterations):
            avp.decode(avp_raw)
        decode_us = (time.perf_counter() - t0) / iterations * 1e6

        ratio = len(json_data) / len(avp_raw)

        print(f"{dim:>6} | {len(avp_raw):>7,} | {len(avp_zstd):>8,} | {len(json_data):>9,} | "
              f"{ratio:>5.1f}x | {encode_us:>8.1f} | {decode_us:>8.1f}")

    print(f"\n({iterations} iterations per measurement)")


if __name__ == "__main__":
    main()
