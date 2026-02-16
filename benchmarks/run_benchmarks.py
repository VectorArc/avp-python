#!/usr/bin/env python3
"""Full benchmark suite: AVP vs JSON across dimensions, dtypes, and compression.

Outputs:
  - Markdown table to stdout
  - JSON results to benchmarks/results/benchmark_results.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np

import avp
from avp.utils import embedding_to_json

DIMS = [384, 768, 1024, 4096]
DTYPES = ["float32", "float16"]
ITERATIONS = 1000
RESULTS_DIR = Path(__file__).parent / "results"


def benchmark_size(emb: np.ndarray, dim: int, dtype: str) -> dict:
    """Measure encoded sizes."""
    avp_raw = avp.encode(emb, model_id="bench")
    avp_fast = avp.encode(emb, model_id="bench", compression=avp.CompressionLevel.FAST)
    avp_balanced = avp.encode(emb, model_id="bench", compression=avp.CompressionLevel.BALANCED)
    avp_max = avp.encode(emb, model_id="bench", compression=avp.CompressionLevel.MAX)
    json_data = embedding_to_json(emb, {"model_id": "bench"})

    return {
        "dim": dim,
        "dtype": dtype,
        "avp_raw": len(avp_raw),
        "avp_fast": len(avp_fast),
        "avp_balanced": len(avp_balanced),
        "avp_max": len(avp_max),
        "json": len(json_data),
        "ratio_raw": round(len(json_data) / len(avp_raw), 1),
        "ratio_balanced": round(len(json_data) / len(avp_balanced), 1),
    }


def benchmark_speed(emb: np.ndarray, iterations: int = ITERATIONS) -> dict:
    """Measure encode/decode latency."""
    avp_raw = avp.encode(emb, model_id="bench")

    # Encode (no compression)
    t0 = time.perf_counter()
    for _ in range(iterations):
        avp.encode(emb, model_id="bench")
    encode_us = (time.perf_counter() - t0) / iterations * 1e6

    # Encode (balanced compression)
    t0 = time.perf_counter()
    for _ in range(iterations):
        avp.encode(emb, model_id="bench", compression=avp.CompressionLevel.BALANCED)
    encode_compressed_us = (time.perf_counter() - t0) / iterations * 1e6

    # Decode (no compression)
    t0 = time.perf_counter()
    for _ in range(iterations):
        avp.decode(avp_raw)
    decode_us = (time.perf_counter() - t0) / iterations * 1e6

    # Decode (compressed)
    avp_compressed = avp.encode(emb, model_id="bench", compression=avp.CompressionLevel.BALANCED)
    t0 = time.perf_counter()
    for _ in range(iterations):
        avp.decode(avp_compressed)
    decode_compressed_us = (time.perf_counter() - t0) / iterations * 1e6

    return {
        "encode_us": round(encode_us, 1),
        "encode_compressed_us": round(encode_compressed_us, 1),
        "decode_us": round(decode_us, 1),
        "decode_compressed_us": round(decode_compressed_us, 1),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {"size": [], "speed": []}

    # --- Size benchmarks ---
    print("## Size Comparison: AVP vs JSON\n")
    print(f"| Dim   | dtype   | AVP raw | AVP+zstd | JSON      | Ratio (raw) | Ratio (zstd) |")
    print(f"|-------|---------|---------|----------|-----------|-------------|--------------|")

    for dtype in DTYPES:
        for dim in DIMS:
            np_dtype = np.float32 if dtype == "float32" else np.float16
            emb = np.random.randn(dim).astype(np_dtype)
            result = benchmark_size(emb, dim, dtype)
            all_results["size"].append(result)

            print(f"| {dim:>5} | {dtype:>7} | {result['avp_raw']:>7,} | "
                  f"{result['avp_balanced']:>8,} | {result['json']:>9,} | "
                  f"{result['ratio_raw']:>10.1f}x | {result['ratio_balanced']:>11.1f}x |")

    # --- Speed benchmarks ---
    print(f"\n## Encode/Decode Latency ({ITERATIONS} iterations)\n")
    print(f"| Dim   | dtype   | Encode (µs) | Enc+zstd (µs) | Decode (µs) | Dec+zstd (µs) |")
    print(f"|-------|---------|-------------|---------------|-------------|----------------|")

    for dtype in DTYPES:
        for dim in DIMS:
            np_dtype = np.float32 if dtype == "float32" else np.float16
            emb = np.random.randn(dim).astype(np_dtype)
            speed = benchmark_speed(emb)
            speed["dim"] = dim
            speed["dtype"] = dtype
            all_results["speed"].append(speed)

            print(f"| {dim:>5} | {dtype:>7} | {speed['encode_us']:>11.1f} | "
                  f"{speed['encode_compressed_us']:>13.1f} | {speed['decode_us']:>11.1f} | "
                  f"{speed['decode_compressed_us']:>14.1f} |")

    # --- Save results ---
    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
