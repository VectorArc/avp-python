"""Modal sanity: llama.cpp latent think → generate pipeline.

Tests the full pipeline: cb_eval captures hidden state during think(),
batch.embd injects it during generate(). Uses a small GGUF model
(Qwen2.5-1.5B Q4_K_M) for fast validation.

Usage:
    modal run benchmarks/modal_llamacpp_sanity.py
"""

import modal

app = modal.App("avp-llamacpp-sanity")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgomp1")
    .pip_install(
        "llama-cpp-python>=0.3",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "huggingface-hub>=0.20",
        "gguf>=0.6",
    )
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def run_test():
    import time

    from huggingface_hub import hf_hub_download

    # Download a small GGUF model
    MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    print("=" * 60)
    print(f"Downloading {MODEL_FILE}...")
    print("=" * 60)

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model path: {model_path}")

    results = {}

    # ==============================================================
    # Test 1: Basic generation (no latent, verify model works)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Basic generation (no latent)")
    print("=" * 60)

    from avp.connectors.llamacpp import LlamaCppConnector

    connector = LlamaCppConnector.from_pretrained(
        model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False,
    )

    t0 = time.monotonic()
    text = connector.generate(
        "What is 2 + 2? Answer with just the number.",
        max_tokens=20,
        temperature=0.0,
    )
    elapsed = time.monotonic() - t0
    print(f"  Output: {text!r}")
    print(f"  Time: {elapsed:.2f}s")
    results["basic"] = {"text": text[:100], "ok": len(text) > 0}

    # ==============================================================
    # Test 1b: GGUF weight extraction diagnostic
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 1b: GGUF embed weight extraction")
    print("=" * 60)
    try:
        from avp.connectors._llamacpp_compat import extract_gguf_embedding_weights
        emb_w = extract_gguf_embedding_weights(model_path)
        print(f"  embed_tokens shape: {emb_w.shape}")
        import numpy as np
        print(f"  embed_tokens norm (mean): {np.linalg.norm(emb_w, axis=1).mean():.3f}")
        results["gguf_extract"] = {"shape": list(emb_w.shape), "ok": True}
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["gguf_extract"] = {"ok": False, "error": str(e)}

    # ==============================================================
    # Test 2: Think (cb_eval captures hidden state)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Think (hidden state extraction via cb_eval)")
    print("=" * 60)

    t0 = time.monotonic()
    context = connector.think(
        "Analyze this math problem carefully: 24 * 17 + 3",
        steps=1,
    )
    elapsed = time.monotonic() - t0

    if context is not None:
        hidden = getattr(context, "last_hidden_state", None)
        if hidden is not None:
            print(f"  Hidden state shape: {list(hidden.shape)}")
            print(f"  Hidden state norm: {hidden.float().norm():.3f}")
            print(f"  Time: {elapsed:.2f}s")
            results["think"] = {
                "shape": list(hidden.shape),
                "norm": hidden.float().norm().item(),
                "ok": True,
            }
        else:
            print("  WARNING: context has no hidden_state")
            results["think"] = {"ok": False, "error": "no hidden_state"}
    else:
        print("  WARNING: think() returned None")
        results["think"] = {"ok": False, "error": "returned None"}

    # ==============================================================
    # Test 3: Generate with latent context (batch.embd injection)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Generate with latent context (embedding injection)")
    print("=" * 60)

    if context is not None and getattr(context, "last_hidden_state", None) is not None:
        t0 = time.monotonic()
        answer = connector.generate(
            "Solve step by step: 24 * 17 + 3",
            context=context,
            max_tokens=100,
            temperature=0.0,
        )
        elapsed = time.monotonic() - t0
        print(f"  Answer: {answer[:200]!r}")
        print(f"  Time: {elapsed:.2f}s")
        has_411 = "411" in answer
        print(f"  Contains 411: {has_411}")
        results["generate_latent"] = {
            "text": answer[:200],
            "has_411": has_411,
            "ok": len(answer) > 0,
        }
    else:
        print("  SKIP: No context from think()")
        results["generate_latent"] = {"ok": False, "error": "no context"}

    # ==============================================================
    # Test 4: Text-only baseline (same problem, no latent)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Text baseline (no latent)")
    print("=" * 60)

    t0 = time.monotonic()
    baseline = connector.generate(
        "Solve step by step: 24 * 17 + 3",
        max_tokens=100,
        temperature=0.0,
    )
    elapsed = time.monotonic() - t0
    print(f"  Answer: {baseline[:200]!r}")
    print(f"  Time: {elapsed:.2f}s")
    has_411 = "411" in baseline
    print(f"  Contains 411: {has_411}")
    results["baseline"] = {
        "text": baseline[:200],
        "has_411": has_411,
        "ok": len(baseline) > 0,
    }

    # ==============================================================
    # Summary
    # ==============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, data in results.items():
        status = "OK" if data.get("ok") else "FAILED"
        print(f"  {name}: {status}")

    return results


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
