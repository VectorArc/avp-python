"""Modal sanity: HuggingFace same-model think → generate + cross-model.

Tests the full HF pipeline after numpy conversion:
- Same-model latent: think() → generate(context=)
- Cross-model rosetta: think() on source → generate(context=, source=, cross_model=True)

Usage:
    modal run benchmarks/modal_hf_sanity.py
"""

import modal

app = modal.App("avp-hf-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers>=5.0",
        "numpy>=1.24",
        "accelerate",
        "sentencepiece",
    )
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface")])
def run_test():
    import time

    results = {}

    # ==============================================================
    # Test 1: Same-model latent (Qwen 1.5B — small, fast)
    # ==============================================================
    print("=" * 60)
    print("TEST 1: Same-model latent (Qwen2.5-1.5B)")
    print("=" * 60)

    from avp import HuggingFaceConnector

    connector = HuggingFaceConnector.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", device="cuda",
    )

    t0 = time.monotonic()
    context = connector.think("Analyze this math problem: 24 * 17 + 3", steps=10)
    think_time = time.monotonic() - t0

    assert context is not None, "think() returned None"
    assert context.last_hidden_state is not None, "No hidden state captured"
    print(f"  think(): {think_time:.2f}s, hidden shape: {list(context.last_hidden_state.shape)}")

    t0 = time.monotonic()
    answer = connector.generate("Solve step by step: 24 * 17 + 3", context=context, max_new_tokens=100)
    gen_time = time.monotonic() - t0

    assert isinstance(answer, str), f"generate() returned {type(answer)}"
    assert len(answer) > 0, "generate() returned empty string"
    has_411 = "411" in answer
    print(f"  generate(): {gen_time:.2f}s, len={len(answer)}, has_411={has_411}")
    print(f"  answer[:200]: {answer[:200]!r}")

    results["same_model"] = {
        "ok": True,
        "think_time": think_time,
        "gen_time": gen_time,
        "has_411": has_411,
        "answer_len": len(answer),
    }

    # ==============================================================
    # Test 2: Cross-model rosetta (Qwen 1.5B → Llama 1B — small)
    # ==============================================================
    print(f"\n{'='*60}")
    print("TEST 2: Cross-model rosetta (Qwen 1.5B → Llama 3.2-1B)")
    print("=" * 60)

    try:
        solver = HuggingFaceConnector.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", device="cuda",
        )

        t0 = time.monotonic()
        ctx = connector.think("Analyze: what is 2 + 2?", steps=10)
        answer = solver.generate(
            "Solve: 2 + 2",
            context=ctx, source=connector, cross_model=True,
            max_new_tokens=50,
        )
        cross_time = time.monotonic() - t0

        assert isinstance(answer, str)
        print(f"  cross-model: {cross_time:.2f}s, len={len(answer)}")
        print(f"  answer[:200]: {answer[:200]!r}")
        has_4 = "4" in answer
        print(f"  has_4={has_4}")

        results["cross_model"] = {
            "ok": True,
            "time": cross_time,
            "has_4": has_4,
            "answer_len": len(answer),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["cross_model"] = {"ok": False, "error": str(e)}

    # ==============================================================
    # Test 3: Easy API (avp.generate one-liner)
    # ==============================================================
    print(f"\n{'='*60}")
    print("TEST 3: Easy API (avp.generate)")
    print("=" * 60)

    import avp

    t0 = time.monotonic()
    answer = avp.generate("What is 3 + 5? Answer with just the number.",
                           model="Qwen/Qwen2.5-1.5B-Instruct")
    easy_time = time.monotonic() - t0

    assert isinstance(answer, str)
    print(f"  easy API: {easy_time:.2f}s, answer[:100]: {answer[:100]!r}")

    results["easy_api"] = {
        "ok": True,
        "time": easy_time,
        "answer_len": len(answer),
    }

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'='*60}")
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
