"""Modal sanity: Verify latent thinking works with CUDA graphs enabled.

Production vLLM defaults to CUDA graphs ON. Our benchmarks used
enforce_eager=True. This test runs WITHOUT enforce_eager to confirm
the latent loop works with piecewise CUDA graph capture.

Usage:
    modal run benchmarks/modal_vllm_cudagraph_check.py
"""

import modal

app = modal.App("avp-vllm-cudagraph-check")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0", "torch>=2.0", "transformers>=4.36")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def run_test():
    import os
    import tempfile
    import time

    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from avp.connectors.vllm_kv_connector import prepare_latent_prompt

    prompt = "Solve step by step: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    padded = prepare_latent_prompt(list(ids), 10)

    results = {}

    # ==============================================================
    # Test 1: enforce_eager=True (our standard, no CUDA graphs)
    # ==============================================================
    print("=" * 60)
    print("TEST 1: enforce_eager=True (no CUDA graphs)")
    print("=" * 60)

    os.environ["AVP_LATENT_STEPS"] = "10"
    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 10, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=512,
            gpu_memory_utilization=0.7, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )
        params = vllm.SamplingParams(max_tokens=100, temperature=0.0)
        t0 = time.monotonic()
        out = engine.generate([vllm.TokensPrompt(prompt_token_ids=padded)], params)
        elapsed = time.monotonic() - t0
        answer = out[0].outputs[0].text.strip()
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Answer: {answer[:150]}")
        results["eager"] = {"time": elapsed, "answer": answer[:150], "ok": len(answer) > 10}
        del engine

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # ==============================================================
    # Test 2: CUDA graphs ENABLED (production default)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 2: CUDA graphs ENABLED (production default)")
    print("=" * 60)

    os.environ["AVP_LATENT_STEPS"] = "10"
    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 10, "avp_store_dir": td},
        )
        try:
            engine = vllm.LLM(
                model=MODEL,
                # NO enforce_eager — CUDA graphs are ON
                max_model_len=512,
                gpu_memory_utilization=0.7,
                kv_transfer_config=ktc,
                hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
                max_num_seqs=1,
            )
            params = vllm.SamplingParams(max_tokens=100, temperature=0.0)
            t0 = time.monotonic()
            out = engine.generate([vllm.TokensPrompt(prompt_token_ids=padded)], params)
            elapsed = time.monotonic() - t0
            answer = out[0].outputs[0].text.strip()
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Answer: {answer[:150]}")
            results["cudagraph"] = {"time": elapsed, "answer": answer[:150], "ok": len(answer) > 10}
            del engine
        except Exception as e:
            print(f"  FAILED: {e}")
            results["cudagraph"] = {"error": str(e), "ok": False}

    # ==============================================================
    # Summary
    # ==============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode, data in results.items():
        status = "OK" if data.get("ok") else "FAILED"
        print(f"  {mode}: {status}")
    return results


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print(json.dumps(result, indent=2, default=str))
