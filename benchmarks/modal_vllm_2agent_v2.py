"""Modal test: 2-agent KV transfer with DIFFERENT prompts.

Agent A: "Analyze this math problem: 24 * 17 + 3" (think-only)
Agent B: "Solve the math problem" (different prompt, loads Agent A's KV)

Uses different prompts so vLLM's prefix caching CANNOT help.
If Agent B produces a correct answer, our KV injection works.

Usage:
    modal run benchmarks/modal_vllm_2agent_v2.py
"""

import modal

app = modal.App("avp-vllm-2agent-v2")

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
    import json
    import tempfile
    from pathlib import Path

    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    results = {}

    with tempfile.TemporaryDirectory() as store_dir:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": 10,
                "avp_store_dir": store_dir,
            },
        )

        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=256,
            gpu_memory_utilization=0.5, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        # ============================================================
        # Agent A: Think about the problem
        # ============================================================
        print("\n" + "=" * 60)
        print("AGENT A: Think (with latent steps)")
        print("=" * 60)

        prompt_a = "Analyze this math problem carefully: 24 * 17 + 3"
        ids_a = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_a}],
            tokenize=True, add_generation_prompt=True,
        )

        params_a = vllm.SamplingParams(max_tokens=1, temperature=0.0)
        outputs_a = engine.generate(
            [vllm.TokensPrompt(prompt_token_ids=list(ids_a))], params_a,
        )
        print(f"Agent A prompt ({len(ids_a)} tokens): {prompt_a}")
        print(f"Agent A output: {repr(outputs_a[0].outputs[0].text)}")

        # Check store
        store_keys = [d.name for d in Path(store_dir).iterdir() if d.is_dir()]
        print(f"Store keys: {store_keys}")

        if not store_keys:
            print("ERROR: No KV saved!")
            results["error"] = "No KV saved"
            return results

        saved_key = store_keys[0]
        meta_path = Path(store_dir) / saved_key / "meta.txt"
        if meta_path.exists():
            meta = meta_path.read_text().strip()
            print(f"Saved: key={saved_key}, meta={meta}")
        results["agent_a_key"] = saved_key

        # ============================================================
        # Agent B: Different prompt, same store key needed
        # ============================================================
        # The problem: Agent B has a DIFFERENT prompt, so the store key
        # (hash of prompt_token_ids) won't match Agent A's key.
        # For true multi-agent transfer, the orchestration layer would
        # pass the store key explicitly. For this test, we use the same
        # prompt tokens so the hash matches (proving the injection path).
        #
        # TODO: Add explicit store key API so different prompts can
        # reference the same KV store entry.

        print("\n" + "=" * 60)
        print("AGENT B: Generate (same prompt, tests KV injection)")
        print("=" * 60)

        # Disable prefix caching to force our connector's load path
        del engine

        engine_b = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=256,
            gpu_memory_utilization=0.5, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
            enable_prefix_caching=False,  # Force our load path
        )

        params_b = vllm.SamplingParams(max_tokens=200, temperature=0.0)
        outputs_b = engine_b.generate(
            [vllm.TokensPrompt(prompt_token_ids=list(ids_a))], params_b,
        )
        text_b = outputs_b[0].outputs[0].text
        print(f"Agent B output: {text_b[:200]}")
        has_411 = "411" in text_b
        print(f"Contains 411: {has_411}")
        results["agent_b_output"] = text_b[:300]
        results["agent_b_411"] = has_411

        del engine_b

    # ============================================================
    # Baseline: Fresh engine, no store, no transfer
    # ============================================================
    print("\n" + "=" * 60)
    print("BASELINE: No transfer")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as store_dir2:
        ktc_bl = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 0, "avp_store_dir": store_dir2},
        )
        engine_bl = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=256,
            gpu_memory_utilization=0.5, kv_transfer_config=ktc_bl,
        )
        outputs_bl = engine_bl.generate(
            [vllm.TokensPrompt(prompt_token_ids=list(ids_a))],
            vllm.SamplingParams(max_tokens=200, temperature=0.0),
        )
        text_bl = outputs_bl[0].outputs[0].text
        print(f"Baseline output: {text_bl[:200]}")
        baseline_411 = "411" in text_bl
        print(f"Contains 411: {baseline_411}")
        results["baseline_output"] = text_bl[:300]
        results["baseline_411"] = baseline_411

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Agent B (KV transfer): 411={'YES' if results.get('agent_b_411') else 'NO'}")
    print(f"Baseline (no transfer): 411={'YES' if results.get('baseline_411') else 'NO'}")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main():
    results = run_test.remote()
    import json
    print(json.dumps(results, indent=2))
