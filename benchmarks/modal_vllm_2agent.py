"""Modal test: 2-agent KV transfer via vLLM.

Agent A: processes prompt with latent thinking steps, KV saved to store.
Agent B: loads Agent A's KV from store, generates answer.

This is the core AVP multi-agent flow (Gate 1 compliant).

Usage:
    modal run benchmarks/modal_vllm_2agent.py
"""

import modal

app = modal.App("avp-vllm-2agent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0,<0.19.0", "torch>=2.0", "transformers>=5.0")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def run_2agent_test():
    import json
    import os
    import tempfile
    from pathlib import Path

    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    results = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Shared store directory for KV transfer
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

        # Single engine instance (both agents use same model)
        engine = vllm.LLM(
            model=MODEL,
            enforce_eager=True,
            max_model_len=256,
            gpu_memory_utilization=0.5,
            kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        # ============================================================
        # Agent A: "Think" about the problem (generate minimal output)
        # ============================================================
        print("\n" + "=" * 60)
        print("AGENT A: Thinking about the problem")
        print("=" * 60)

        prompt_a = "Analyze this math problem carefully: 24 * 17 + 3"
        ids_a = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_a}],
            tokenize=True, add_generation_prompt=True,
        )

        # Agent A generates minimal output (the thinking happens in latent steps)
        params_a = vllm.SamplingParams(max_tokens=1, temperature=0.0)
        outputs_a = engine.generate(
            [vllm.TokensPrompt(prompt_token_ids=list(ids_a))], params_a,
        )

        text_a = outputs_a[0].outputs[0].text
        print(f"Agent A prompt: {prompt_a}")
        print(f"Agent A output: {repr(text_a)} (minimal — thinking was latent)")

        # Check if KV was saved to store
        store_files = list(Path(store_dir).rglob("*.pt"))
        meta_files = list(Path(store_dir).rglob("meta.txt"))
        print(f"Store: {len(store_files)} layer files, {len(meta_files)} meta files")

        results["agent_a"] = {
            "prompt": prompt_a,
            "output": text_a,
            "kv_files": len(store_files),
            "meta_files": len(meta_files),
        }

        if len(store_files) == 0:
            print("ERROR: No KV files saved! Transfer cannot work.")
            results["error"] = "No KV files saved by Agent A"
            return results

        # Read meta to see what was saved
        for meta_file in meta_files:
            content = meta_file.read_text().strip()
            print(f"  Meta ({meta_file.parent.name}): {content}")

        # ============================================================
        # Agent B: Generate answer using Agent A's KV
        # ============================================================
        print("\n" + "=" * 60)
        print("AGENT B: Generating from Agent A's computation")
        print("=" * 60)

        # Agent B uses the SAME prompt (so store key matches)
        prompt_b = prompt_a
        ids_b = list(ids_a)  # Same token IDs

        params_b = vllm.SamplingParams(max_tokens=200, temperature=0.0)
        outputs_b = engine.generate(
            [vllm.TokensPrompt(prompt_token_ids=ids_b)], params_b,
        )

        text_b = outputs_b[0].outputs[0].text
        print(f"Agent B prompt: {prompt_b}")
        print(f"Agent B output: {text_b[:200]}")

        results["agent_b"] = {
            "prompt": prompt_b,
            "output": text_b[:300],
        }

        # ============================================================
        # Baseline: Same prompt, no KV transfer, no latent steps
        # ============================================================
        print("\n" + "=" * 60)
        print("BASELINE: Same prompt, no transfer")
        print("=" * 60)

        del engine

    # Fresh engine without model wrapper
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

        results["baseline"] = {"output": text_bl[:300]}
        del engine_bl

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("2-AGENT TEST SUMMARY")
    print("=" * 60)
    print(f"Agent A KV saved: {results['agent_a']['kv_files']} layer files")
    print(f"Agent B output length: {len(results.get('agent_b', {}).get('output', ''))}")
    print(f"Baseline output length: {len(results.get('baseline', {}).get('output', ''))}")

    has_411 = "411" in results.get("agent_b", {}).get("output", "")
    baseline_411 = "411" in results.get("baseline", {}).get("output", "")
    print(f"Agent B contains '411': {has_411}")
    print(f"Baseline contains '411': {baseline_411}")

    print(f"\n{json.dumps(results, indent=2)}")
    return results


@app.local_entrypoint()
def main():
    results = run_2agent_test.remote()
    import json
    print(json.dumps(results, indent=2))
