"""Modal runner for vLLM integration tests.

Runs the latent thinking model plugin + KV connector on an A100 GPU
with a real vLLM instance and Qwen2.5-0.5B-Instruct.

Usage:
    modal run benchmarks/modal_vllm_integration.py
"""

import modal

app = modal.App("avp-vllm-integration")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm>=0.17.0",
        "torch>=2.0",
        "transformers>=4.36",
    )
    # Install avp from the engine_integration branch
    # Cache bust: bump this comment to force re-install: v5
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
)
def run_vllm_integration_tests():
    """Run vLLM integration tests on A100."""
    import os
    import tempfile
    import traceback

    results = {}

    # ---------------------------------------------------------------
    # Test 1: Latent thinking produces non-degenerate output
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: Latent thinking end-to-end")
    print("=" * 60)

    try:
        import vllm
        from vllm.config import KVTransferConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            ktc = KVTransferConfig(
                kv_connector="AVPKVConnectorV1Dynamic",
                kv_connector_module_path="avp.connectors.vllm_kv_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "avp_latent_steps": 20,
                    "avp_store_dir": tmpdir,
                },
            )
            engine = vllm.LLM(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                enforce_eager=True,
                max_model_len=256,
                gpu_memory_utilization=0.5,
                kv_transfer_config=ktc,
                hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            )

            # Pad prompt with N dummy tokens for the extend pattern.
            # The model plugin expects num_tokens = L + N where the last N
            # positions are dummy tokens whose KV entries get overwritten
            # by latent steps.
            from transformers import AutoTokenizer
            N_STEPS = 20
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            prompt_text = "Solve step by step: 24 * 17 + 3"
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            padded_ids = list(prompt_ids) + [pad_id] * N_STEPS

            print(f"Prompt tokens: {len(prompt_ids)}, padded: {len(padded_ids)} (+{N_STEPS} dummy)")

            params = vllm.SamplingParams(max_tokens=100, temperature=0.0)
            outputs = engine.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded_ids)], params
            )

            text = outputs[0].outputs[0].text
            print(f"Output: {text[:200]}")
            print(f"Output length: {len(text)} chars")
            has_digits = any(c.isdigit() for c in text)
            print(f"Contains digits: {has_digits}")

            results["test1_latent_output"] = {
                "status": "PASS" if len(text) > 0 else "FAIL",
                "output_preview": text[:200],
                "output_len": len(text),
            }

            # ---------------------------------------------------------------
            # Test 2: KV files were written to store
            # ---------------------------------------------------------------
            print("\n" + "=" * 60)
            print("TEST 2: KV store file creation")
            print("=" * 60)

            from pathlib import Path

            store_files = list(Path(tmpdir).rglob("*.pt"))
            meta_files = list(Path(tmpdir).rglob("meta.txt"))
            print(f"Store .pt files: {len(store_files)}")
            print(f"Store meta files: {len(meta_files)}")

            results["test2_kv_store"] = {
                "status": "PASS" if len(store_files) > 0 else "FAIL",
                "pt_files": len(store_files),
                "meta_files": len(meta_files),
            }

            # ---------------------------------------------------------------
            # Test 3: Baseline comparison (no latent steps)
            # ---------------------------------------------------------------
            print("\n" + "=" * 60)
            print("TEST 3: Baseline comparison (latent_steps=0)")
            print("=" * 60)

            del engine

        # Restart without latent steps
        with tempfile.TemporaryDirectory() as tmpdir2:
            ktc_baseline = KVTransferConfig(
                kv_connector="AVPKVConnectorV1Dynamic",
                kv_connector_module_path="avp.connectors.vllm_kv_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "avp_latent_steps": 0,
                    "avp_store_dir": tmpdir2,
                },
            )
            engine_baseline = vllm.LLM(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                enforce_eager=True,
                max_model_len=256,
                gpu_memory_utilization=0.5,
                kv_transfer_config=ktc_baseline,
            )

            # Use same chat-templated prompt as Test 1, but WITHOUT padding
            from transformers import AutoTokenizer
            tokenizer_bl = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            baseline_ids = tokenizer_bl.apply_chat_template(
                [{"role": "user", "content": "Solve step by step: 24 * 17 + 3"}],
                tokenize=True, add_generation_prompt=True,
            )
            print(f"Baseline tokens: {len(baseline_ids)} (no padding)")

            params = vllm.SamplingParams(max_tokens=100, temperature=0.0)
            outputs_baseline = engine_baseline.generate(
                [vllm.TokensPrompt(prompt_token_ids=list(baseline_ids))], params
            )

            text_baseline = outputs_baseline[0].outputs[0].text
            print(f"Baseline output: {text_baseline[:200]}")
            print(f"Baseline length: {len(text_baseline)} chars")

            results["test3_baseline"] = {
                "status": "PASS",
                "output_preview": text_baseline[:200],
                "output_len": len(text_baseline),
            }

            del engine_baseline

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        results["error"] = str(e)

    # ---------------------------------------------------------------
    # Test 4: Multiple prompts (verify no cross-contamination)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 4: Multiple prompts")
    print("=" * 60)

    try:
        import vllm

        with tempfile.TemporaryDirectory() as tmpdir3:
            ktc_multi = KVTransferConfig(
                kv_connector="AVPKVConnectorV1Dynamic",
                kv_connector_module_path="avp.connectors.vllm_kv_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "avp_latent_steps": 20,
                    "avp_store_dir": tmpdir3,
                },
            )
            engine_multi = vllm.LLM(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                enforce_eager=True,
                max_model_len=256,
                gpu_memory_utilization=0.5,
                kv_transfer_config=ktc_multi,
                hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            )

            # Pad both prompts with N dummy tokens
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            prompts_text = [
                "What is 2+2? Answer with just the number:",
                "What is the capital of France? Answer in one word:",
            ]
            padded_prompts = []
            for p in prompts_text:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True, add_generation_prompt=True,
                )
                padded_prompts.append(
                    vllm.TokensPrompt(prompt_token_ids=list(ids) + [pad_id] * N_STEPS)
                )

            params = vllm.SamplingParams(max_tokens=50, temperature=0.0)
            outputs = engine_multi.generate(
                padded_prompts,
                params,
            )

            text0 = outputs[0].outputs[0].text
            text1 = outputs[1].outputs[0].text
            print(f"Prompt 1 output: {text0[:100]}")
            print(f"Prompt 2 output: {text1[:100]}")

            results["test4_multi_prompt"] = {
                "status": "PASS" if len(text0) > 0 and len(text1) > 0 else "FAIL",
                "output_0": text0[:100],
                "output_1": text1[:100],
            }

            del engine_multi

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        results["test4_error"] = str(e)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    import json

    print(json.dumps(results, indent=2))

    all_pass = all(
        v.get("status") == "PASS"
        for v in results.values()
        if isinstance(v, dict) and "status" in v
    )
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    return results


@app.local_entrypoint()
def main():
    results = run_vllm_integration_tests.remote()
    import json

    print("\n" + "=" * 60)
    print("RESULTS (from Modal)")
    print("=" * 60)
    print(json.dumps(results, indent=2))
