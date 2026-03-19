"""Debug: isolate the accuracy degradation on 7B.

Tests multiple configurations on the SAME 5 problems to pinpoint
what causes the accuracy drop.

Configurations:
1. Baseline: no latent steps, no padding, no model wrapper
2. Padding only: 20 pad tokens, NO latent steps, no model wrapper
3. Wrapper only: model wrapper loaded, latent_steps=0, no padding
4. Overwrite: model wrapper + latent_steps=20, NO padding (old overwrite pattern)
5. Extend: model wrapper + latent_steps=20, WITH padding (new extend pattern)

If #2 degrades → pad tokens themselves are the problem
If #3 degrades → the model wrapper changes behavior even without latent steps
If #4 degrades but #3 doesn't → the projection/injection is the problem
If #5 degrades but #4 doesn't → the extend pattern is the problem
"""

import modal

app = modal.App("avp-vllm-debug")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0", "torch>=2.0", "transformers>=4.36", "datasets>=2.14")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=3600)
def run_debug():
    import json
    import tempfile
    import time

    import torch
    import vllm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    N_STEPS = 20
    N_PROBLEMS = 5

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Prepare 5 problems
    questions = [ds[i]["question"] for i in range(N_PROBLEMS)]
    gold = []
    import re
    for i in range(N_PROBLEMS):
        m = re.search(r"####\s*(-?\d[\d,]*)", ds[i]["answer"])
        gold.append(m.group(1).replace(",", "") if m else "")

    # Tokenize with chat template
    base_ids = []
    for q in questions:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve step by step: {q}"}],
            tokenize=True, add_generation_prompt=True,
        )
        base_ids.append(list(ids))

    padded_ids = [ids + [pad_id] * N_STEPS for ids in base_ids]

    params = vllm.SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95, seed=42)

    def eval_outputs(outputs):
        correct = 0
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            nums = re.findall(r"-?\d[\d,]*", text)
            predicted = nums[-1].replace(",", "") if nums else ""
            if predicted == gold[i]:
                correct += 1
        return correct

    results = {}

    # ================================================================
    # Config 1: Baseline (no wrapper, no padding)
    # ================================================================
    print("\n" + "=" * 60)
    print("CONFIG 1: Baseline")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 0, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=2048,
            gpu_memory_utilization=0.85, kv_transfer_config=ktc,
        )
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in base_ids]
        outputs = engine.generate(prompts, params)
        c = eval_outputs(outputs)
        print(f"Baseline: {c}/{N_PROBLEMS}")
        for i, out in enumerate(outputs):
            print(f"  Q{i}: {'✓' if re.findall(r'-?\\d[\\d,]*', out.outputs[0].text)[-1:] == [gold[i]] else '✗'} (gold={gold[i]}, last_num={re.findall(r'-?\\d[\\d,]*', out.outputs[0].text)[-1:]})")
        results["1_baseline"] = c
        del engine

    # ================================================================
    # Config 2: Padding only (20 pad tokens, no wrapper, no latent)
    # ================================================================
    print("\n" + "=" * 60)
    print("CONFIG 2: Padding only (20 pad tokens, no wrapper)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 0, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=2048,
            gpu_memory_utilization=0.85, kv_transfer_config=ktc,
        )
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in padded_ids]
        outputs = engine.generate(prompts, params)
        c = eval_outputs(outputs)
        print(f"Padding only: {c}/{N_PROBLEMS}")
        for i, out in enumerate(outputs):
            print(f"  Q{i}: {'✓' if re.findall(r'-?\\d[\\d,]*', out.outputs[0].text)[-1:] == [gold[i]] else '✗'} (gold={gold[i]}, out={out.outputs[0].text[:80]})")
        results["2_padding_only"] = c
        del engine

    # ================================================================
    # Config 3: Wrapper only (model wrapper, latent_steps=0, no padding)
    # ================================================================
    print("\n" + "=" * 60)
    print("CONFIG 3: Wrapper only (latent_steps=0)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 0, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=2048,
            gpu_memory_utilization=0.85, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
        )
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in base_ids]
        outputs = engine.generate(prompts, params)
        c = eval_outputs(outputs)
        print(f"Wrapper only: {c}/{N_PROBLEMS}")
        results["3_wrapper_only"] = c
        del engine

    # ================================================================
    # Config 4: Extend with padding (current implementation)
    # ================================================================
    print("\n" + "=" * 60)
    print("CONFIG 4: Extend (wrapper + padding + 20 latent steps)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": 20, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=2048,
            gpu_memory_utilization=0.85, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
        )
        # Process one at a time (batch guard)
        outputs = []
        for i, ids in enumerate(padded_ids):
            out = engine.generate([vllm.TokensPrompt(prompt_token_ids=ids)], params)
            outputs.extend(out)
        c = eval_outputs(outputs)
        print(f"Extend: {c}/{N_PROBLEMS}")
        for i, out in enumerate(outputs):
            print(f"  Q{i}: {'✓' if re.findall(r'-?\\d[\\d,]*', out.outputs[0].text)[-1:] == [gold[i]] else '✗'} (gold={gold[i]}, out={out.outputs[0].text[:80]})")
        results["4_extend"] = c
        del engine

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("ISOLATION SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}/{N_PROBLEMS}")

    return results


@app.local_entrypoint()
def main():
    results = run_debug.remote()
    import json
    print(json.dumps(results, indent=2))
