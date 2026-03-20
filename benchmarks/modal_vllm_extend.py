"""Modal test: extend pattern with repeated last token padding.

Tests that the extend-pattern latent loop produces correct output
on 5 GSM8K problems. Compares: baseline (no latent), extend (padded),
and overwrite (no padding, for reference).

Usage:
    modal run benchmarks/modal_vllm_extend.py
"""

import modal

app = modal.App("avp-vllm-extend")

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
def run_test():
    import json
    import re
    import tempfile

    import vllm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    N_STEPS = 20
    N_PROBLEMS = 5

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    num_pat = re.compile(r"-?\d[\d,]*")

    # Prepare problems
    questions = [ds[i]["question"] for i in range(N_PROBLEMS)]
    gold = []
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

    # Padded versions: append N copies of last token
    from avp.connectors.vllm_kv_connector import prepare_latent_prompt
    padded_ids = [prepare_latent_prompt(ids, N_STEPS) for ids in base_ids]

    params = vllm.SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95, seed=42)

    def eval_outputs(outputs):
        correct = 0
        for i, out in enumerate(outputs):
            nums = num_pat.findall(out.outputs[0].text)
            if nums and nums[-1].replace(",", "") == gold[i]:
                correct += 1
        return correct

    results = {}

    # ============================================================
    # Config 1: Baseline (no latent, no padding)
    # ============================================================
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
            text = out.outputs[0].text
            pred = num_pat.findall(text)[-1].replace(",", "") if num_pat.findall(text) else ""
            print(f"  Q{i}: {'Y' if pred == gold[i] else 'N'} (gold={gold[i]}, pred={pred})")
        results["1_baseline"] = c
        del engine

    # ============================================================
    # Config 2: Extend (padded prompt, latent steps)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"CONFIG 2: Extend ({N_STEPS} steps, padded prompt)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={"avp_latent_steps": N_STEPS, "avp_store_dir": td},
        )
        engine = vllm.LLM(
            model=MODEL, enforce_eager=True, max_model_len=2048,
            gpu_memory_utilization=0.85, kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,  # Start with single-request to validate
        )
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in padded_ids]
        outputs = []
        for i, p in enumerate(prompts):
            out = engine.generate([p], params)
            outputs.extend(out)
        c = eval_outputs(outputs)
        print(f"Extend: {c}/{N_PROBLEMS}")
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            pred = num_pat.findall(text)[-1].replace(",", "") if num_pat.findall(text) else ""
            print(f"  Q{i}: {'Y' if pred == gold[i] else 'N'} (gold={gold[i]}, pred={pred}, out={text[:60]})")
        results["2_extend"] = c
        del engine

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}/{N_PROBLEMS}")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main():
    results = run_test.remote()
    import json
    print(json.dumps(results, indent=2))
