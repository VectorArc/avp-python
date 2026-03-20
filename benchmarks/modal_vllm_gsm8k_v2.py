"""GSM8K benchmark: single-engine vLLM latent thinking vs baseline.

Single engine runs both latent (with extend-pattern padding) and
baseline (no latent). Tests that latent thinking improves accuracy
within one vLLM instance (no multi-agent KV transfer).

Usage:
    modal run benchmarks/modal_vllm_gsm8k_v2.py
"""

import modal

app = modal.App("avp-vllm-gsm8k-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0", "torch>=2.0", "transformers>=4.36", "datasets>=2.14")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=7200)
def run_benchmark(n_problems: int = 200):
    import json
    import re
    import tempfile
    import time

    import torch
    import vllm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    num_pattern = re.compile(r"-?\d[\d,]*")

    print(f"{'=' * 60}")
    print(f"GSM8K Benchmark: {MODEL}, n={n_problems}")
    print(f"{'=' * 60}")

    # Load dataset and tokenizer
    ds = load_dataset("openai/gsm8k", "main", split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Prepare problems with chat template
    questions, gold_answers, formatted_ids = [], [], []
    for i in range(min(n_problems, len(ds))):
        questions.append(ds[i]["question"])
        m = re.search(r"####\s*(-?\d[\d,]*)", ds[i]["answer"])
        gold_answers.append(m.group(1).replace(",", "") if m else "")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve step by step: {ds[i]['question']}"}],
            tokenize=True, add_generation_prompt=True,
        )
        formatted_ids.append(list(ids))

    avg_prompt_len = sum(len(ids) for ids in formatted_ids) / len(formatted_ids)
    print(f"Loaded {len(questions)} problems, avg prompt: {avg_prompt_len:.0f} tokens")

    def evaluate(outputs):
        correct, total_gen_tokens = 0, 0
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            total_gen_tokens += len(out.outputs[0].token_ids)
            nums = num_pattern.findall(text)
            if nums and nums[-1].replace(",", "") == gold_answers[i]:
                correct += 1
        return correct, total_gen_tokens

    params = vllm.SamplingParams(
        max_tokens=512, temperature=0.7, top_p=0.95, seed=42,
    )
    prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in formatted_ids]

    results = {}

    # ================================================================
    # Phase 1: Baseline (no latent steps, batched)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: Baseline (no latent steps)")
    print(f"{'=' * 60}")

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

        t0 = time.time()
        outputs = engine.generate(prompts, params)
        baseline_time = time.time() - t0

        baseline_correct, baseline_tokens = evaluate(outputs)
        baseline_acc = baseline_correct / len(questions) * 100

        print(f"Baseline: {baseline_correct}/{len(questions)} = {baseline_acc:.1f}%")
        print(f"Time: {baseline_time:.1f}s ({baseline_time/len(questions):.2f}s/problem)")
        print(f"Generated tokens: {baseline_tokens} ({baseline_tokens/len(questions):.0f}/problem)")

        results["baseline"] = {
            "accuracy": baseline_acc,
            "correct": baseline_correct,
            "total": len(questions),
            "time_s": baseline_time,
            "gen_tokens": baseline_tokens,
        }
        del engine

    # ================================================================
    # Phase 2: Latent (extend pattern, padded prompts)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: Latent (extend, 20 steps)")
    print(f"{'=' * 60}")

    import os
    os.environ["AVP_LATENT_STEPS"] = "20"

    from avp.connectors.vllm_kv_connector import prepare_latent_prompt
    padded_ids = [prepare_latent_prompt(ids, 20) for ids in formatted_ids]
    padded_prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in padded_ids]

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

        t0 = time.time()
        outputs = engine.generate(padded_prompts, params)
        latent_time = time.time() - t0

        latent_correct, latent_tokens = evaluate(outputs)
        latent_acc = latent_correct / len(questions) * 100

        print(f"Latent: {latent_correct}/{len(questions)} = {latent_acc:.1f}%")
        print(f"Time: {latent_time:.1f}s ({latent_time/len(questions):.2f}s/problem)")
        print(f"Generated tokens: {latent_tokens} ({latent_tokens/len(questions):.0f}/problem)")

        results["latent"] = {
            "accuracy": latent_acc,
            "correct": latent_correct,
            "total": len(questions),
            "time_s": latent_time,
            "gen_tokens": latent_tokens,
            "latent_steps": 20,
        }
        del engine

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    delta_acc = latent_acc - baseline_acc
    latent_overhead = latent_time - baseline_time

    print(f"\n{'Metric':<35} {'Baseline':>12} {'Latent':>12} {'Delta':>12}")
    print("-" * 71)
    print(f"{'Accuracy':<35} {baseline_acc:>11.1f}% {latent_acc:>11.1f}% {delta_acc:>+11.1f}%")
    print(f"{'Time (total)':<35} {baseline_time:>11.1f}s {latent_time:>11.1f}s {latent_overhead:>+11.1f}s")
    print(f"{'Time (per problem)':<35} {baseline_time/len(questions):>11.2f}s {latent_time/len(questions):>11.2f}s {latent_overhead/len(questions):>+11.2f}s")
    print(f"{'Generated tokens':<35} {baseline_tokens:>11d} {latent_tokens:>11d} {latent_tokens-baseline_tokens:>+11d}")
    print(f"{'Tokens/problem':<35} {baseline_tokens/len(questions):>11.0f} {latent_tokens/len(questions):>11.0f}")

    print(f"\nPublished HF numbers (Qwen 7B, n=200, seed=42, temp=0.7):")
    print(f"  Direct: 91.0%, Latent (extend): 90.5%, Text Chain: 87.0%")
    print(f"  Latent step time: 0.9s (45ms/step × 20 steps)")

    print(f"\nvLLM vs HF latent step comparison:")
    print(f"  vLLM latent overhead: {latent_overhead:.1f}s total = {latent_overhead/len(questions):.2f}s/problem")
    est_step_time = latent_overhead / len(questions) / 20 * 1000 if latent_overhead > 0 else 0
    print(f"  Estimated per-step: ~{est_step_time:.0f}ms (HF: 45ms)")

    results["summary"] = {
        "accuracy_delta_pp": delta_acc,
        "latent_overhead_s": latent_overhead,
        "per_problem_overhead_s": latent_overhead / len(questions),
        "hf_latent_acc": 90.5,
        "hf_baseline_acc": 91.0,
    }

    print(f"\n{json.dumps(results, indent=2)}")
    return results


@app.local_entrypoint()
def main(n_problems: int = 200):
    results = run_benchmark.remote(n_problems=n_problems)
    import json
    print(json.dumps(results, indent=2))
