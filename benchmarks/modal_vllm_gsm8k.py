"""Modal benchmark: vLLM latent thinking vs baseline on GSM8K.

Compares vLLM extend-pattern latent steps against baseline (no latent)
on Qwen2.5-7B-Instruct. Measures accuracy, latency, and memory.

Usage:
    modal run benchmarks/modal_vllm_gsm8k.py
    modal run benchmarks/modal_vllm_gsm8k.py --n-problems 200
"""

import modal

app = modal.App("avp-vllm-gsm8k")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm>=0.17.0",
        "torch>=2.0",
        "transformers>=4.36",
        "datasets>=2.14",
    )
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


def extract_gsm8k_answer(text: str) -> str:
    """Extract the numeric answer from GSM8K model output."""
    import re
    # Look for #### pattern (GSM8K standard)
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    # Look for "answer is X" pattern
    match = re.search(r"(?:answer|result)\s+(?:is|=)\s+\\?[({]?\s*(-?\d[\d,]*)", text, re.I)
    if match:
        return match.group(1).replace(",", "")
    # Look for boxed answer \boxed{X}
    match = re.search(r"\\boxed\{(-?\d[\d,]*)\}", text)
    if match:
        return match.group(1).replace(",", "")
    # Last number in the text
    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract gold answer from GSM8K dataset."""
    import re
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours for 7B model
)
def run_gsm8k_benchmark(n_problems: int = 50, latent_steps: int = 20):
    """Run GSM8K benchmark comparing vLLM latent vs baseline."""
    import json
    import os
    import tempfile
    import time

    import torch
    import vllm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    N = latent_steps

    print(f"{'=' * 60}")
    print(f"GSM8K Benchmark: {MODEL}")
    print(f"n_problems={n_problems}, latent_steps={N}")
    print(f"{'=' * 60}")

    # Load dataset and tokenizer
    ds = load_dataset("openai/gsm8k", "main", split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Prepare prompts
    problems = []
    gold_answers = []
    for i in range(min(n_problems, len(ds))):
        question = ds[i]["question"]
        gold = extract_gsm8k_gold(ds[i]["answer"])
        problems.append(question)
        gold_answers.append(gold)

    # Format with chat template
    formatted_ids = []
    for q in problems:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve step by step: {q}"}],
            tokenize=True, add_generation_prompt=True,
        )
        formatted_ids.append(list(ids))

    # Padded versions (for latent steps)
    padded_ids = [ids + [pad_id] * N for ids in formatted_ids]

    print(f"Loaded {len(problems)} problems")
    print(f"Avg prompt length: {sum(len(ids) for ids in formatted_ids) / len(formatted_ids):.0f} tokens")
    print(f"Avg padded length: {sum(len(ids) for ids in padded_ids) / len(padded_ids):.0f} tokens")

    results = {}

    # ================================================================
    # Phase 1: Baseline (no latent steps, no padding, no model wrapper)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: Baseline (no latent steps)")
    print(f"{'=' * 60}")

    # Memory before
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9

    with tempfile.TemporaryDirectory() as tmpdir:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": 0,
                "avp_store_dir": tmpdir,
            },
        )
        engine = vllm.LLM(
            model=MODEL,
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            kv_transfer_config=ktc,
        )

        params = vllm.SamplingParams(
            max_tokens=512, temperature=0.7, top_p=0.95, seed=42,
        )

        # Run baseline
        t0 = time.time()
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in formatted_ids]
        outputs = engine.generate(prompts, params)
        baseline_time = time.time() - t0

        baseline_mem_peak = torch.cuda.max_memory_allocated() / 1e9

        # Evaluate
        baseline_correct = 0
        baseline_texts = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            baseline_texts.append(text)
            predicted = extract_gsm8k_answer(text)
            if predicted == gold_answers[i]:
                baseline_correct += 1

        baseline_acc = baseline_correct / len(problems) * 100
        baseline_per_problem = baseline_time / len(problems)

        print(f"Baseline accuracy: {baseline_correct}/{len(problems)} = {baseline_acc:.1f}%")
        print(f"Baseline time: {baseline_time:.1f}s ({baseline_per_problem:.2f}s/problem)")
        print(f"Baseline peak GPU memory: {baseline_mem_peak:.2f} GB")

        del engine

    results["baseline"] = {
        "accuracy": baseline_acc,
        "correct": baseline_correct,
        "total": len(problems),
        "time_s": baseline_time,
        "per_problem_s": baseline_per_problem,
        "peak_gpu_gb": baseline_mem_peak,
    }

    # ================================================================
    # Phase 2: Latent (extend pattern with dummy padding + model wrapper)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Latent ({N} steps, extend pattern)")
    print(f"{'=' * 60}")

    torch.cuda.reset_peak_memory_stats()

    with tempfile.TemporaryDirectory() as tmpdir:
        ktc = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": N,
                "avp_store_dir": tmpdir,
            },
        )
        engine = vllm.LLM(
            model=MODEL,
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            kv_transfer_config=ktc,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
        )

        params = vllm.SamplingParams(
            max_tokens=512, temperature=0.7, top_p=0.95, seed=42,
        )

        # Run latent
        t0 = time.time()
        prompts = [vllm.TokensPrompt(prompt_token_ids=ids) for ids in padded_ids]
        outputs = engine.generate(prompts, params)
        latent_time = time.time() - t0

        latent_mem_peak = torch.cuda.max_memory_allocated() / 1e9

        # Evaluate
        latent_correct = 0
        latent_texts = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            latent_texts.append(text)
            predicted = extract_gsm8k_answer(text)
            if predicted == gold_answers[i]:
                latent_correct += 1

        latent_acc = latent_correct / len(problems) * 100
        latent_per_problem = latent_time / len(problems)

        print(f"Latent accuracy: {latent_correct}/{len(problems)} = {latent_acc:.1f}%")
        print(f"Latent time: {latent_time:.1f}s ({latent_per_problem:.2f}s/problem)")
        print(f"Latent peak GPU memory: {latent_mem_peak:.2f} GB")

        del engine

    results["latent"] = {
        "accuracy": latent_acc,
        "correct": latent_correct,
        "total": len(problems),
        "time_s": latent_time,
        "per_problem_s": latent_per_problem,
        "peak_gpu_gb": latent_mem_peak,
        "latent_steps": N,
    }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Latent':>12} {'Delta':>12}")
    print("-" * 66)
    print(f"{'Accuracy':<30} {baseline_acc:>11.1f}% {latent_acc:>11.1f}% {latent_acc - baseline_acc:>+11.1f}%")
    print(f"{'Time (total)':<30} {baseline_time:>11.1f}s {latent_time:>11.1f}s {latent_time - baseline_time:>+11.1f}s")
    print(f"{'Time (per problem)':<30} {baseline_per_problem:>11.2f}s {latent_per_problem:>11.2f}s {latent_per_problem - baseline_per_problem:>+11.2f}s")
    print(f"{'Peak GPU memory':<30} {baseline_mem_peak:>10.2f}GB {latent_mem_peak:>10.2f}GB {latent_mem_peak - baseline_mem_peak:>+10.2f}GB")

    print(f"\nPublished HF numbers (Qwen 7B, n=200, seed=42):")
    print(f"  Direct: 91.0%, Latent: 90.5%, Text Chain: 87.0%")
    print(f"  Latent steps: 0.9s/problem (45ms/step)")
    print(f"  KV-cache: ~390 MB for 7B")

    results["comparison"] = {
        "accuracy_delta_pp": latent_acc - baseline_acc,
        "time_overhead_s": latent_time - baseline_time,
        "memory_overhead_gb": latent_mem_peak - baseline_mem_peak,
        "hf_published_latent_acc": 90.5,
        "hf_published_baseline_acc": 91.0,
    }

    print(f"\n{json.dumps(results, indent=2)}")
    return results


@app.local_entrypoint()
def main(n_problems: int = 50):
    results = run_gsm8k_benchmark.remote(n_problems=n_problems)
    import json
    print(f"\n{'=' * 60}")
    print("RESULTS (from Modal)")
    print(f"{'=' * 60}")
    print(json.dumps(results, indent=2))
