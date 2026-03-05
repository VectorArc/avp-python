"""Direct single-agent baseline: one agent fixes the buggy code without a chain."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import DIRECT_SOLVE_PROMPT, SYSTEM_MESSAGE, format_examples
from .evaluate import check_correct


def run_direct_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    buggy_code: str,
    solution: str,
    examples: List[str],
    slug: str,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Fix a single DebugBench problem with one agent (no chain)."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()

        examples_text = format_examples(examples)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": DIRECT_SOLVE_PROMPT.format(
                question=question, examples=examples_text, buggy_code=buggy_code,
            )},
        ]

        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])

        text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        wall_time = time.perf_counter() - t0

    output_encoded = tokenizer(text, add_special_tokens=False)
    output_tokens = len(output_encoded["input_ids"])
    total_tokens = prompt_tokens + output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    eval_result = check_correct(text, solution, question, examples)
    correct = eval_result["passed"]

    if verbose:
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {slug}, time={wall_time:.1f}s, "
              f"tokens={total_tokens} ({tokens_per_sec:.1f} tok/s)")
        if eval_result["error"]:
            print(f"  Error: {eval_result['error'][:200]}")
        print(f"  Output: {text[:200]}...")

    return {
        "slug": slug,
        "question": question[:100],
        "raw_output": text,
        "extracted_code": eval_result["code"],
        "correct": correct,
        "error": eval_result["error"],
        "eval_method": eval_result.get("eval_method", "unknown"),
        "wall_time": wall_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "mode": "direct",
    }


def run_direct_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run direct single-agent pipeline on a list of DebugBench samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Direct] Sample {i + 1}/{len(dataset)}: {sample['slug']}")

        result = run_direct_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            buggy_code=sample["buggy_code"],
            solution=sample["solution"],
            examples=sample["examples"],
            slug=sample["slug"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if not verbose:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [Direct] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
