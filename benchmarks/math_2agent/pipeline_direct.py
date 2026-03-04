"""Direct single-agent baseline: one agent solves the MATH problem without a chain."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import DIRECT_SOLVE_PROMPT, SYSTEM_MESSAGE
from .evaluate import extract_gold, extract_answer, check_correct


def run_direct_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    problem: str,
    gold_answer: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Solve a single MATH problem with one agent (no chain)."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": DIRECT_SOLVE_PROMPT.format(problem=problem)},
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

    gold = extract_gold(gold_answer)
    prediction = extract_answer(text)
    correct = check_correct(prediction, gold)

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] pred={prediction}, gold={gold}, time={wall_time:.1f}s, "
              f"tokens={total_tokens} ({tokens_per_sec:.1f} tok/s)")
        print(f"  Output: {text[:200]}...")

    return {
        "problem": problem,
        "gold": gold,
        "prediction": prediction,
        "raw_output": text,
        "correct": correct,
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
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run direct single-agent pipeline on a list of MATH samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Direct] Sample {i + 1}/{len(dataset)}: {sample['problem'][:80]}...")

        result = run_direct_pipeline(
            model, tokenizer, device,
            problem=sample["problem"],
            gold_answer=sample["answer"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        result["level"] = sample.get("level")
        result["subject"] = sample.get("subject")
        results.append(result)

    return results
