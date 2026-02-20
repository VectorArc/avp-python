"""Direct single-agent baseline: one agent solves the problem without a chain."""

import time
from typing import Any, Dict, List

import torch

from .agents import generate_text, render_prompt, tokenize_prompt
from .evaluate import extract_gold, extract_gsm8k_answer


DIRECT_SOLVE_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your work clearly, then give the final numeric answer "
    "inside \\boxed{{}}.\n\n"
    "Question: {question}\n\n"
    "Solution:"
)


def run_direct_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    gold_solution: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Solve a single GSM8K problem with one agent (no chain)."""
    t0 = time.perf_counter()

    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": DIRECT_SOLVE_PROMPT.format(question=question)},
    ]

    prompt_text = render_prompt(tokenizer, messages)
    input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

    text, _ = generate_text(
        model, tokenizer, input_ids, attention_mask, device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    wall_time = time.perf_counter() - t0
    gold = extract_gold(gold_solution)
    prediction = extract_gsm8k_answer(text)

    from .evaluate import check_correct
    correct = check_correct(prediction, gold)

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{status}] pred={prediction}, gold={gold}, time={wall_time:.1f}s")
        print(f"  Output: {text[:200]}...")

    return {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "raw_output": text,
        "correct": correct,
        "wall_time": wall_time,
        "prompt_tokens": int(input_ids.shape[-1]),
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
    """Run direct single-agent pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Direct] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_direct_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            gold_solution=sample["answer"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

    return results
