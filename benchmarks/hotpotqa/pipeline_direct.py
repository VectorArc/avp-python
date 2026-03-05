"""Direct single-agent baseline for HotpotQA."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import DIRECT_PROMPT, SYSTEM_MESSAGE, format_paragraphs
from .evaluate import check_correct, exact_match, extract_answer, token_f1


def run_direct_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    gold_answer: str,
    paragraphs: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Solve a single HotpotQA problem with one agent."""
    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": DIRECT_PROMPT.format(
                paragraphs=paragraphs_text, question=question
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

    prediction = extract_answer(text)
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)

    if verbose:
        status = "CORRECT" if em else "WRONG"
        print(f"  [{status}] pred='{prediction}', gold='{gold_answer}', "
              f"F1={f1:.2f}, time={wall_time:.1f}s")
        print(f"  Output: {text[:200]}...")

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": text,
        "correct": em,
        "exact_match": em,
        "f1": f1,
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run direct single-agent pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Direct] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_direct_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if not verbose:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [Direct] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
