"""Text-mode pipeline for fan-out benchmark: specialists pass text to aggregator.

Both specialists independently generate text, which is combined in the
Aggregator's prompt.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGGREGATOR, SPECIALISTS, build_text_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_text_pipeline(
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
    """Run fan-out text pipeline on a single GSM8K problem.

    Specialists generate text independently, then Aggregator reviews both.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0
        specialist_outputs: List[str] = []
        specialist_times: List[float] = []

        # --- Specialists (independent, but sequential on single GPU) ---
        for specialist in SPECIALISTS:
            agent_t0 = time.perf_counter()

            messages = build_text_prompt(specialist.role, question)
            prompt_text = render_prompt(tokenizer, messages)
            input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
            prompt_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += prompt_tokens

            text, _ = generate_text(
                model, tokenizer, input_ids, attention_mask, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            output_encoded = tokenizer(text, add_special_tokens=False)
            output_tokens = len(output_encoded["input_ids"])
            total_output_tokens += output_tokens
            agent_time_ms = (time.perf_counter() - agent_t0) * 1000
            specialist_times.append(agent_time_ms)

            specialist_outputs.append(f"[{specialist.name}]:\n{text}")

            agent_traces.append({
                "name": specialist.name,
                "role": specialist.role,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "context_tokens": 0,
                "agent_time_ms": agent_time_ms,
                "output": text,
            })

            if verbose:
                print(f"  [{specialist.name}] output ({len(text)} chars): {text[:200]}...")

        # --- Aggregator ---
        agent_t0 = time.perf_counter()

        context = "\n\n".join(specialist_outputs)
        context_encoded = tokenizer(context, add_special_tokens=False)
        context_token_count = len(context_encoded["input_ids"])
        total_context_tokens += context_token_count

        messages = build_text_prompt(AGGREGATOR.role, question, context)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer(text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": AGGREGATOR.name,
            "role": AGGREGATOR.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "context_tokens": context_token_count,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{AGGREGATOR.name}] output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Parallel speedup potential
    parallel_speedup_potential = (
        sum(specialist_times) / max(specialist_times) if specialist_times else 1.0
    )

    gold = extract_gold(gold_solution)
    prediction = extract_gsm8k_answer(agent_traces[-1]["output"])
    correct = check_correct(prediction, gold)

    return {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"],
        "correct": correct,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "parallel_speedup_potential": parallel_speedup_potential,
        "agents": agent_traces,
        "mode": "text",
    }


def run_text_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run fan-out text pipeline on GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_text_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            gold_solution=sample["answer"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"parallel_speedup={result['parallel_speedup_potential']:.1f}x)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Text] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
