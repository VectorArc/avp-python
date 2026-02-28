"""Text-mode pipeline for Supervisor benchmark: Router + Specialist via text."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import (
    ROUTER,
    SPECIALISTS,
    SUBJECT_TO_CATEGORY,
    build_router_prompt,
    build_text_specialist_prompt,
    extract_route,
)
from .evaluate import check_correct, extract_answer_letter


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    choices: List[str],
    gold_answer: int,
    subject: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run Router + Specialist text pipeline on a single MMLU question."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        # --- Step 1: Router classifies the question ---
        router_t0 = time.perf_counter()
        messages = build_router_prompt(question, choices)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        router_text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            max_new_tokens=64,
            temperature=temperature,
            top_p=top_p,
        )

        route = extract_route(router_text)
        specialist = SPECIALISTS.get(route, SPECIALISTS["stem"])

        output_encoded = tokenizer(router_text, add_special_tokens=False)
        router_output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += router_output_tokens
        router_time_ms = (time.perf_counter() - router_t0) * 1000

        agent_traces.append({
            "name": ROUTER.name,
            "role": ROUTER.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": router_output_tokens,
            "context_tokens": 0,
            "agent_time_ms": router_time_ms,
            "output": router_text,
            "route": route,
        })

        if verbose:
            print(f"  [Router] route={route}, output: {router_text[:100]}...")

        # --- Step 2: Selected Specialist answers ---
        specialist_t0 = time.perf_counter()

        # Count context tokens (Router's text)
        context_encoded = tokenizer(router_text, add_special_tokens=False)
        context_token_count = len(context_encoded["input_ids"])
        total_context_tokens += context_token_count

        messages = build_text_specialist_prompt(question, choices, router_text)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        specialist_text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer(specialist_text, add_special_tokens=False)
        specialist_output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += specialist_output_tokens
        specialist_time_ms = (time.perf_counter() - specialist_t0) * 1000

        agent_traces.append({
            "name": specialist.name,
            "role": specialist.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": specialist_output_tokens,
            "context_tokens": context_token_count,
            "agent_time_ms": specialist_time_ms,
            "output": specialist_text,
        })

        if verbose:
            print(f"  [{specialist.name}] output: {specialist_text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer_letter(specialist_text)
    correct = check_correct(prediction, gold_answer)
    expected_category = SUBJECT_TO_CATEGORY.get(subject, "unknown")
    routing_correct = (route == expected_category)

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": specialist_text,
        "correct": correct,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "agents": agent_traces,
        "subject": subject,
        "expected_category": expected_category,
        "route": route,
        "routing_correct": routing_correct,
        "mode": "text",
    }


def run_text_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run text-mode pipeline on MMLU samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_text_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            choices=sample["choices"],
            gold_answer=sample["answer"],
            subject=sample.get("subject", ""),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"route={result['route']}, time={result['wall_time']:.1f}s)")

    return results
