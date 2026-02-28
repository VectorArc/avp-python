"""Text-mode pipeline for Compiler benchmark: agents pass generated text as context."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_text_prompt
from .evaluate import check_correct, extract_answer


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    problem: str,
    gold_answer: str,
    level: str = "",
    subject: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 3-agent text-mode pipeline on a single MATH-500 problem."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        context = ""
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        for agent in AGENTS:
            agent_t0 = time.perf_counter()

            # Count context tokens being re-processed
            if context:
                context_encoded = tokenizer(context, add_special_tokens=False)
                context_token_count = len(context_encoded["input_ids"])
            else:
                context_token_count = 0
            total_context_tokens += context_token_count

            messages = build_text_prompt(agent.role, problem, context)
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

            context += f"[{agent.name}]:\n{text}\n\n"

            output_encoded = tokenizer(text, add_special_tokens=False)
            output_tokens = len(output_encoded["input_ids"])
            total_output_tokens += output_tokens
            agent_time_ms = (time.perf_counter() - agent_t0) * 1000

            agent_traces.append({
                "name": agent.name,
                "role": agent.role,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "context_tokens": context_token_count,
                "agent_time_ms": agent_time_ms,
                "output": text,
            })

            if verbose:
                print(f"  [{agent.name}] output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer(agent_traces[-1]["output"])
    correct = check_correct(prediction, gold_answer)

    return {
        "question": problem,
        "gold": gold_answer,
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
        "agents": agent_traces,
        "level": level,
        "subject": subject,
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
    """Run text-mode pipeline on MATH-500 samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: {sample['problem'][:80]}...")

        result = run_text_pipeline(
            model, tokenizer, device,
            problem=sample["problem"],
            gold_answer=sample["answer"],
            level=sample.get("level", ""),
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
                  f"time={result['wall_time']:.1f}s)")

    return results
