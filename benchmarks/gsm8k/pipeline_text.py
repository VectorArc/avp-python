"""Text-mode baseline pipeline: agents pass generated text as context."""

import time
from typing import Any, Dict, List, Optional

import torch

from .agents import (
    AGENTS,
    build_text_prompt,
    generate_text,
    render_prompt,
    tokenize_prompt,
)
from .evaluate import extract_gold, extract_gsm8k_answer


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    gold_solution: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 4-agent text-mode pipeline on a single GSM8K problem.

    Each agent generates text, which is concatenated into the next agent's prompt.
    """
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()

    t0 = time.perf_counter()
    context = ""
    agent_traces: List[Dict] = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_context_tokens = 0

    for agent in AGENTS:
        agent_t0 = time.perf_counter()

        # Count context tokens being re-processed (the "communication tax")
        if context:
            context_encoded = tokenizer(context, add_special_tokens=False)
            context_token_count = len(context_encoded["input_ids"])
        else:
            context_token_count = 0
        total_context_tokens += context_token_count

        if agent.role == "judger":
            # Judger generates the final answer
            messages = build_text_prompt(agent.role, question, context)
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
        else:
            # Non-judger agents generate text that becomes context
            messages = build_text_prompt(agent.role, question, context)
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

    peak_memory_mb = None
    if device == "cuda":
        peak_memory_mb = (torch.cuda.max_memory_allocated() - mem_before) / (1024 * 1024)

    gold = extract_gold(gold_solution)
    prediction = extract_gsm8k_answer(agent_traces[-1]["output"])

    from .evaluate import check_correct
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
        "peak_memory_mb": peak_memory_mb,
        "agents": agent_traces,
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
    """Run text-mode pipeline on a list of GSM8K samples."""
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
                  f"time={result['wall_time']:.1f}s)")

    return results
