"""Text-mode pipeline for HotpotQA: agents pass text as context.

Supports 2-agent (Finder → Answerer) and 3-agent (Decomposer → Finder → Answerer).
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS_2, AGENTS_3, build_text_prompt, format_paragraphs
from .evaluate import check_correct, exact_match, extract_answer, token_f1


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    gold_answer: str,
    paragraphs: List[Dict],
    num_agents: int = 2,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run text-mode pipeline on a single HotpotQA problem."""
    agents = AGENTS_2 if num_agents == 2 else AGENTS_3
    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        context = ""
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        for agent in agents:
            agent_t0 = time.perf_counter()

            # Count context tokens being re-processed
            if context:
                context_encoded = tokenizer(context, add_special_tokens=False)
                context_token_count = len(context_encoded["input_ids"])
            else:
                context_token_count = 0
            total_context_tokens += context_token_count

            # Build prompt — Finder gets paragraphs in text mode too
            needs_paragraphs = agent.role in ("finder", "finder_3agent")
            messages = build_text_prompt(
                agent.role, question,
                paragraphs_text=paragraphs_text if needs_paragraphs else "",
                context=context,
            )
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

            # Accumulate context for next agent
            context += f"[{agent.name}]:\n{text}\n\n"

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer(agent_traces[-1]["output"])
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"],
        "correct": em,
        "exact_match": em,
        "f1": f1,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "agents": agent_traces,
        "mode": "text",
    }


def run_text_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    num_agents: int = 2,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run text-mode pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_text_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            num_agents=num_agents,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["exact_match"] else "WRONG"
            print(f"  => {status} (pred='{result['prediction']}', "
                  f"gold='{result['gold']}', F1={result['f1']:.2f}, "
                  f"time={result['wall_time']:.1f}s)")
        else:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [Text] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
