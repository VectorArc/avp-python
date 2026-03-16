"""Cross-model text pipeline: 2-agent chain where Coder (model A) passes text to Reviewer (model B).

This is the text baseline for cross-model comparison. Both agents communicate via
text (like pipeline_text.py), but each agent runs on a different model. This lets
us measure whether rosetta projection adds value over simply piping text between
different models.
"""

import time
from typing import Any, Dict, List

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_text_prompt
from .evaluate import check_correct


def run_text_cross_model_pipeline(
    model_a: Any,
    tokenizer_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    prompt: str,
    test: str,
    entry_point: str,
    task_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent cross-model text pipeline on a single HumanEval problem.

    Coder (model A) generates code draft.
    Reviewer (model B) receives that text in its prompt and generates the final code.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        coder = AGENTS[0]
        reviewer = AGENTS[1]

        # --- Agent 1: Coder on model A ---
        agent_t0 = time.perf_counter()
        messages = build_text_prompt(coder.role, prompt)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

        coder_text, _ = generate_text(
            model_a, tokenizer_a, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer_a(coder_text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": coder.name,
            "role": coder.role,
            "model": "model_a",
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "context_tokens": 0,
            "agent_time_ms": agent_time_ms,
            "output": coder_text,
        })

        if verbose:
            print(f"  [{coder.name} (A)] output ({len(coder_text)} chars): "
                  f"{coder_text[:200]}...")

        # --- Agent 2: Reviewer on model B ---
        agent_t0 = time.perf_counter()

        # Count context tokens — Coder's text re-tokenized by model B's tokenizer
        context_encoded = tokenizer_b(coder_text, add_special_tokens=False)
        context_token_count = len(context_encoded["input_ids"])
        total_context_tokens += context_token_count

        messages = build_text_prompt(reviewer.role, prompt, coder_text)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

        reviewer_text, _ = generate_text(
            model_b, tokenizer_b, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer_b(reviewer_text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": reviewer.name,
            "role": reviewer.role,
            "model": "model_b",
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "context_tokens": context_token_count,
            "agent_time_ms": agent_time_ms,
            "output": reviewer_text,
        })

        if verbose:
            print(f"  [{reviewer.name} (B)] output ({len(reviewer_text)} chars): "
                  f"{reviewer_text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    eval_result = check_correct(
        agent_traces[-1]["output"], prompt, test, entry_point,
    )
    correct = eval_result["passed"]

    return {
        "task_id": task_id,
        "prompt": prompt,
        "raw_output": agent_traces[-1]["output"],
        "extracted_code": eval_result["code"],
        "correct": correct,
        "error": eval_result["error"],
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "agents": agent_traces,
        "mode": "text_cross_model",
    }


def run_text_cross_model_benchmark(
    model_a: Any,
    tokenizer_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run cross-model text pipeline on a list of HumanEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[TextCrossModel] Sample {i + 1}/{len(dataset)}: {sample['task_id']}")

        result = run_text_cross_model_pipeline(
            model_a, tokenizer_a, model_b, tokenizer_b, device,
            prompt=sample["prompt"],
            test=sample["test"],
            entry_point=sample["entry_point"],
            task_id=sample["task_id"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "PASS" if result["correct"] else "FAIL"
            print(f"  => {status} (time={result['wall_time']:.1f}s)")
        else:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [TextCrossModel] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
