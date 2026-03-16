"""Cross-model text pipeline: 2-agent chain where Detector (model A) passes text to Fixer (model B).

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
    """Run the 2-agent cross-model text pipeline on a single DebugBench problem.

    Detector (model A) generates text analysis.
    Fixer (model B) receives that text in its prompt and generates the fix.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        detector = AGENTS[0]
        fixer = AGENTS[1]

        # --- Agent 1: Detector on model A ---
        agent_t0 = time.perf_counter()
        messages = build_text_prompt(detector.role, question, buggy_code)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

        detector_text, _ = generate_text(
            model_a, tokenizer_a, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer_a(detector_text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": detector.name,
            "role": detector.role,
            "model": "model_a",
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "context_tokens": 0,
            "agent_time_ms": agent_time_ms,
            "output": detector_text,
        })

        if verbose:
            print(f"  [{detector.name} (A)] output ({len(detector_text)} chars): "
                  f"{detector_text[:200]}...")

        # --- Agent 2: Fixer on model B ---
        agent_t0 = time.perf_counter()

        # Count context tokens — Detector's text re-tokenized by model B's tokenizer
        context_encoded = tokenizer_b(detector_text, add_special_tokens=False)
        context_token_count = len(context_encoded["input_ids"])
        total_context_tokens += context_token_count

        messages = build_text_prompt(fixer.role, question, buggy_code, detector_text)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

        fixer_text, _ = generate_text(
            model_b, tokenizer_b, input_ids, attention_mask, device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer_b(fixer_text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": fixer.name,
            "role": fixer.role,
            "model": "model_b",
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "context_tokens": context_token_count,
            "agent_time_ms": agent_time_ms,
            "output": fixer_text,
        })

        if verbose:
            print(f"  [{fixer.name} (B)] output ({len(fixer_text)} chars): "
                  f"{fixer_text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    eval_result = check_correct(
        agent_traces[-1]["output"], solution, question, examples,
    )
    correct = eval_result["passed"]

    return {
        "slug": slug,
        "question": question[:100],
        "raw_output": agent_traces[-1]["output"],
        "extracted_code": eval_result["code"],
        "correct": correct,
        "error": eval_result["error"],
        "eval_method": eval_result.get("eval_method", "unknown"),
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
    """Run cross-model text pipeline on a list of DebugBench samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[TextCrossModel] Sample {i + 1}/{len(dataset)}: {sample['slug']}")

        result = run_text_cross_model_pipeline(
            model_a, tokenizer_a, model_b, tokenizer_b, device,
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

        if verbose:
            status = "PASS" if result["correct"] else "FAIL"
            print(f"  => {status} (time={result['wall_time']:.1f}s)")
        else:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [TextCrossModel] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
