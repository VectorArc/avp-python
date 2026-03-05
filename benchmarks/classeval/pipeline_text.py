"""Text-mode pipeline: incremental class generation with text context accumulation.

For each method in dependency order:
1. Build prompt with class skeleton + all prior generated methods (as text)
2. Generate the current method
3. Append generated method text to accumulated context

The text context grows with each step -- later methods see all prior methods
as text in the prompt, causing redundant re-processing of prior context.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import build_text_prompt
from .evaluate import (
    assemble_class,
    build_test_code,
    check_correct,
    extract_method_code,
)


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    sample: Dict,
    methods_order: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run incremental text-chain generation on a single ClassEval class.

    Each method is generated with all prior methods included as text context.
    """
    task_id = sample["task_id"]
    class_name = sample["class_name"]
    skeleton = sample["skeleton"]
    import_statement = sample["import_statement"]
    test_code = sample["test"]
    class_description = sample.get("class_description", "")

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0

        generated_methods: Dict[str, str] = {}
        prior_methods_text = ""

        for step_idx, method_info in enumerate(methods_order):
            method_name = method_info["method_name"]
            agent_t0 = time.perf_counter()

            # Build prompt with text context from all prior methods
            messages = build_text_prompt(
                skeleton, class_description, method_info,
                prior_methods_text, import_statement,
            )
            prompt_text = render_prompt(tokenizer, messages)
            input_ids, attention_mask = tokenize_prompt(
                tokenizer, prompt_text, device,
            )
            p_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += p_tokens

            # Count context tokens (prior methods text re-processed)
            if prior_methods_text.strip():
                ctx_encoded = tokenizer(
                    prior_methods_text, add_special_tokens=False,
                )
                context_token_count = len(ctx_encoded["input_ids"])
                total_context_tokens += context_token_count
            else:
                context_token_count = 0

            # Generate the method
            method_text, _ = generate_text(
                model, tokenizer, input_ids, attention_mask, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            output_encoded = tokenizer(method_text, add_special_tokens=False)
            output_tokens = len(output_encoded["input_ids"])
            total_output_tokens += output_tokens
            agent_time_ms = (time.perf_counter() - agent_t0) * 1000

            # Extract the method code
            method_code = extract_method_code(method_text, method_name)
            generated_methods[method_name] = method_code

            # Accumulate text context for subsequent methods
            if prior_methods_text:
                prior_methods_text += "\n\n" + method_code
            else:
                prior_methods_text = method_code

            agent_traces.append({
                "step": step_idx,
                "method_name": method_name,
                "prompt_tokens": p_tokens,
                "output_tokens": output_tokens,
                "context_tokens": context_token_count,
                "agent_time_ms": agent_time_ms,
                "output": method_text,
                "extracted_method": method_code,
            })

            if verbose:
                print(f"  [Text step {step_idx + 1}/{len(methods_order)}] "
                      f"{method_name}: {output_tokens} tokens, "
                      f"context={context_token_count} tokens, "
                      f"{agent_time_ms:.0f}ms")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Assemble the complete class and evaluate
    class_code = assemble_class(
        class_name, skeleton, generated_methods, import_statement,
    )
    test_harness = build_test_code(test_code, class_name)
    eval_result = check_correct(class_code, test_harness, import_statement)
    correct = eval_result["passed"]

    # Per-method test results (run each method's test individually if available)
    method_results = _evaluate_per_method(
        class_code, sample.get("methods_info", []), import_statement,
    )

    return {
        "task_id": task_id,
        "class_name": class_name,
        "raw_output": prior_methods_text,
        "extracted_code": class_code,
        "correct": correct,
        "error": eval_result["error"],
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "num_methods": len(methods_order),
        "method_results": method_results,
        "agents": agent_traces,
        "mode": "text",
    }


def _evaluate_per_method(
    class_code: str,
    methods_info: List[Dict],
    import_statement: str,
) -> Dict[str, bool]:
    """Run each method's individual test to get per-method pass rates.

    Returns {method_name: passed_bool}.
    """
    results = {}
    for method_info in methods_info:
        method_name = method_info["method_name"]
        test_code = method_info.get("test_code", "")
        if not test_code or not test_code.strip():
            continue
        test_harness = (
            "import unittest\n\n"
            + test_code
            + "\n\nif __name__ == '__main__':\n"
            + "    unittest.main()\n"
        )
        eval_result = check_correct(
            class_code, test_harness, import_statement, timeout=15,
        )
        results[method_name] = eval_result["passed"]
    return results


def run_text_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run text-mode pipeline on a list of ClassEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: "
                  f"{sample['task_id']} ({sample['class_name']}, "
                  f"{len(sample['methods_order'])} methods)")

        result = run_text_pipeline(
            model, tokenizer, device, sample,
            methods_order=sample["methods_order"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if not verbose:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [Text] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
