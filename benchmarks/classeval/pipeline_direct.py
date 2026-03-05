"""Direct single-agent baseline: one agent generates the entire class in one shot."""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import SYSTEM_MESSAGE, build_direct_prompt
from .evaluate import check_correct, extract_class_code, build_test_code


def run_direct_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    sample: Dict,
    max_new_tokens: int = 1024,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Generate an entire class in one shot (no incremental chain)."""
    task_id = sample["task_id"]
    class_name = sample["class_name"]
    skeleton = sample["skeleton"]
    import_statement = sample["import_statement"]
    test_code = sample["test"]
    class_description = sample.get("class_description", "")

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()

        messages = build_direct_prompt(skeleton, class_description, import_statement)
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

    # Extract class code from generation
    class_code = extract_class_code(text, class_name)

    # Prepend imports if not already present
    if import_statement.strip() and import_statement.strip() not in class_code:
        class_code = import_statement.strip() + "\n\n" + class_code

    # Build test harness and evaluate
    test_harness = build_test_code(test_code, class_name)
    eval_result = check_correct(class_code, test_harness, import_statement)
    correct = eval_result["passed"]

    if verbose:
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {task_id} ({class_name}), time={wall_time:.1f}s, "
              f"tokens={total_tokens} ({tokens_per_sec:.1f} tok/s)")
        if eval_result["error"]:
            print(f"  Error: {eval_result['error'][:200]}")

    return {
        "task_id": task_id,
        "class_name": class_name,
        "raw_output": text,
        "extracted_code": class_code,
        "correct": correct,
        "error": eval_result["error"],
        "wall_time": wall_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "num_methods": len(sample.get("methods_info", [])),
        "mode": "direct",
    }


def run_direct_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run direct single-agent pipeline on a list of ClassEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Direct] Sample {i + 1}/{len(dataset)}: "
                  f"{sample['task_id']} ({sample['class_name']})")

        result = run_direct_pipeline(
            model, tokenizer, device, sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if not verbose:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [Direct] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
