"""Latent-mode pipeline: incremental class generation with KV-cache transfer.

For each method in dependency order:
1. Build prompt with class skeleton + current method description
2. Run latent steps, building on accumulated KV-cache from all prior methods
3. AVP roundtrip the KV-cache
4. On the final method, generate text output
5. For non-final methods, generate text too (we need the code), but carry KV-cache forward

Key difference from text mode: context transfers via KV-cache (no re-processing
of prior methods in the prompt). Each step's prompt only contains the skeleton
and current method description.
"""

import time
import uuid
from typing import Any, Dict, List

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import build_latent_prompt
from .evaluate import (
    assemble_class,
    build_test_code,
    check_correct,
    extract_method_code,
)


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    sample: Dict,
    methods_order: List[Dict],
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run incremental latent-chain generation on a single ClassEval class.

    Each method generation step:
    1. Latent steps with accumulated KV-cache (context from prior methods)
    2. AVP codec roundtrip
    3. Generate method text (need the code to assemble the class)
    4. Carry KV-cache forward to next method step
    """
    task_id = sample["task_id"]
    class_name = sample["class_name"]
    skeleton = sample["skeleton"]
    import_statement = sample["import_statement"]
    test_code = sample["test"]
    class_description = sample.get("class_description", "")

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        generated_methods: Dict[str, str] = {}
        past_kv = None  # Accumulated KV-cache across methods

        for step_idx, method_info in enumerate(methods_order):
            method_name = method_info["method_name"]
            agent_t0 = time.perf_counter()

            # Build prompt for this method (no prior text context -- KV-cache has it)
            messages = build_latent_prompt(
                skeleton, class_description, method_info, import_statement,
            )
            prompt_text = render_prompt(tokenizer, messages)
            input_ids, attention_mask = tokenize_prompt(
                tokenizer, prompt_text, device,
            )
            p_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += p_tokens
            total_latent_steps += latent_steps

            # Run latent steps with accumulated KV-cache
            step_past_kv = connector.generate_latent_steps(
                input_ids,
                latent_steps=latent_steps,
                attention_mask=attention_mask,
                past_key_values=past_kv,
            )

            # AVP codec roundtrip
            step_past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
                step_past_kv, session_id,
                f"method_{step_idx}", f"method_{step_idx + 1}",
                model_name, identity, device,
            )
            total_codec_time_ms += codec_time_ms
            total_wire_bytes += wire_size

            kv_seq_len = get_past_length(step_past_kv)

            # Generate the method text (we need the actual code for every method)
            method_text, gen_past_kv = generate_text(
                model, tokenizer, input_ids, attention_mask, device,
                past_key_values=step_past_kv,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            output_encoded = tokenizer(method_text, add_special_tokens=False)
            output_tokens = len(output_encoded["input_ids"])
            total_output_tokens += output_tokens

            # Carry generation KV-cache forward — later methods can attend to
            # what earlier methods actually produced (code, signatures, etc.),
            # not just the latent "thinking" context.
            past_kv = gen_past_kv

            agent_time_ms = (time.perf_counter() - agent_t0) * 1000

            # Extract method code
            method_code = extract_method_code(method_text, method_name)
            generated_methods[method_name] = method_code

            agent_traces.append({
                "step": step_idx,
                "method_name": method_name,
                "prompt_tokens": p_tokens,
                "latent_steps": latent_steps,
                "output_tokens": output_tokens,
                "kv_seq_len_after": kv_seq_len,
                "wire_bytes": wire_size,
                "codec_time_ms": codec_time_ms,
                "agent_time_ms": agent_time_ms,
                "output": method_text,
                "extracted_method": method_code,
            })

            if verbose:
                print(f"  [Latent step {step_idx + 1}/{len(methods_order)}] "
                      f"{method_name}: latent={latent_steps}, "
                      f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                      f"codec={codec_time_ms:.1f}ms, gen={output_tokens} tokens")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Assemble the complete class and evaluate
    class_code = assemble_class(
        class_name, skeleton, generated_methods, import_statement,
    )
    test_harness = build_test_code(test_code, class_name)
    eval_result = check_correct(class_code, test_harness, import_statement)
    correct = eval_result["passed"]

    # Per-method test results
    method_results = _evaluate_per_method(
        class_code, sample.get("methods_info", []), import_statement,
    )

    return {
        "task_id": task_id,
        "class_name": class_name,
        "raw_output": "\n\n".join(
            t["output"] for t in agent_traces if t.get("output")
        ),
        "extracted_code": class_code,
        "correct": correct,
        "error": eval_result["error"],
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_latent_steps": total_latent_steps,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
        "num_methods": len(methods_order),
        "method_results": method_results,
        "agents": agent_traces,
        "mode": "latent",
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


def run_latent_benchmark(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    dataset: List[Dict],
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode pipeline on a list of ClassEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: "
                  f"{sample['task_id']} ({sample['class_name']}, "
                  f"{len(sample['methods_order'])} methods)")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity, sample,
            methods_order=sample["methods_order"],
            model_name=model_name,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if not verbose:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [Latent] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
