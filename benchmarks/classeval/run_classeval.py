#!/usr/bin/env python3
"""ClassEval benchmark: incremental class-level code generation.

Tests class generation with method dependencies. Methods are generated
in dependency order, with context carried forward via text or KV-cache.

Evaluation: execute generated class against unit tests (class-level pass@1
and per-method pass rate).

Usage:
    # Run all modes
    python benchmarks/classeval/run_classeval.py --mode all --max_samples 10 --verbose

    # Direct single-agent baseline only
    python benchmarks/classeval/run_classeval.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/classeval/run_classeval.py --mode both --verbose
"""

import argparse
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure avp-python root is on sys.path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ClassEval benchmark: incremental class-level code generation"
    )
    parser.add_argument(
        "--mode",
        choices=["latent", "text", "direct", "both", "all"],
        default="all",
        help="Pipeline(s) to run (default: all)",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cpu/mps/cuda",
    )
    parser.add_argument(
        "--max_samples", type=int, default=10,
        help="Number of classes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--latent_steps", type=int, default=10,
        help="Latent steps per method (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens per method generation (default: 512)",
    )
    parser.add_argument(
        "--max_new_tokens_direct", type=int, default=1024,
        help="Max tokens for direct whole-class generation (default: 1024)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01,
        help="Sampling temperature (default: 0.01 for near-greedy pass@1)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-sample details",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results directory",
    )
    return parser.parse_args()


def topological_sort_methods(methods_info: List[Dict]) -> List[Dict]:
    """Sort methods in dependency order using Kahn's algorithm.

    Methods with no dependencies come first, then methods whose dependencies
    are all already generated. Falls back to original order if the dependency
    graph has cycles.

    Args:
        methods_info: List of method dicts, each with 'method_name' and
                      'dependencies' containing 'method_dependencies' list.

    Returns:
        Methods sorted in dependency order.
    """
    # Build adjacency list and in-degree map
    method_names = {m["method_name"] for m in methods_info}
    method_by_name = {m["method_name"]: m for m in methods_info}

    # in_degree[m] = number of method dependencies m has (that are in this class)
    in_degree = defaultdict(int)
    # dependents[m] = list of methods that depend on m
    dependents = defaultdict(list)

    for m in methods_info:
        name = m["method_name"]
        deps = m.get("dependencies", {})
        method_deps = deps.get("method_dependencies", [])
        # Only count dependencies that are methods in this class
        valid_deps = [d for d in method_deps if d in method_names and d != name]
        in_degree[name] = len(valid_deps)
        for dep in valid_deps:
            dependents[dep].append(name)

    # Kahn's algorithm
    queue = deque()
    for m in methods_info:
        if in_degree[m["method_name"]] == 0:
            queue.append(m["method_name"])

    sorted_names = []
    while queue:
        name = queue.popleft()
        sorted_names.append(name)
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If cycle detected, fall back to original order
    if len(sorted_names) != len(methods_info):
        print(f"  Warning: dependency cycle detected, using original order")
        return list(methods_info)

    return [method_by_name[name] for name in sorted_names]


def load_dataset(max_samples: int) -> List[Dict]:
    """Load ClassEval dataset from HuggingFace.

    Preprocesses each sample:
    - Joins import_statement list into a string
    - Topologically sorts methods by dependencies
    - Builds class description from fields and constructor
    """
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading ClassEval (first {max_samples} samples)...")
    ds = hf_load_dataset("FudanSELab/ClassEval", split="test")

    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        # Join import statements
        import_stmts = item.get("import_statement", [])
        if isinstance(import_stmts, list):
            import_statement = "\n".join(import_stmts)
        else:
            import_statement = str(import_stmts)

        # Parse methods_info
        methods_info = item.get("methods_info", [])

        # Topological sort
        methods_order = topological_sort_methods(methods_info)

        # Build class description from available fields
        class_description = ""
        fields = item.get("fields", "")
        if fields:
            class_description += f"Fields: {fields}\n"
        constructor = item.get("class_constructor", "")
        if constructor:
            class_description += f"Constructor: {constructor}\n"
        # Add method descriptions
        for m in methods_info:
            desc = m.get("method_description", "")
            if desc:
                class_description += f"- {m['method_name']}: {desc}\n"

        # Build test code from test field
        test_code = item.get("test", "")

        samples.append({
            "task_id": item.get("task_id", f"ClassEval_{i}"),
            "class_name": item.get("class_name", f"Class_{i}"),
            "skeleton": item.get("skeleton", ""),
            "import_statement": import_statement,
            "test": test_code,
            "solution_code": item.get("solution_code", ""),
            "methods_info": methods_info,
            "methods_order": methods_order,
            "class_description": class_description,
            "num_methods": len(methods_info),
        })

    total_methods = sum(s["num_methods"] for s in samples)
    print(f"Loaded {len(samples)} classes ({total_methods} methods total).")
    return samples


def run_benchmark(config: dict) -> dict:
    """Run ClassEval benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 512)
    max_new_tokens_direct = config.get("max_new_tokens_direct", 1024)
    temperature = config.get("temperature", 0.01)
    top_p = config.get("top_p", 0.95)
    verbose = config.get("verbose", False)
    output_dir = config.get("output_dir")

    run_direct = mode in ("direct", "all")
    run_latent = mode in ("latent", "both", "all")
    run_text = mode in ("text", "both", "all")

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Samples: {max_samples}")
    print(f"Latent steps/method: {latent_steps}")
    print(f"Max new tokens/method: {max_new_tokens}")
    print(f"Max new tokens/class (direct): {max_new_tokens_direct}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(max_samples)
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.classeval.pipeline_direct import run_direct_benchmark

        print("\n" + "=" * 50)
        print("Running DIRECT (single-agent, whole-class) baseline...")
        print("=" * 50)
        set_seed(seed)

        direct_results = run_direct_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens_direct, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_text:
        from benchmarks.classeval.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (incremental, text context) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_latent:
        from benchmarks.classeval.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (incremental, AVP KV-cache) pipeline...")
        print("=" * 50)
        set_seed(seed)

        latent_results = run_latent_benchmark(
            connector=connector, model=model, tokenizer=tokenizer,
            device=device, identity=identity, dataset=dataset,
            model_name=model_name, latent_steps=latent_steps,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    # Print summary
    from benchmarks.classeval.evaluate import compute_accuracy, compute_method_accuracy
    from benchmarks.shared.evaluate_common import (
        compute_agreement,
        compute_stats,
        print_summary,
    )

    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    # Compute agreement across available modes
    available = {}
    if direct_results is not None:
        available["direct"] = direct_results
    if text_results is not None:
        available["text"] = text_results
    if latent_results is not None:
        available["latent"] = latent_results
    agreement_data = compute_agreement(available) if len(available) > 1 else None

    print_summary(
        benchmark_name="ClassEval",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
        agreement=agreement_data,
    )

    # Print class-level and method-level accuracy
    for label, _, res in modes:
        if res:
            acc = compute_accuracy(res)
            method_acc = compute_method_accuracy(res)
            print(f"\n  {label}:")
            print(f"    Class pass@1 = {acc['pass_at_1']:.1%} "
                  f"({acc['passed']}/{acc['total']})")
            if method_acc["methods_total"] > 0:
                print(f"    Method pass rate = {method_acc['method_pass_rate']:.1%} "
                      f"({method_acc['methods_passed']}/{method_acc['methods_total']})")

            # Show failed class IDs
            failed = [r["task_id"] for r in res if not r.get("correct", False)]
            if failed and len(failed) <= 20:
                print(f"    Failed classes: {', '.join(failed)}")

    # Save results
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(_SCRIPT_DIR)),
            "benchmarks", "results",
        )

    output_data = {
        "config": {
            "benchmark": "classeval",
            "model": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "latent_steps": latent_steps,
            "max_new_tokens": max_new_tokens,
            "max_new_tokens_direct": max_new_tokens_direct,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }
    if direct_results is not None:
        output_data["direct"] = {
            "summary": compute_stats(direct_results),
            "class_accuracy": compute_accuracy(direct_results),
            "samples": direct_results,
        }
    if latent_results is not None:
        output_data["latent"] = {
            "summary": compute_stats(latent_results),
            "class_accuracy": compute_accuracy(latent_results),
            "method_accuracy": compute_method_accuracy(latent_results),
            "samples": latent_results,
        }
    if text_results is not None:
        output_data["text"] = {
            "summary": compute_stats(text_results),
            "class_accuracy": compute_accuracy(text_results),
            "method_accuracy": compute_method_accuracy(text_results),
            "samples": text_results,
        }
    if agreement_data is not None:
        output_data["agreement"] = agreement_data

    save_results(output_data, output_dir, "classeval", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
