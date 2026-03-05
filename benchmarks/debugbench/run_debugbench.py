#!/usr/bin/env python3
"""DebugBench 2-agent benchmark: Detector -> Fixer handoff.

Tests bug detection and fixing with a 2-agent Detector/Fixer pattern.
Dataset: Rtian/DebugBench (4,253 instances, LeetCode-style problems).
Evaluation: run generated code against example test cases (pass@1).
Python-only subset (cpp/java filtered out -- no compilers needed).

Usage:
    # Run all modes
    python benchmarks/debugbench/run_debugbench.py --mode all --max_samples 10 --verbose

    # Direct single-agent baseline only
    python benchmarks/debugbench/run_debugbench.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/debugbench/run_debugbench.py --mode both --verbose

    # Filter by difficulty level
    python benchmarks/debugbench/run_debugbench.py --mode all --level easy --max_samples 20

    # Filter by bug category
    python benchmarks/debugbench/run_debugbench.py --mode all --category "logic error" --max_samples 20
"""

import argparse
import os
import sys

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
        description="DebugBench 2-agent benchmark: Detector -> Fixer handoff"
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
    parser.add_argument("--device", type=str, default=None, help="Device: cpu/mps/cuda")
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples (default: 10)")
    parser.add_argument("--latent_steps", type=int, default=10, help="Latent steps for Detector (default: 10)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature (default: 0.01)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory")
    parser.add_argument(
        "--level", type=str, default=None,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by bug category (e.g. 'syntax error', 'logic error')",
    )
    return parser.parse_args()


def load_dataset(
    max_samples: int,
    level: str = None,
    category: str = None,
) -> list:
    """Load DebugBench dataset from HuggingFace, filtered to Python only.

    Args:
        max_samples: Maximum number of samples to return.
        level: Optional difficulty filter (easy/medium/hard).
        category: Optional bug category filter.

    Returns:
        List of sample dicts with standardized field names.
    """
    from datasets import load_dataset as hf_load_dataset

    filters = ["language=python3"]
    if level:
        filters.append(f"level={level}")
    if category:
        filters.append(f"category={category}")
    filter_desc = ", ".join(filters)

    print(f"Loading DebugBench ({filter_desc}, max {max_samples} samples)...")
    ds = hf_load_dataset("Rtian/DebugBench", split="test")

    samples = []
    skipped_lang = 0
    skipped_level = 0
    skipped_category = 0

    for item in ds:
        # Filter to Python only
        if item.get("language", "").lower() not in ("python", "python3"):
            skipped_lang += 1
            continue

        # Filter by level if specified
        if level and item.get("level", "").lower() != level.lower():
            skipped_level += 1
            continue

        # Filter by category if specified
        if category and item.get("category", "").lower() != category.lower():
            skipped_category += 1
            continue

        if len(samples) >= max_samples:
            break

        # Normalize the examples field -- it may be a list of strings or a single string
        examples_raw = item.get("examples", [])
        if isinstance(examples_raw, str):
            examples_raw = [examples_raw] if examples_raw.strip() else []

        samples.append({
            "slug": item.get("slug", f"sample-{len(samples)}"),
            "question": item.get("question", ""),
            "buggy_code": item.get("buggy_code", ""),
            "solution": item.get("solution", ""),
            "bug_explanation": item.get("bug_explanation", ""),
            "examples": examples_raw,
            "constraints": item.get("constraints", ""),
            "category": item.get("category", ""),
            "subtype": item.get("subtype", ""),
            "level": item.get("level", ""),
        })

    print(f"Loaded {len(samples)} Python samples "
          f"(skipped: {skipped_lang} non-Python"
          + (f", {skipped_level} wrong level" if level else "")
          + (f", {skipped_category} wrong category" if category else "")
          + f").")

    if samples:
        # Print distribution summary
        levels = {}
        categories = {}
        subtypes = {}
        for s in samples:
            lvl = s.get("level", "unknown")
            cat = s.get("category", "unknown")
            sub = s.get("subtype", "unknown")
            levels[lvl] = levels.get(lvl, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
            subtypes[sub] = subtypes.get(sub, 0) + 1

        print(f"  Levels: {dict(sorted(levels.items()))}")
        print(f"  Categories: {dict(sorted(categories.items()))}")
        if len(subtypes) <= 10:
            print(f"  Subtypes: {dict(sorted(subtypes.items()))}")

    return samples


def run_benchmark(config: dict) -> dict:
    """Run DebugBench 2-agent benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 512)
    temperature = config.get("temperature", 0.01)
    top_p = config.get("top_p", 0.95)
    verbose = config.get("verbose", False)
    output_dir = config.get("output_dir")
    level = config.get("level")
    category = config.get("category")

    run_direct = mode in ("direct", "all")
    run_latent = mode in ("latent", "both", "all")
    run_text = mode in ("text", "both", "all")

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Samples: {max_samples}")
    print(f"Latent steps: {latent_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    if level:
        print(f"Level filter: {level}")
    if category:
        print(f"Category filter: {category}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(max_samples, level=level, category=category)
    if not dataset:
        print("No samples matched filters. Exiting.")
        return {"config": config, "error": "no_samples"}

    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.debugbench.pipeline_direct import run_direct_benchmark

        print("\n" + "=" * 50)
        print("Running DIRECT (single-agent) baseline...")
        print("=" * 50)
        set_seed(seed)

        direct_results = run_direct_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_text:
        from benchmarks.debugbench.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (2-agent chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_latent:
        from benchmarks.debugbench.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP 2-agent chain) pipeline...")
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
    from benchmarks.debugbench.evaluate import compute_accuracy
    from benchmarks.shared.evaluate_common import print_summary, compute_stats, compute_agreement

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
        benchmark_name="DebugBench 2-Agent",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
        agreement=agreement_data,
    )

    # Print pass@1 breakdown
    for label, _, res in modes:
        if res:
            acc = compute_accuracy(res)
            print(f"\n  {label}: pass@1 = {acc['pass_at_1']:.1%} "
                  f"({acc['passed']}/{acc['total']})")

            # Show failed slugs
            failed = [r["slug"] for r in res if not r.get("correct", False)]
            if failed and len(failed) <= 20:
                print(f"    Failed: {', '.join(failed)}")

            # Show eval method breakdown
            eval_methods = {}
            for r in res:
                m = r.get("eval_method", "unknown")
                eval_methods[m] = eval_methods.get(m, 0) + 1
            print(f"    Eval methods: {eval_methods}")

    # Save results
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(_SCRIPT_DIR)),
            "benchmarks", "results",
        )

    output_data = {
        "config": {
            "benchmark": "debugbench",
            "model": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "latent_steps": latent_steps,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "level": level,
            "category": category,
        },
    }
    if direct_results is not None:
        output_data["direct"] = {
            "summary": compute_stats(direct_results),
            "samples": direct_results,
        }
    if latent_results is not None:
        output_data["latent"] = {
            "summary": compute_stats(latent_results),
            "samples": latent_results,
        }
    if text_results is not None:
        output_data["text"] = {
            "summary": compute_stats(text_results),
            "samples": text_results,
        }
    if agreement_data is not None:
        output_data["agreement"] = agreement_data

    save_results(output_data, output_dir, "debugbench", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
