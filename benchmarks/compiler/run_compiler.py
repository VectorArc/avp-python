#!/usr/bin/env python3
"""Compiler (Plan-Execute) benchmark: 3-agent chain on MATH-500.

Topology: Planner --> Executor --> Verifier --> Answer

Usage:
    # Run all modes
    python benchmarks/compiler/run_compiler.py --mode all --max_samples 10 --verbose

    # Direct baseline only
    python benchmarks/compiler/run_compiler.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/compiler/run_compiler.py --mode both --verbose

    # Filter by difficulty level
    python benchmarks/compiler/run_compiler.py --mode all --max_level 3 --verbose
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
        description="Compiler (Plan-Execute) benchmark on MATH-500"
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
    parser.add_argument("--max_level", type=int, default=5, help="Max difficulty level 1-5 (default: 5)")
    parser.add_argument("--latent_steps", type=int, default=10, help="Latent steps per non-terminal agent (default: 10)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens for generation (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory")
    return parser.parse_args()


def load_dataset(max_samples: int, max_level: int = 5) -> list:
    """Load MATH-500 test split from HuggingFace datasets."""
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading MATH-500 test split (max {max_samples} samples, level <= {max_level})...")
    ds = hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
    samples = []
    for item in ds:
        level_num = item["level"]
        if level_num > max_level:
            continue
        samples.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": item["answer"],
            "subject": item["subject"],
            "level": level_num,
            "unique_id": item["unique_id"],
        })
        if len(samples) >= max_samples:
            break
    print(f"Loaded {len(samples)} samples.")
    return samples


def run_benchmark(config: dict) -> dict:
    """Run Compiler (MATH-500) benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    max_level = config.get("max_level", 5)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 512)
    temperature = config.get("temperature", 0.7)
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
    print(f"Max level: {max_level}")
    print(f"Latent steps: {latent_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(max_samples, max_level)
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.compiler.pipeline_direct import run_direct_benchmark

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
        from benchmarks.compiler.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (3-agent chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_latent:
        from benchmarks.compiler.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP 3-agent chain) pipeline...")
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
    from benchmarks.compiler.evaluate import compute_accuracy, print_level_summary
    from benchmarks.shared.evaluate_common import print_summary

    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    print_summary(
        benchmark_name="Compiler (MATH-500)",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
    )

    # Print per-level and per-subject breakdowns
    for label, _, res in modes:
        if res:
            print(f"\n--- {label} Mode ---")
            print_level_summary(res)

    # Save results
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "benchmarks", "results")

    output_data = {
        "config": {
            "benchmark": "compiler",
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "max_level": max_level,
            "latent_steps": latent_steps,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }
    if direct_results is not None:
        output_data["direct"] = {
            "summary": compute_accuracy(direct_results),
            "samples": direct_results,
        }
    if latent_results is not None:
        output_data["latent"] = {
            "summary": compute_accuracy(latent_results),
            "samples": latent_results,
        }
    if text_results is not None:
        output_data["text"] = {
            "summary": compute_accuracy(text_results),
            "samples": text_results,
        }

    save_results(output_data, output_dir, "compiler", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
