#!/usr/bin/env python3
"""GSM8K 2-agent benchmark: Researcher → Solver handoff.

Tests the most common real-world pattern (simple delegation) on the same
dataset as the 4-agent benchmark for direct comparison.

Usage:
    # Run all modes
    python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py --mode all --max_samples 10 --verbose

    # Direct single-agent baseline only
    python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py --mode both --verbose
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
        description="GSM8K 2-agent benchmark: Researcher → Solver handoff"
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
    parser.add_argument("--latent_steps", type=int, default=10, help="Latent steps for Researcher (default: 10)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory")
    return parser.parse_args()


def load_dataset(max_samples: int) -> list:
    """Load GSM8K test split from HuggingFace datasets."""
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading GSM8K test split (first {max_samples} samples)...")
    ds = hf_load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        samples.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    print(f"Loaded {len(samples)} samples.")
    return samples


def main() -> None:
    args = parse_args()

    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    set_seed(args.seed)

    device = auto_device(args.device)
    run_direct = args.mode in ("direct", "all")
    run_latent = args.mode in ("latent", "both", "all")
    run_text = args.mode in ("text", "both", "all")

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.max_samples}")
    print(f"Latent steps: {args.latent_steps}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(args.max_samples)
    model, tokenizer, connector, identity = load_model(args.model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.gsm8k_2agent.pipeline_direct import run_direct_benchmark

        print("\n" + "=" * 50)
        print("Running DIRECT (single-agent) baseline...")
        print("=" * 50)
        set_seed(args.seed)

        direct_results = run_direct_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, verbose=args.verbose,
        )

    if run_text:
        from benchmarks.gsm8k_2agent.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (2-agent chain) pipeline...")
        print("=" * 50)
        set_seed(args.seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, verbose=args.verbose,
        )

    if run_latent:
        from benchmarks.gsm8k_2agent.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP 2-agent chain) pipeline...")
        print("=" * 50)
        set_seed(args.seed)

        latent_results = run_latent_benchmark(
            connector=connector, model=model, tokenizer=tokenizer,
            device=device, identity=identity, dataset=dataset,
            model_name=args.model_name, latent_steps=args.latent_steps,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, verbose=args.verbose,
        )

    # Print summary
    from benchmarks.shared.evaluate_common import print_summary, compute_accuracy

    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    print_summary(
        benchmark_name="GSM8K 2-Agent",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
    )

    # Save results
    from benchmarks.shared.results import save_results

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "benchmarks", "results")

    output_data = {
        "config": {
            "benchmark": "gsm8k_2agent",
            "model_name": args.model_name,
            "device": device,
            "mode": args.mode,
            "max_samples": args.max_samples,
            "latent_steps": args.latent_steps,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
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

    save_results(output_data, output_dir, "gsm8k_2agent", args.model_name, args.mode, args.max_samples)


if __name__ == "__main__":
    main()
