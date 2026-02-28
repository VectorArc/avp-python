#!/usr/bin/env python3
"""Supervisor (Router) benchmark: dynamic routing on MMLU.

Topology: Router --> select 1 of 4 Specialists --> Answer

4 MMLU subjects (one per category):
- STEM: elementary_mathematics
- Humanities: high_school_us_history
- Social: high_school_psychology
- Logic: formal_logic

Usage:
    # Run all modes
    python benchmarks/supervisor/run_supervisor.py --mode all --max_samples 8 --verbose

    # Direct baseline only
    python benchmarks/supervisor/run_supervisor.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/supervisor/run_supervisor.py --mode both --verbose
"""

import argparse
import os
import random
import sys

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure avp-python root is on sys.path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


SUBJECTS = [
    "elementary_mathematics",
    "high_school_us_history",
    "high_school_psychology",
    "formal_logic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervisor (Router) benchmark on MMLU"
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
    parser.add_argument("--max_samples", type=int, default=8, help="Total samples across all subjects (default: 8)")
    parser.add_argument("--latent_steps", type=int, default=10, help="Latent steps (default: 10)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory")
    return parser.parse_args()


def load_dataset(max_samples: int, seed: int = 42) -> list:
    """Load MMLU test split, balanced across 4 subjects."""
    from datasets import load_dataset as hf_load_dataset

    per_subject = max(1, max_samples // len(SUBJECTS))
    samples = []

    for subject in SUBJECTS:
        print(f"Loading MMLU subject: {subject}...")
        ds = hf_load_dataset("cais/mmlu", subject, split="test")

        subject_samples = []
        for item in ds:
            subject_samples.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "subject": subject,
            })

        # Shuffle and take per_subject samples
        rng = random.Random(seed)
        rng.shuffle(subject_samples)
        samples.extend(subject_samples[:per_subject])

    # Shuffle the combined dataset
    rng = random.Random(seed)
    rng.shuffle(samples)

    # Trim to max_samples
    samples = samples[:max_samples]
    print(f"Loaded {len(samples)} samples across {len(SUBJECTS)} subjects.")
    return samples


def run_benchmark(config: dict) -> dict:
    """Run Supervisor (MMLU) benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 8)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 256)
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
    print(f"Latent steps: {latent_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(max_samples, seed)
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.supervisor.pipeline_direct import run_direct_benchmark

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
        from benchmarks.supervisor.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (Router + Specialist) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_latent:
        from benchmarks.supervisor.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP Router + Specialist) pipeline...")
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
    from benchmarks.supervisor.evaluate import compute_accuracy, print_routing_summary
    from benchmarks.shared.evaluate_common import print_summary

    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    print_summary(
        benchmark_name="Supervisor (MMLU)",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
    )

    # Print routing-specific summaries
    for label, _, res in modes:
        if res and res is not direct_results:
            print(f"\n--- {label} Mode ---")
            print_routing_summary(res)

    # Save results
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "benchmarks", "results")

    output_data = {
        "config": {
            "benchmark": "supervisor",
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "subjects": SUBJECTS,
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

    save_results(output_data, output_dir, "supervisor", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
