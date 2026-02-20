#!/usr/bin/env python3
"""GSM8K benchmark: 4-agent chain comparing latent (AVP) vs text mode.

Usage:
    python benchmarks/gsm8k/run_gsm8k.py --mode both --max_samples 10 --verbose
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

# Ensure avp-python root is on sys.path when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K 4-agent benchmark: latent (AVP) vs text baseline"
    )
    parser.add_argument(
        "--mode",
        choices=["latent", "text", "both"],
        default="both",
        help="Pipeline(s) to run (default: both)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2-0.5B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu/mps/cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Number of GSM8K problems to evaluate (default: 10)",
    )
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=10,
        help="Latent steps per non-judger agent (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens for text generation (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample details",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results directory (default: benchmarks/results/)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def load_model(model_name: str, device: str):
    """Load model and tokenizer, return (model, tokenizer, connector, identity)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from avp.connectors.huggingface import HuggingFaceConnector

    print(f"Loading model: {model_name} on {device}...")
    t0 = time.perf_counter()

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    connector = HuggingFaceConnector(model=model, tokenizer=tokenizer, device=device)
    identity = connector.get_model_identity()

    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s. Identity: {identity.model_family}, "
          f"hidden_dim={identity.hidden_dim}, layers={identity.num_layers}, "
          f"kv_heads={identity.num_kv_heads}")

    return model, tokenizer, connector, identity


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = auto_device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.max_samples}")
    print(f"Latent steps: {args.latent_steps}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print()

    # Load dataset
    dataset = load_dataset(args.max_samples)

    # Load model
    model, tokenizer, connector, identity = load_model(args.model_name, device)

    latent_results = None
    text_results = None

    # Run latent pipeline
    if args.mode in ("latent", "both"):
        from benchmarks.gsm8k.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP) pipeline...")
        print("=" * 50)
        set_seed(args.seed)

        latent_results = run_latent_benchmark(
            connector=connector,
            model=model,
            tokenizer=tokenizer,
            device=device,
            identity=identity,
            dataset=dataset,
            model_name=args.model_name,
            latent_steps=args.latent_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
        )

    # Run text pipeline
    if args.mode in ("text", "both"):
        from benchmarks.gsm8k.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (baseline) pipeline...")
        print("=" * 50)
        set_seed(args.seed)

        text_results = run_text_benchmark(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=dataset,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
        )

    # Print summary
    from benchmarks.gsm8k.evaluate import print_summary

    print_summary(latent_results, text_results)

    # Save results
    output_dir = args.output_dir
    if output_dir is None:
        # Default: benchmarks/results/ relative to the avp-python dir
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "results")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k_results.json")

    # Build JSON-serializable results
    output_data = {
        "config": {
            "model_name": args.model_name,
            "device": device,
            "max_samples": args.max_samples,
            "latent_steps": args.latent_steps,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    }
    if latent_results is not None:
        from benchmarks.gsm8k.evaluate import compute_accuracy
        output_data["latent"] = {
            "summary": compute_accuracy(latent_results),
            "samples": latent_results,
        }
    if text_results is not None:
        from benchmarks.gsm8k.evaluate import compute_accuracy
        output_data["text"] = {
            "summary": compute_accuracy(text_results),
            "samples": text_results,
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
