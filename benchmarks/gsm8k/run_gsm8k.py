#!/usr/bin/env python3
"""GSM8K benchmark: 4-agent chain comparing latent (AVP) vs text mode.

Usage:
    # Run all modes (direct baseline + text chain + latent chain)
    python benchmarks/gsm8k/run_gsm8k.py --mode all --max_samples 10 --verbose

    # Direct single-agent baseline only (to check model capability)
    python benchmarks/gsm8k/run_gsm8k.py --mode direct --verbose

    # Latent vs text comparison
    python benchmarks/gsm8k/run_gsm8k.py --mode both --verbose
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

# Fix Windows console encoding for model output containing Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
        choices=["latent", "text", "direct", "hybrid", "both", "all"],
        default="all",
        help="Pipeline(s) to run. 'direct' = single-agent baseline, "
             "'hybrid' = latent+text summary, "
             "'both' = latent+text chains, 'all' = direct+text+latent+hybrid (default: all)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
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
        "--kv_mode",
        choices=["full", "sequential", "latent_only"],
        default="full",
        help="KV-cache content policy between hops: full (keep all), "
             "sequential (keep current agent only), latent_only (keep latent "
             "steps only) (default: full)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens for text generation (default: 512)",
    )
    parser.add_argument(
        "--summary_max_tokens",
        type=int,
        default=128,
        help="Max tokens for hybrid text summary per hop (default: 128)",
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

    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
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


def run_benchmark(config: dict) -> dict:
    """Run GSM8K 4-agent benchmark. Returns results dict."""
    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    latent_steps = config.get("latent_steps", 10)
    kv_mode = config.get("kv_mode", "full")
    max_new_tokens = config.get("max_new_tokens", 512)
    summary_max_tokens = config.get("summary_max_tokens", 128)
    temperature = config.get("temperature", 0.7)
    top_p = config.get("top_p", 0.95)
    verbose = config.get("verbose", False)
    output_dir = config.get("output_dir")

    run_direct = mode in ("direct", "all")
    run_latent = mode in ("latent", "both", "all")
    run_text = mode in ("text", "both", "all")
    run_hybrid = mode in ("hybrid", "all")

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Samples: {max_samples}")
    print(f"Latent steps: {latent_steps}")
    print(f"KV mode: {kv_mode}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}, hybrid={run_hybrid}")
    print()

    # Load dataset
    dataset = load_dataset(max_samples)

    # Load model
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None
    hybrid_results = None

    # Run direct baseline
    if run_direct:
        from benchmarks.gsm8k.pipeline_direct import run_direct_benchmark

        print("\n" + "=" * 50)
        print("Running DIRECT (single-agent) baseline...")
        print("=" * 50)
        set_seed(seed)

        direct_results = run_direct_benchmark(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=dataset,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )

    # Run text pipeline
    if run_text:
        from benchmarks.gsm8k.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT (4-agent chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=dataset,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )

    # Run latent pipeline
    if run_latent:
        from benchmarks.gsm8k.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        print("Running LATENT (AVP 4-agent chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        latent_results = run_latent_benchmark(
            connector=connector,
            model=model,
            tokenizer=tokenizer,
            device=device,
            identity=identity,
            dataset=dataset,
            model_name=model_name,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            kv_mode=kv_mode,
            verbose=verbose,
        )

    # Run hybrid pipeline
    if run_hybrid:
        from benchmarks.gsm8k.pipeline_hybrid import run_hybrid_benchmark

        print("\n" + "=" * 50)
        print("Running HYBRID (AVP latent+text 4-agent chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        hybrid_results = run_hybrid_benchmark(
            connector=connector,
            model=model,
            tokenizer=tokenizer,
            device=device,
            identity=identity,
            dataset=dataset,
            model_name=model_name,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            summary_max_tokens=summary_max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )

    # Print comprehensive summary
    from benchmarks.gsm8k.evaluate import compute_accuracy, print_summary
    from benchmarks.shared.evaluate_common import compute_agreement

    # Compute agreement across available modes
    available = {}
    if direct_results is not None:
        available["direct"] = direct_results
    if text_results is not None:
        available["text"] = text_results
    if latent_results is not None:
        available["latent"] = latent_results
    if hybrid_results is not None:
        available["hybrid"] = hybrid_results
    agreement_data = compute_agreement(available) if len(available) > 1 else None

    print_summary(
        latent_results=latent_results,
        text_results=text_results,
        direct_results=direct_results,
        hybrid_results=hybrid_results,
        agreement=agreement_data,
    )

    # Save results
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "results")

    os.makedirs(output_dir, exist_ok=True)

    # Timestamped filename so runs don't overwrite each other
    from datetime import datetime
    model_short = model_name.split("/")[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gsm8k_{model_short}_{mode}_n{max_samples}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    output_data = {
        "config": {
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "latent_steps": latent_steps,
            "kv_mode": kv_mode,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "timestamp": timestamp,
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
    if hybrid_results is not None:
        output_data["hybrid"] = {
            "summary": compute_accuracy(hybrid_results),
            "samples": hybrid_results,
        }
    if agreement_data is not None:
        output_data["agreement"] = agreement_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Also write a latest symlink/copy for easy access
    latest_path = os.path.join(output_dir, "gsm8k_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Latest copy: {latest_path}")

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
