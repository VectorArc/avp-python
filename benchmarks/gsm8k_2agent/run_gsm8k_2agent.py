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

    # Cross-model rosetta mode
    python benchmarks/gsm8k_2agent/run_gsm8k_2agent.py --mode rosetta --model_name Qwen/Qwen2.5-1.5B-Instruct --model_b Qwen/Qwen2.5-0.5B-Instruct --verbose
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
        choices=["latent", "text", "direct", "rosetta", "logit_guided",
                 "mid_layer", "trained", "text_cross_model", "both", "all"],
        default="all",
        help="Pipeline(s) to run (default: all)",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for primary model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--model_b", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Second model for rosetta mode (default: Qwen/Qwen2.5-0.5B-Instruct)",
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
    parser.add_argument("--projection_temperature", type=float, default=1.0,
                        help="Softmax temperature for cross-model projection (default: 1.0)")
    parser.add_argument("--num_transfer_states", type=int, default=1,
                        help="Number of hidden states to transfer in rosetta mode (default: 1)")
    parser.add_argument("--logit_bias_alpha", type=float, default=0.5,
                        help="Logit bias scaling factor for logit_guided mode (default: 0.5)")
    parser.add_argument("--logit_bias_confidence_threshold", type=float, default=0.8,
                        help="Confidence threshold for logit bias gating (default: 0.8)")
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


def run_benchmark(config: dict) -> dict:
    """Run GSM8K 2-agent benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 512)
    temperature = config.get("temperature", 0.7)
    top_p = config.get("top_p", 0.95)
    verbose = config.get("verbose", False)
    output_dir = config.get("output_dir")
    projection_temperature = config.get("projection_temperature", 1.0)
    num_transfer_states = config.get("num_transfer_states", 1)
    logit_bias_alpha = config.get("logit_bias_alpha", 0.5)
    logit_bias_confidence_threshold = config.get("logit_bias_confidence_threshold", 0.8)

    model_b_name = config.get("model_b", "Qwen/Qwen2.5-0.5B-Instruct")

    run_direct = mode in ("direct", "all")
    run_latent = mode in ("latent", "both", "all")
    run_text = mode in ("text", "both", "all")
    run_rosetta = mode in ("rosetta", "all")
    run_logit_guided = mode in ("logit_guided", "all")
    run_mid_layer = mode in ("mid_layer", "all")
    run_trained = mode in ("trained",)  # not in "all" — requires training
    run_text_cross_model = mode in ("text_cross_model", "all")

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Model A: {model_name}")
    if run_rosetta or run_logit_guided or run_mid_layer or run_trained or run_text_cross_model:
        print(f"Model B: {model_b_name}")
    print(f"Samples: {max_samples}")
    print(f"Latent steps: {latent_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}, "
          f"rosetta={run_rosetta}, logit_guided={run_logit_guided}, "
          f"mid_layer={run_mid_layer}, trained={run_trained}, "
          f"text_cross_model={run_text_cross_model}")
    print()

    dataset = load_dataset(max_samples)
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None
    rosetta_results = None
    logit_guided_results = None
    mid_layer_results = None
    trained_results = None
    text_cross_model_results = None

    if run_direct:
        from benchmarks.gsm8k_2agent.pipeline_direct import run_direct_benchmark

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
        from benchmarks.gsm8k_2agent.pipeline_text import run_text_benchmark

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
        from benchmarks.gsm8k_2agent.pipeline_latent import run_latent_benchmark

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

    # Load model B if needed for cross-model modes
    model_b = tokenizer_b = connector_b = identity_b = None
    if run_rosetta or run_logit_guided or run_mid_layer or run_trained or run_text_cross_model:
        model_b, tokenizer_b, connector_b, identity_b = load_model(model_b_name, device)

    if run_text_cross_model:
        from benchmarks.gsm8k_2agent.pipeline_text_cross_model import run_text_cross_model_benchmark

        print("\n" + "=" * 50)
        print("Running TEXT CROSS-MODEL (A generates text → B reads text) pipeline...")
        print(f"  Model A (Researcher): {model_name}")
        print(f"  Model B (Solver):     {model_b_name}")
        print("=" * 50)
        set_seed(seed)

        text_cross_model_results = run_text_cross_model_benchmark(
            model_a=model, tokenizer_a=tokenizer,
            model_b=model_b, tokenizer_b=tokenizer_b,
            device=device, dataset=dataset,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_rosetta:
        from benchmarks.gsm8k_2agent.pipeline_rosetta import run_rosetta_benchmark
        from avp.rosetta.calibrate import calibrate

        print("\n" + "=" * 50)
        print("Running ROSETTA (cross-model projection) pipeline...")
        print(f"  Model A (Researcher): {model_name}")
        print(f"  Model B (Solver):     {model_b_name}")
        print("=" * 50)
        set_seed(seed)

        # Calibrate once — instant for same-family vocab-mediated
        print("Calibrating Rosetta Stone projection...")
        avp_map = calibrate(
            source_model=model, target_model=model_b,
            source_tokenizer=tokenizer, target_tokenizer=tokenizer_b,
            device=device,
        )
        print(f"  Method: {avp_map.method.value}, "
              f"validation_score: {avp_map.validation_score:.4f}, "
              f"{avp_map.source_dim}d → {avp_map.target_dim}d")

        rosetta_results = run_rosetta_benchmark(
            conn_a=connector, model_a=model, tokenizer_a=tokenizer,
            identity_a=identity, model_b=model_b, tokenizer_b=tokenizer_b,
            device=device, avp_map=avp_map, dataset=dataset,
            latent_steps=latent_steps, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, verbose=verbose,
            projection_temperature=projection_temperature,
            num_transfer_states=num_transfer_states,
        )

    if run_logit_guided:
        from benchmarks.gsm8k_2agent.pipeline_logit_guided import run_logit_guided_benchmark
        from avp.rosetta.calibrate import calibrate

        print("\n" + "=" * 50)
        print("Running LOGIT-GUIDED (cross-model logit bias) pipeline...")
        print(f"  Model A (Researcher): {model_name}")
        print(f"  Model B (Solver):     {model_b_name}")
        print(f"  Alpha: {logit_bias_alpha}")
        print(f"  Confidence threshold: {logit_bias_confidence_threshold}")
        print("=" * 50)
        set_seed(seed)

        # Calibrate (reuse if already done for rosetta)
        if 'avp_map' not in dir() or avp_map is None:
            print("Calibrating Rosetta Stone projection...")
            avp_map = calibrate(
                source_model=model, target_model=model_b,
                source_tokenizer=tokenizer, target_tokenizer=tokenizer_b,
                device=device,
            )
            print(f"  Method: {avp_map.method.value}, "
                  f"validation_score: {avp_map.validation_score:.4f}, "
                  f"{avp_map.source_dim}d → {avp_map.target_dim}d")

        logit_guided_results = run_logit_guided_benchmark(
            conn_a=connector, model_a=model, tokenizer_a=tokenizer,
            identity_a=identity, model_b=model_b, tokenizer_b=tokenizer_b,
            device=device, avp_map=avp_map, dataset=dataset,
            latent_steps=latent_steps, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, verbose=verbose,
            logit_bias_alpha=logit_bias_alpha,
            logit_bias_confidence_threshold=logit_bias_confidence_threshold,
        )

    if run_mid_layer:
        from benchmarks.gsm8k_2agent.pipeline_mid_layer import run_mid_layer_benchmark
        from avp.rosetta.calibrate import calibrate

        print("\n" + "=" * 50)
        print("Running MID-LAYER (cross-model mid-layer injection) pipeline...")
        print(f"  Model A (Researcher): {model_name}")
        print(f"  Model B (Solver):     {model_b_name}")
        print(f"  Depth ratio: 0.75")
        print("=" * 50)
        set_seed(seed)

        # Calibrate (reuse if already done)
        if 'avp_map' not in dir() or avp_map is None:
            print("Calibrating Rosetta Stone projection...")
            avp_map = calibrate(
                source_model=model, target_model=model_b,
                source_tokenizer=tokenizer, target_tokenizer=tokenizer_b,
                device=device,
            )
            print(f"  Method: {avp_map.method.value}, "
                  f"validation_score: {avp_map.validation_score:.4f}, "
                  f"{avp_map.source_dim}d -> {avp_map.target_dim}d")

        mid_layer_results = run_mid_layer_benchmark(
            conn_a=connector, model_a=model, tokenizer_a=tokenizer,
            identity_a=identity, model_b=model_b, tokenizer_b=tokenizer_b,
            device=device, avp_map=avp_map, dataset=dataset,
            latent_steps=latent_steps, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, verbose=verbose,
        )

    if run_trained:
        from benchmarks.gsm8k_2agent.pipeline_trained import run_trained_benchmark
        from avp.rosetta.train import train_projector, TrainConfig

        print("\n" + "=" * 50)
        print("Running TRAINED (per-layer learned projection) pipeline...")
        print(f"  Model A (Researcher): {model_name}")
        print(f"  Model B (Solver):     {model_b_name}")
        print("=" * 50)

        # Training phase
        train_config = TrainConfig(
            num_samples=config.get("train_samples", 2000),
            batch_size=config.get("train_batch_size", 4),
            num_epochs=config.get("train_epochs", 2),
            learning_rate=config.get("train_lr", 1e-4),
        )
        print(f"Training projector: {train_config.num_samples} samples, "
              f"{train_config.num_epochs} epochs...")

        trained_map = train_projector(
            source_model=model,
            target_model=model_b,
            source_tokenizer=tokenizer,
            target_tokenizer=tokenizer_b,
            device=device,
            config=train_config,
        )
        active = [i for i, g in enumerate(trained_map.layer_gates) if g > 0.01]
        print(f"Training complete. Active layers: {len(active)}/{len(trained_map.layer_gates)}")
        print(f"Validation score: {trained_map.validation_score:.4f}")

        set_seed(seed)

        trained_results = run_trained_benchmark(
            conn_a=connector, model_a=model, tokenizer_a=tokenizer,
            identity_a=identity, model_b=model_b, tokenizer_b=tokenizer_b,
            device=device, avp_map=trained_map, dataset=dataset,
            latent_steps=latent_steps, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, verbose=verbose,
        )

    # Free model B to reclaim GPU memory
    if model_b is not None:
        del model_b, tokenizer_b, connector_b, identity_b
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    # Print summary
    from benchmarks.shared.evaluate_common import print_summary, compute_stats, compute_agreement

    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))
    if rosetta_results is not None:
        modes.append(("Rosetta", 13, rosetta_results))
    if logit_guided_results is not None:
        modes.append(("Logit-Guided", 13, logit_guided_results))
    if mid_layer_results is not None:
        modes.append(("Mid-Layer", 13, mid_layer_results))
    if trained_results is not None:
        modes.append(("Trained", 13, trained_results))
    if text_cross_model_results is not None:
        modes.append(("Text Cross-Model", 16, text_cross_model_results))

    # Compute agreement across available modes
    available = {}
    if direct_results is not None:
        available["direct"] = direct_results
    if text_results is not None:
        available["text"] = text_results
    if latent_results is not None:
        available["latent"] = latent_results
    if rosetta_results is not None:
        available["rosetta"] = rosetta_results
    if logit_guided_results is not None:
        available["logit_guided"] = logit_guided_results
    if mid_layer_results is not None:
        available["mid_layer"] = mid_layer_results
    if trained_results is not None:
        available["trained"] = trained_results
    if text_cross_model_results is not None:
        available["text_cross_model"] = text_cross_model_results
    agreement_data = compute_agreement(available) if len(available) > 1 else None

    print_summary(
        benchmark_name="GSM8K 2-Agent",
        modes=modes,
        text_results=text_results,
        direct_results=direct_results,
        agreement=agreement_data,
    )

    # Save results
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "benchmarks", "results")

    output_data = {
        "config": {
            "benchmark": "gsm8k_2agent",
            "model_a": model_name,
            "model_b": model_b_name if (run_rosetta or run_logit_guided or run_mid_layer or run_trained or run_text_cross_model) else None,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "latent_steps": latent_steps,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
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
    if rosetta_results is not None:
        output_data["rosetta"] = {
            "summary": compute_stats(rosetta_results),
            "samples": rosetta_results,
        }
    if logit_guided_results is not None:
        output_data["logit_guided"] = {
            "summary": compute_stats(logit_guided_results),
            "samples": logit_guided_results,
        }
    if mid_layer_results is not None:
        output_data["mid_layer"] = {
            "summary": compute_stats(mid_layer_results),
            "samples": mid_layer_results,
        }
    if trained_results is not None:
        output_data["trained"] = {
            "summary": compute_stats(trained_results),
            "samples": trained_results,
        }
    if text_cross_model_results is not None:
        output_data["text_cross_model"] = {
            "summary": compute_stats(text_cross_model_results),
            "samples": text_cross_model_results,
        }
    if agreement_data is not None:
        output_data["agreement"] = agreement_data

    save_results(output_data, output_dir, "gsm8k_2agent", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
