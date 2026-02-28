#!/usr/bin/env python3
"""HotpotQA multi-hop QA benchmark for AVP.

Tests AVP on reading comprehension with natural 2-agent (Finder → Answerer)
or 3-agent (Decomposer → Finder → Answerer) setup.

Usage:
    # Run all modes with 2 agents
    python benchmarks/hotpotqa/run_hotpotqa.py --mode all --max_samples 10 --verbose

    # Run with 3 agents
    python benchmarks/hotpotqa/run_hotpotqa.py --mode all --num_agents 3 --verbose

    # Latent vs text comparison
    python benchmarks/hotpotqa/run_hotpotqa.py --mode both --verbose
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
        description="HotpotQA multi-hop QA benchmark for AVP"
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
    parser.add_argument("--num_agents", type=int, choices=[2, 3], default=2, help="Number of agents (default: 2)")
    parser.add_argument("--latent_steps", type=int, default=10, help="Latent steps per non-terminal agent (default: 10)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation (default: 256)")
    parser.add_argument("--max_context_tokens", type=int, default=1500, help="Max context tokens safety valve (default: 1500)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details")
    parser.add_argument("--output_dir", type=str, default=None, help="Results directory")
    return parser.parse_args()


def load_dataset(max_samples: int, max_context_tokens: int) -> list:
    """Load HotpotQA validation split (distractor setting).

    Each sample has 10 context paragraphs, a multi-hop question, and a short answer.
    Filters out samples where context exceeds max_context_tokens (rough estimate).
    """
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading HotpotQA validation split (up to {max_samples} samples)...")
    ds = hf_load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    samples = []
    skipped = 0
    for item in ds:
        if len(samples) >= max_samples:
            break

        # Parse context: list of (title, sentences) pairs
        context = item["context"]
        paragraphs = []
        for title, sentences in zip(context["title"], context["sentences"]):
            paragraphs.append({"title": title, "sentences": sentences})

        # Rough token count estimate (4 chars ≈ 1 token)
        total_chars = sum(
            len(p["title"]) + sum(len(s) for s in p["sentences"])
            for p in paragraphs
        )
        estimated_tokens = total_chars // 4
        if estimated_tokens > max_context_tokens:
            skipped += 1
            continue

        samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "type": item["type"],
            "level": item["level"],
        })

    print(f"Loaded {len(samples)} samples (skipped {skipped} over context limit).")
    return samples


def print_hotpotqa_summary(
    modes: list,
    text_results=None,
    direct_results=None,
) -> str:
    """Print HotpotQA-specific summary with EM and F1."""
    from benchmarks.shared.evaluate_common import _get_field, _mean

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("HotpotQA Benchmark Results")
    lines.append("=" * 80)

    header = f"| {'Metric':<30} |"
    sep = f"|{'-' * 32}|"
    for label, w, _ in modes:
        header += f" {label:>{w}} |"
        sep += f"{'-' * (w + 2)}|"
    lines.append(header)
    lines.append(sep)

    # Exact Match
    row = f"| {'Exact Match':<30} |"
    for _, w, res in modes:
        em_count = sum(1 for r in res if r.get("exact_match", False))
        total = len(res)
        pct = em_count / total if total > 0 else 0
        cell = f"{pct:.1%} ({em_count}/{total})"
        row += f" {cell:>{w}} |"
    lines.append(row)

    # Token F1
    row = f"| {'Token F1':<30} |"
    for _, w, res in modes:
        f1_scores = [r.get("f1", 0.0) for r in res]
        mean_f1 = _mean(f1_scores)
        row += f" {mean_f1:>{w}.3f} |"
    lines.append(row)

    # Mean time/sample
    row = f"| {'Mean time/sample (s)':<30} |"
    for _, w, res in modes:
        t = _mean(_get_field(res, "wall_time"))
        row += f" {t:>{w}.1f} |"
    lines.append(row)

    # Total tokens/sample
    row = f"| {'Mean total tokens/sample':<30} |"
    for _, w, res in modes:
        tok = _get_field(res, "total_tokens")
        if tok:
            row += f" {_mean(tok):>{w},.0f} |"
        else:
            row += f" {'N/A':>{w}} |"
    lines.append(row)

    # AVP codec overhead
    any_codec = any(_get_field(res, "codec_overhead_ms") for _, _, res in modes)
    if any_codec:
        row = f"| {'Mean AVP codec overhead (ms)':<30} |"
        for _, w, res in modes:
            oh = _get_field(res, "codec_overhead_ms")
            if oh:
                row += f" {_mean(oh):>{w}.1f} |"
            else:
                row += f" {'N/A':>{w}} |"
        lines.append(row)

    lines.append(sep)

    # Token savings
    text_tok = _mean(_get_field(text_results, "total_tokens")) if text_results else 0
    if text_tok > 0:
        lines.append("")
        lines.append("Token Savings vs Text Baseline:")
        for label, _, res in modes:
            if res is text_results or res is direct_results:
                continue
            mode_tok = _mean(_get_field(res, "total_tokens"))
            if mode_tok > 0:
                saved = text_tok - mode_tok
                pct = saved / text_tok * 100 if text_tok > 0 else 0
                lines.append(f"  {label}: {mode_tok:,.0f} vs {text_tok:,.0f} text tokens "
                             f"({pct:+.1f}% {'saved' if saved > 0 else 'more'})")

    lines.append("")
    lines.append("=" * 80)
    output = "\n".join(lines)
    print(output)
    return output


def run_benchmark(config: dict) -> dict:
    """Run HotpotQA benchmark. Returns results dict."""
    from benchmarks.shared.model_utils import auto_device, load_model, set_seed

    seed = config.get("seed", 42)
    set_seed(seed)

    device = auto_device(config.get("device"))
    mode = config.get("mode", "all")
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    max_samples = config.get("max_samples", 10)
    num_agents = config.get("num_agents", 2)
    latent_steps = config.get("latent_steps", 10)
    max_new_tokens = config.get("max_new_tokens", 256)
    max_context_tokens = config.get("max_context_tokens", 1500)
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
    print(f"Agents: {num_agents}")
    print(f"Latent steps: {latent_steps}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Max context tokens: {max_context_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Pipelines: direct={run_direct}, text={run_text}, latent={run_latent}")
    print()

    dataset = load_dataset(max_samples, max_context_tokens)
    model, tokenizer, connector, identity = load_model(model_name, device)

    direct_results = None
    latent_results = None
    text_results = None

    if run_direct:
        from benchmarks.hotpotqa.pipeline_direct import run_direct_benchmark

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
        from benchmarks.hotpotqa.pipeline_text import run_text_benchmark

        print("\n" + "=" * 50)
        agent_label = f"{num_agents}-agent"
        print(f"Running TEXT ({agent_label} chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        text_results = run_text_benchmark(
            model=model, tokenizer=tokenizer, device=device, dataset=dataset,
            num_agents=num_agents,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    if run_latent:
        from benchmarks.hotpotqa.pipeline_latent import run_latent_benchmark

        print("\n" + "=" * 50)
        agent_label = f"{num_agents}-agent"
        print(f"Running LATENT (AVP {agent_label} chain) pipeline...")
        print("=" * 50)
        set_seed(seed)

        latent_results = run_latent_benchmark(
            connector=connector, model=model, tokenizer=tokenizer,
            device=device, identity=identity, dataset=dataset,
            model_name=model_name, num_agents=num_agents,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, verbose=verbose,
        )

    # Print summary
    modes = []
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    print_hotpotqa_summary(modes, text_results=text_results, direct_results=direct_results)

    # Save results
    from benchmarks.hotpotqa.evaluate import compute_accuracy
    from benchmarks.shared.results import save_results

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "benchmarks", "results")

    output_data = {
        "config": {
            "benchmark": "hotpotqa",
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "max_samples": max_samples,
            "num_agents": num_agents,
            "latent_steps": latent_steps,
            "max_new_tokens": max_new_tokens,
            "max_context_tokens": max_context_tokens,
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

    save_results(output_data, output_dir, "hotpotqa", model_name, mode, max_samples)

    return output_data


def main() -> None:
    args = parse_args()
    run_benchmark(vars(args))


if __name__ == "__main__":
    main()
