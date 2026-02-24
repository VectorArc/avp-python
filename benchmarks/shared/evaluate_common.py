"""Common evaluation utilities shared across benchmarks."""

import re
from typing import Dict, List, Optional


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize an answer string for comparison."""
    if ans is None:
        return None
    ans = ans.strip().lower().replace(",", "")
    # Remove trailing .0 for integer comparison
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans


def check_correct(prediction: Optional[str], gold: Optional[str]) -> bool:
    """Check if prediction matches gold answer (numeric comparison)."""
    pred = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if pred is None or gold_norm is None:
        return False
    return pred == gold_norm


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute accuracy and summary statistics from a list of results."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


def _mean(vals: list) -> float:
    """Safe mean of a list, returns 0 if empty."""
    return sum(vals) / len(vals) if vals else 0


def _get_field(results: Optional[List[Dict]], key: str) -> list:
    """Extract non-None values for a key from results."""
    if results is None:
        return []
    return [r[key] for r in results if r.get(key) is not None]


def print_summary(
    benchmark_name: str,
    modes: List[tuple],
    text_results: Optional[List[Dict]] = None,
    direct_results: Optional[List[Dict]] = None,
) -> str:
    """Print a formatted comparison summary and return it as a string.

    Args:
        benchmark_name: Display name for the benchmark
        modes: List of (label, col_width, results) tuples for each mode
        text_results: Text mode results (for token savings comparison)
        direct_results: Direct mode results (excluded from token savings)
    """
    lines = []

    if not modes:
        return ""

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"{benchmark_name} Benchmark Results")
    lines.append("=" * 80)

    header = f"| {'Metric':<30} |"
    sep = f"|{'-' * 32}|"
    for label, w, _ in modes:
        header += f" {label:>{w}} |"
        sep += f"{'-' * (w + 2)}|"
    lines.append(header)
    lines.append(sep)

    # Accuracy
    row = f"| {'Accuracy':<30} |"
    for _, w, res in modes:
        acc = compute_accuracy(res)
        cell = f"{acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})"
        row += f" {cell:>{w}} |"
    lines.append(row)

    # Mean time/sample
    row = f"| {'Mean time/sample (s)':<30} |"
    for _, w, res in modes:
        t = _mean(_get_field(res, "wall_time"))
        row += f" {t:>{w}.1f} |"
    lines.append(row)

    # Tokens/sec
    row = f"| {'Mean tokens/sec':<30} |"
    has_tps = False
    for _, w, res in modes:
        tps = _get_field(res, "tokens_per_sec")
        if tps:
            row += f" {_mean(tps):>{w}.1f} |"
            has_tps = True
        else:
            row += f" {'N/A':>{w}} |"
    if has_tps:
        lines.append(row)

    # Peak GPU memory
    row = f"| {'Mean peak GPU memory (MB)':<30} |"
    has_mem = False
    for _, w, res in modes:
        mem = _get_field(res, "peak_memory_mb")
        if mem:
            row += f" {_mean(mem):>{w}.0f} |"
            has_mem = True
        else:
            row += f" {'N/A':>{w}} |"
    if has_mem:
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

    # AVP codec overhead (only for modes that have it)
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

    # Token savings (latent vs text)
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

    # Redundant context (text pipeline "communication tax")
    if text_results:
        ctx_tokens = _get_field(text_results, "total_context_tokens")
        if ctx_tokens:
            mean_ctx = _mean(ctx_tokens)
            lines.append(f"  Text context re-processing: {mean_ctx:,.0f} tokens/sample "
                         f"(eliminated by AVP latent transfer)")

    # Per-hop latency breakdown
    hop_modes = []
    for label, _, res in modes:
        if res is direct_results:
            continue
        if res and res[0].get("agents"):
            sample_agents = res[0]["agents"]
            if any("agent_time_ms" in a for a in sample_agents):
                hop_modes.append((label, res))

    if hop_modes:
        lines.append("")
        lines.append("Per-Hop Latency (mean across samples):")

        agent_names = [a["name"] for a in hop_modes[0][1][0]["agents"]]
        header_line = f"  {'Agent':<12}"
        for label, _ in hop_modes:
            header_line += f" {label:>14}"
        lines.append(header_line)
        lines.append(f"  {'-' * 12}" + f" {'-' * 14}" * len(hop_modes))

        for i, name in enumerate(agent_names):
            row = f"  {name:<12}"
            for label, res in hop_modes:
                agent_times = []
                for r in res:
                    agents = r.get("agents", [])
                    if i < len(agents) and "agent_time_ms" in agents[i]:
                        agent_times.append(agents[i]["agent_time_ms"])
                if agent_times:
                    mean_ms = _mean(agent_times)
                    row += f" {mean_ms/1000:>13.1f}s"
                else:
                    row += f" {'N/A':>14}"
            lines.append(row)

    # Cost estimation
    cost_modes = [(label, res) for label, _, res in modes
                  if _get_field(res, "total_tokens")]
    if cost_modes:
        lines.append("")
        lines.append("Estimated Cost per Query (API pricing):")

        budget_input, budget_output = 0.15, 0.60
        standard_input, standard_output = 3.00, 15.00

        header_line = f"  {'Tier':<12}"
        for label, _ in cost_modes:
            header_line += f" {label:>14}"
        lines.append(header_line)
        lines.append(f"  {'-' * 12}" + f" {'-' * 14}" * len(cost_modes))

        for tier_name, inp_rate, out_rate in [
            ("Budget", budget_input, budget_output),
            ("Standard", standard_input, standard_output),
        ]:
            row = f"  {tier_name:<12}"
            for label, res in cost_modes:
                input_tok = _mean(_get_field(res, "total_prompt_tokens")) if _get_field(res, "total_prompt_tokens") else _mean(_get_field(res, "prompt_tokens")) if _get_field(res, "prompt_tokens") else 0
                output_tok = _mean(_get_field(res, "total_output_tokens")) if _get_field(res, "total_output_tokens") else _mean(_get_field(res, "output_tokens")) if _get_field(res, "output_tokens") else 0
                cost = (input_tok * inp_rate + output_tok * out_rate) / 1_000_000
                row += f"  ${cost:>11.6f}"
            lines.append(row)

        # GPU-time cost for self-hosted
        lines.append("")
        lines.append("Estimated Cost per Query (self-hosted GPU):")
        gpu_rates = [("H100 cloud", 3.00), ("RTX consumer", 0.10)]
        header_line = f"  {'GPU':<12}"
        for label, _ in cost_modes:
            header_line += f" {label:>14}"
        lines.append(header_line)
        lines.append(f"  {'-' * 12}" + f" {'-' * 14}" * len(cost_modes))

        for gpu_name, hourly_rate in gpu_rates:
            row = f"  {gpu_name:<12}"
            for label, res in cost_modes:
                wall = _mean(_get_field(res, "wall_time"))
                cost = wall * hourly_rate / 3600
                row += f"  ${cost:>11.6f}"
            lines.append(row)

    lines.append("")
    lines.append("=" * 80)
    output = "\n".join(lines)
    print(output)
    return output
