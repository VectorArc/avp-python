"""Common evaluation utilities shared across benchmarks."""

import math
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


def compute_stats(results: List[Dict]) -> Dict:
    """Compute accuracy with Wilson score 95% CI.

    Superset of compute_accuracy() — returns all existing keys plus CI bounds.
    """
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "ci_95_lo": 0.0, "ci_95_hi": 0.0}

    k = sum(1 for r in results if r.get("correct", False))
    acc = k / n
    # Wilson score interval
    z = 1.96
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom
    return {
        "accuracy": acc,
        "correct": k,
        "total": n,
        "ci_95_lo": max(0.0, center - margin),
        "ci_95_hi": min(1.0, center + margin),
    }


def _chi2_sf_1df(x: float) -> float:
    """Survival function for chi-squared distribution with 1 degree of freedom.

    Uses erfc: P(X > x) = erfc(sqrt(x/2) / sqrt(2)) for df=1,
    which simplifies to erfc(sqrt(x/2)).
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


def mcnemar_test(
    results_a: List[Dict],
    results_b: List[Dict],
    label_a: str = "A",
    label_b: str = "B",
) -> Dict:
    """McNemar's test for paired nominal data (continuity-corrected).

    Compares two modes on the same samples (aligned by list index).
    Returns 2x2 contingency counts, chi-square statistic, and p-value.
    """
    n = min(len(results_a), len(results_b))
    both_correct = a_only = b_only = both_wrong = 0
    for i in range(n):
        a_ok = results_a[i].get("correct", False)
        b_ok = results_b[i].get("correct", False)
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok:
            a_only += 1
        elif b_ok:
            b_only += 1
        else:
            both_wrong += 1

    discordant = a_only + b_only
    if discordant == 0:
        return {
            "both_correct": both_correct, "both_wrong": both_wrong,
            f"{label_a}_only": a_only, f"{label_b}_only": b_only,
            "chi2": 0.0, "p_value": 1.0, "significant": False,
        }

    chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant  # continuity-corrected
    p = _chi2_sf_1df(chi2)
    return {
        "both_correct": both_correct, "both_wrong": both_wrong,
        f"{label_a}_only": a_only, f"{label_b}_only": b_only,
        "chi2": round(chi2, 4), "p_value": round(p, 4),
        "significant": p < 0.05,
    }


def compute_agreement(mode_results: Dict[str, List[Dict]]) -> Dict:
    """Per-sample concordance analysis across modes.

    Takes {mode_name: results_list}, returns agreement summary with
    pairwise McNemar tests and latent-specific failure analysis.
    """
    modes = list(mode_results.keys())
    n = min(len(v) for v in mode_results.values())

    per_sample = []
    for i in range(n):
        correct_modes = [m for m in modes if mode_results[m][i].get("correct", False)]
        per_sample.append({
            "index": i,
            "question_prefix": mode_results[modes[0]][i].get("question", "")[:80],
            "correct_by_mode": {m: mode_results[m][i].get("correct", False) for m in modes},
            "agreement": (
                "all" if len(correct_modes) == len(modes)
                else "none" if not correct_modes
                else "partial"
            ),
        })

    all_correct = sum(1 for s in per_sample if s["agreement"] == "all")
    all_wrong = sum(1 for s in per_sample if s["agreement"] == "none")

    # Pairwise concordance tables
    concordance = {}
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            concordance[f"{a}_vs_{b}"] = mcnemar_test(
                mode_results[a], mode_results[b], label_a=a, label_b=b)

    # Latent-specific failure analysis
    latent_unique_fails = None
    latent_unique_wins = None
    if "latent" in modes:
        other_modes = [m for m in modes if m != "latent"]
        latent_unique_fails = [
            s["index"] for s in per_sample
            if not s["correct_by_mode"].get("latent", False)
            and any(s["correct_by_mode"].get(m, False) for m in other_modes)
        ]
        latent_unique_wins = [
            s["index"] for s in per_sample
            if s["correct_by_mode"].get("latent", False)
            and not any(s["correct_by_mode"].get(m, False) for m in other_modes)
        ]

    return {
        "total_samples": n,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "partial_agreement": n - all_correct - all_wrong,
        "concordance": concordance,
        "latent_unique_fails": latent_unique_fails,
        "latent_unique_wins": latent_unique_wins,
        "per_sample": per_sample,
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
    agreement: Optional[Dict] = None,
) -> str:
    """Print a formatted comparison summary and return it as a string.

    Args:
        benchmark_name: Display name for the benchmark
        modes: List of (label, col_width, results) tuples for each mode
        text_results: Text mode results (for token savings comparison)
        direct_results: Direct mode results (excluded from token savings)
        agreement: Output of compute_agreement() for concordance analysis
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
    stats_per_mode = []
    for _, w, res in modes:
        stats = compute_stats(res)
        stats_per_mode.append(stats)
        cell = f"{stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})"
        row += f" {cell:>{w}} |"
    lines.append(row)

    # Accuracy 95% CI
    row = f"| {'Accuracy 95% CI':<30} |"
    for (_, w, _res), stats in zip(modes, stats_per_mode):
        cell = f"[{stats['ci_95_lo']:.0%}-{stats['ci_95_hi']:.0%}]"
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

        agent_names = [a.get("name", a.get("method_name", f"step_{i}")) for i, a in enumerate(hop_modes[0][1][0]["agents"])]
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

    # Agreement analysis
    if agreement and agreement.get("total_samples", 0) > 0:
        n_agree = agreement["total_samples"]
        lines.append("")
        lines.append(f"Agreement Analysis (N={n_agree}):")
        lines.append(f"  All modes agree correct:  {agreement['all_correct']:>3} ({agreement['all_correct']/n_agree:.0%})"
                     f"    All modes agree wrong: {agreement['all_wrong']:>3} ({agreement['all_wrong']/n_agree:.0%})")
        lines.append(f"  Partial agreement:        {agreement['partial_agreement']:>3} ({agreement['partial_agreement']/n_agree:.0%})")

        concordance = agreement.get("concordance", {})
        if concordance:
            lines.append("")
            lines.append("  Pairwise Significance (McNemar's test):")
            for pair_key, pair_data in concordance.items():
                parts = pair_key.split("_vs_")
                if len(parts) == 2:
                    la, lb = parts
                else:
                    la, lb = "A", "B"
                a_only_key = f"{la}_only"
                b_only_key = f"{lb}_only"
                a_only = pair_data.get(a_only_key, 0)
                b_only = pair_data.get(b_only_key, 0)
                lines.append(
                    f"    {pair_key}: "
                    f" both_ok={pair_data['both_correct']}"
                    f"  {la}_only={a_only}"
                    f"  {lb}_only={b_only}"
                    f"  both_wrong={pair_data['both_wrong']}"
                    f"  p={pair_data['p_value']:.4f}"
                )

        if agreement.get("latent_unique_fails") is not None:
            n_fails = len(agreement["latent_unique_fails"])
            n_wins = len(agreement.get("latent_unique_wins", []))
            lines.append("")
            s_fail = "sample" if n_fails == 1 else "samples"
            s_win = "sample" if n_wins == 1 else "samples"
            lines.append(f"  Latent unique failures: {n_fails} {s_fail} (wrong when others right)")
            lines.append(f"  Latent unique wins:     {n_wins} {s_win} (right when others wrong)")

    lines.append("")
    lines.append("=" * 80)
    output = "\n".join(lines)
    print(output)
    return output
