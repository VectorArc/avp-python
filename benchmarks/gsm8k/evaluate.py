"""GSM8K answer extraction and accuracy evaluation."""

import re
from typing import Dict, List, Optional


def extract_gsm8k_answer(text: str) -> Optional[str]:
    r"""Extract numeric answer from model output.

    Priority order:
    1. \boxed{...} — the explicitly requested format
    2. "#### <number>" — GSM8K convention some models learn
    3. "the answer is <number>" — common natural language pattern
    4. "= <number>" at end of line — calculation conclusion
    5. Last number in text — fallback

    Adapted from LatentMAS utils.py.
    """
    _NUM = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"

    # 1. \boxed{...}
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        # Strip \text{} wrapper if present (model sometimes wraps non-numeric text)
        text_inner = re.search(r"\\text\{([^}]*)\}", content)
        if text_inner:
            content = text_inner.group(1)
        number = re.search(_NUM, content)
        if number:
            return number.group(0).replace(",", "")
        # No number in \boxed{} — fall through to other extraction patterns

    # 2. #### <number> (GSM8K convention)
    hashes = re.findall(r"####\s*(" + _NUM + r")", text)
    if hashes:
        return hashes[-1].replace(",", "")

    # 3. "the answer is <number>" / "the final answer is <number>"
    answer_match = re.findall(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?\s*(" + _NUM + r")",
        text, re.IGNORECASE,
    )
    if answer_match:
        return answer_match[-1].replace(",", "")

    # 4. Last number in text
    numbers = re.findall(_NUM, text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_gold(solution: str) -> Optional[str]:
    """Extract gold answer from GSM8K solution string (#### <number>)."""
    match = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", solution)
    if match:
        return match.group(1).replace(",", "")
    return None


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
    """Check if prediction matches gold answer."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(gold)
    if pred is None or gold is None:
        return False
    return pred == gold


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


def _get_field(results: List[Dict], key: str) -> list:
    """Extract non-None values for a key from results."""
    return [r[key] for r in results if r.get(key) is not None]


def print_summary(
    latent_results: Optional[List[Dict]] = None,
    text_results: Optional[List[Dict]] = None,
    direct_results: Optional[List[Dict]] = None,
    hybrid_results: Optional[List[Dict]] = None,
) -> str:
    """Print a formatted comparison summary and return it as a string."""
    lines = []

    # Collect all available modes
    modes: List[tuple] = []  # (label, col_width, results)
    if direct_results is not None:
        modes.append(("Direct", 13, direct_results))
    if latent_results is not None:
        modes.append(("Latent (AVP)", 13, latent_results))
    if hybrid_results is not None:
        modes.append(("Hybrid", 13, hybrid_results))
    if text_results is not None:
        modes.append(("Text", 13, text_results))

    if not modes:
        return ""

    # --- Main comparison table ---
    lines.append("")
    lines.append("=" * 80)
    lines.append("GSM8K Benchmark Results")
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

    # --- Token savings (latent/hybrid vs text) ---
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

    # --- Redundant context (text pipeline "communication tax") ---
    if text_results:
        ctx_tokens = _get_field(text_results, "total_context_tokens")
        if ctx_tokens:
            mean_ctx = _mean(ctx_tokens)
            lines.append(f"  Text context re-processing: {mean_ctx:,.0f} tokens/sample "
                         f"(eliminated by AVP latent transfer)")

    # --- Per-hop latency breakdown ---
    # Show for any multi-agent mode that has agent traces
    hop_modes = []
    for label, _, res in modes:
        if res is direct_results:
            continue
        # Check if agent traces have agent_time_ms
        if res and res[0].get("agents"):
            sample_agents = res[0]["agents"]
            if any("agent_time_ms" in a for a in sample_agents):
                hop_modes.append((label, res))

    if hop_modes:
        lines.append("")
        lines.append("Per-Hop Latency (mean across samples):")

        # Get agent names from first result's traces
        agent_names = [a["name"] for a in hop_modes[0][1][0]["agents"]]
        header = f"  {'Agent':<12}"
        for label, _ in hop_modes:
            header += f" {label:>14}"
        lines.append(header)
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

    # --- Cost estimation ---
    # Two tiers: budget ($0.15/$0.60 per M tok) and standard ($3/$15 per M tok)
    cost_modes = [(label, res) for label, _, res in modes
                  if _get_field(res, "total_tokens")]
    if cost_modes:
        lines.append("")
        lines.append("Estimated Cost per Query (API pricing):")

        budget_input, budget_output = 0.15, 0.60   # $ per M tokens
        standard_input, standard_output = 3.00, 15.00

        header = f"  {'Tier':<12}"
        for label, _ in cost_modes:
            header += f" {label:>14}"
        lines.append(header)
        lines.append(f"  {'-' * 12}" + f" {'-' * 14}" * len(cost_modes))

        for tier_name, inp_rate, out_rate in [
            ("Budget", budget_input, budget_output),
            ("Standard", standard_input, standard_output),
        ]:
            row = f"  {tier_name:<12}"
            for label, res in cost_modes:
                # Use prompt tokens as input, output tokens as output
                input_tok = _mean(_get_field(res, "total_prompt_tokens")) if _get_field(res, "total_prompt_tokens") else _mean(_get_field(res, "prompt_tokens")) if _get_field(res, "prompt_tokens") else 0
                output_tok = _mean(_get_field(res, "total_output_tokens")) if _get_field(res, "total_output_tokens") else _mean(_get_field(res, "output_tokens")) if _get_field(res, "output_tokens") else 0
                cost = (input_tok * inp_rate + output_tok * out_rate) / 1_000_000
                row += f"  ${cost:>11.6f}"
            lines.append(row)

        # GPU-time cost for self-hosted
        lines.append("")
        lines.append("Estimated Cost per Query (self-hosted GPU):")
        gpu_rates = [("H100 cloud", 3.00), ("RTX consumer", 0.10)]  # $/hour
        header = f"  {'GPU':<12}"
        for label, _ in cost_modes:
            header += f" {label:>14}"
        lines.append(header)
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
