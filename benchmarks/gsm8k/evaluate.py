"""GSM8K answer extraction and accuracy evaluation."""

import re
from typing import Dict, List, Optional


def extract_gsm8k_answer(text: str) -> Optional[str]:
    r"""Extract numeric answer from model output.

    Tries \boxed{...} first, then falls back to last number in text.
    Adapted from LatentMAS utils.py.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", content)
        if number:
            return number.group(0).replace(",", "")
        return content.strip()

    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
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


def print_summary(
    latent_results: Optional[List[Dict]] = None,
    text_results: Optional[List[Dict]] = None,
) -> str:
    """Print a formatted comparison table and return it as a string."""
    lines = []
    lines.append("")
    lines.append("=" * 65)
    lines.append("GSM8K Benchmark Results")
    lines.append("=" * 65)

    header = f"| {'Metric':<35} |"
    sep = f"|{'-' * 37}|"
    if latent_results is not None:
        header += f" {'Latent (AVP)':>13} |"
        sep += f"{'-' * 15}|"
    if text_results is not None:
        header += f" {'Text (baseline)':>15} |"
        sep += f"{'-' * 17}|"

    lines.append(header)
    lines.append(sep)

    # Accuracy
    row = f"| {'Accuracy':<35} |"
    if latent_results is not None:
        lat = compute_accuracy(latent_results)
        row += f" {lat['accuracy']:.1%} ({lat['correct']}/{lat['total']}) |"
    if text_results is not None:
        txt = compute_accuracy(text_results)
        row += f"   {txt['accuracy']:.1%} ({txt['correct']}/{txt['total']}) |"
    lines.append(row)

    # Mean time
    row = f"| {'Mean time/sample (s)':<35} |"
    if latent_results is not None:
        times = [r["wall_time"] for r in latent_results if "wall_time" in r]
        mean_t = sum(times) / len(times) if times else 0
        row += f" {mean_t:>13.1f} |"
    if text_results is not None:
        times = [r["wall_time"] for r in text_results if "wall_time" in r]
        mean_t = sum(times) / len(times) if times else 0
        row += f" {mean_t:>15.1f} |"
    lines.append(row)

    # AVP-specific metrics
    if latent_results is not None:
        overheads = [r["codec_overhead_ms"] for r in latent_results if "codec_overhead_ms" in r]
        mean_oh = sum(overheads) / len(overheads) if overheads else 0
        row = f"| {'Mean AVP codec overhead (ms)':<35} |"
        row += f" {mean_oh:>13.1f} |"
        if text_results is not None:
            row += f" {'N/A':>15} |"
        lines.append(row)

        wire = [r["avp_wire_bytes"] for r in latent_results if "avp_wire_bytes" in r]
        mean_w = sum(wire) / len(wire) if wire else 0
        row = f"| {'Mean AVP wire bytes/sample':<35} |"
        row += f" {mean_w:>13,.0f} |"
        if text_results is not None:
            row += f" {'N/A':>15} |"
        lines.append(row)

        seq_lens = [r["kv_seq_len_judger"] for r in latent_results if "kv_seq_len_judger" in r]
        mean_s = sum(seq_lens) / len(seq_lens) if seq_lens else 0
        row = f"| {'Mean KV seq len (judger)':<35} |"
        row += f" {mean_s:>13.0f} |"
        if text_results is not None:
            row += f" {'N/A':>15} |"
        lines.append(row)

    lines.append("=" * 65)
    output = "\n".join(lines)
    print(output)
    return output
