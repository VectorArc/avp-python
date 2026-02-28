"""MATH-500 answer extraction and accuracy evaluation."""

import re
from typing import Dict, List, Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract answer from \boxed{...}, handling nested braces.

    Handles cases like \boxed{\frac{3}{4}}, \boxed{2^{10}}, etc.
    """
    # Find all \boxed{ positions and match braces
    pattern = r"\\boxed\{"
    matches = []
    for m in re.finditer(pattern, text):
        start = m.end()  # Position after the opening {
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            matches.append(text[start:pos - 1])

    if matches:
        return matches[-1].strip()
    return None


def normalize_math_answer(ans: Optional[str]) -> Optional[str]:
    """Normalize a math answer string for comparison.

    Strips LaTeX formatting, whitespace, and trailing .0.
    """
    if ans is None:
        return None
    ans = ans.strip()
    # Remove $ signs
    ans = ans.replace("$", "")
    # Remove \text{} wrapper
    text_match = re.search(r"\\text\{([^}]*)\}", ans)
    if text_match:
        ans = text_match.group(1).strip()
    # Remove LaTeX spacing and sizing commands
    for cmd in [r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad",
                r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg"]:
        ans = ans.replace(cmd, "")
    # Remove trailing .0
    if ans.endswith(".0"):
        ans = ans[:-2]
    # Remove trailing period
    if ans.endswith("."):
        ans = ans[:-1]
    # Remove all whitespace (math notation is whitespace-insensitive)
    ans = re.sub(r"\s+", "", ans)
    return ans


def extract_answer(text: str) -> Optional[str]:
    r"""Extract answer from model output.

    Priority:
    1. \boxed{...} — the explicitly requested format
    2. "the answer is ..." pattern
    3. Last \boxed-like expression
    """
    # 1. \boxed{...}
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    # 2. "the answer is ..."
    answer_match = re.findall(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if answer_match:
        return answer_match[-1].strip()

    return None


def check_correct(prediction: Optional[str], gold: Optional[str]) -> bool:
    """Check if prediction matches gold answer."""
    pred = normalize_math_answer(prediction)
    gold_norm = normalize_math_answer(gold)
    if pred is None or gold_norm is None:
        return False
    return pred == gold_norm


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute accuracy with per-level and per-subject breakdown."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))

    # Per-level breakdown
    per_level = {}
    for r in results:
        level = r.get("level", "unknown")
        if level not in per_level:
            per_level[level] = {"correct": 0, "total": 0}
        per_level[level]["total"] += 1
        if r.get("correct", False):
            per_level[level]["correct"] += 1
    for level in per_level:
        d = per_level[level]
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    # Per-subject breakdown
    per_subject = {}
    for r in results:
        subject = r.get("subject", "unknown")
        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if r.get("correct", False):
            per_subject[subject]["correct"] += 1
    for subject in per_subject:
        d = per_subject[subject]
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_level": per_level,
        "per_subject": per_subject,
    }


def print_level_summary(results: List[Dict]) -> None:
    """Print per-level accuracy table."""
    acc = compute_accuracy(results)
    per_level = acc.get("per_level", {})
    if not per_level:
        return

    print("\nPer-Level Accuracy:")
    print(f"  {'Level':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>6}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 6}")
    for level in sorted(per_level.keys()):
        d = per_level[level]
        print(f"  {level:<12} {d['accuracy']:>9.1%} {d['correct']:>8} {d['total']:>6}")

    per_subject = acc.get("per_subject", {})
    if per_subject:
        print("\nPer-Subject Accuracy:")
        print(f"  {'Subject':<25} {'Accuracy':>10} {'Correct':>8} {'Total':>6}")
        print(f"  {'-' * 25} {'-' * 10} {'-' * 8} {'-' * 6}")
        for subject in sorted(per_subject.keys()):
            d = per_subject[subject]
            print(f"  {subject:<25} {d['accuracy']:>9.1%} {d['correct']:>8} {d['total']:>6}")
