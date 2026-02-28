"""MMLU answer extraction and accuracy evaluation for Supervisor benchmark."""

import re
from typing import Dict, List, Optional


LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
REVERSE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter (A/B/C/D) from model output.

    Priority:
    1. Standalone letter at start or end of text
    2. "answer is X" pattern
    3. First standalone A/B/C/D found
    """
    text = text.strip()

    # 1. Check if text is just a single letter
    if text.upper() in REVERSE_MAP:
        return text.upper()

    # 2. Check last line for standalone letter
    last_line = text.strip().split("\n")[-1].strip()
    if last_line.upper() in REVERSE_MAP:
        return last_line.upper()

    # 3. "answer is X" pattern
    match = re.search(
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([A-Da-d])\b",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 4. First standalone letter A-D (word boundary)
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        letter = match.group(1).upper()
        if letter in REVERSE_MAP:
            return letter

    return None


def check_correct(prediction_letter: Optional[str], gold_int: int) -> bool:
    """Check if predicted letter matches gold answer index (0-3)."""
    if prediction_letter is None:
        return False
    gold_letter = LETTER_MAP.get(gold_int)
    if gold_letter is None:
        return False
    return prediction_letter == gold_letter


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute accuracy with routing and per-category breakdown."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))

    # Routing accuracy (did the Router pick the right category?)
    routing_correct = sum(1 for r in results if r.get("routing_correct", False))

    # Per-category breakdown
    per_category = {}
    for r in results:
        cat = r.get("expected_category", "unknown")
        if cat not in per_category:
            per_category[cat] = {"correct": 0, "total": 0, "routing_correct": 0}
        per_category[cat]["total"] += 1
        if r.get("correct", False):
            per_category[cat]["correct"] += 1
        if r.get("routing_correct", False):
            per_category[cat]["routing_correct"] += 1
    for cat in per_category:
        d = per_category[cat]
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        d["routing_accuracy"] = d["routing_correct"] / d["total"] if d["total"] > 0 else 0.0

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "routing_accuracy": routing_correct / total if total > 0 else 0.0,
        "routing_correct": routing_correct,
        "per_category": per_category,
    }


def print_routing_summary(results: List[Dict]) -> None:
    """Print routing accuracy and per-category breakdown."""
    acc = compute_accuracy(results)

    print(f"\n  Routing accuracy: {acc['routing_accuracy']:.1%} "
          f"({acc['routing_correct']}/{acc['total']})")

    per_category = acc.get("per_category", {})
    if per_category:
        print(f"\n  Per-Category Breakdown:")
        print(f"  {'Category':<15} {'Accuracy':>10} {'Routing':>10} {'Correct':>8} {'Total':>6}")
        print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 6}")
        for cat in sorted(per_category.keys()):
            d = per_category[cat]
            print(f"  {cat:<15} {d['accuracy']:>9.1%} {d['routing_accuracy']:>9.1%} "
                  f"{d['correct']:>8} {d['total']:>6}")
