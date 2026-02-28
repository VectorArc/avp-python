"""HotpotQA evaluation: Exact Match and token-level F1.

Standard HotpotQA evaluation after answer normalization:
lowercase, remove articles/punctuation/extra whitespace.
"""

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional


def normalize_hotpot_answer(text: str) -> str:
    """Normalize answer for HotpotQA evaluation.

    Follows the official HotpotQA evaluation script:
    lowercase, remove articles, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_answer(text: str) -> str:
    """Extract the answer from model output.

    Looks for patterns like "Answer: X", "the answer is X", or takes
    the first sentence/line as the answer.
    """
    # Try "Answer: X" pattern
    match = re.search(r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try "the answer is X" pattern
    match = re.search(r"the answer is\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: take the last non-empty line (often the final answer)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return text.strip()


def exact_match(prediction: str, gold: str) -> bool:
    """Check exact match after normalization."""
    return normalize_hotpot_answer(prediction) == normalize_hotpot_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold."""
    pred_tokens = normalize_hotpot_answer(prediction).split()
    gold_tokens = normalize_hotpot_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute HotpotQA metrics from a list of results, with Wilson score 95% CI."""
    total = len(results)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "correct": 0, "total": 0,
                "accuracy": 0.0, "ci_95_lo": 0.0, "ci_95_hi": 0.0}

    em_count = sum(1 for r in results if r.get("exact_match", False))
    f1_scores = [r.get("f1", 0.0) for r in results]
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    acc = em_count / total

    # Wilson score interval
    z = 1.96
    denom = 1 + z**2 / total
    center = (acc + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((acc * (1 - acc) + z**2 / (4 * total)) / total) / denom

    return {
        "exact_match": acc,
        "f1": mean_f1,
        "correct": em_count,
        "total": total,
        "accuracy": acc,  # For compatibility with shared print_summary
        "ci_95_lo": max(0.0, center - margin),
        "ci_95_hi": min(1.0, center + margin),
    }


def check_correct(prediction: str, gold: str) -> bool:
    """Check if prediction is correct (exact match)."""
    return exact_match(prediction, gold)
