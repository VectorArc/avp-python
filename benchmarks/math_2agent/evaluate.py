"""MATH-500 answer extraction and evaluation for 2-agent benchmark.

Reuses the same extraction logic as the compiler (3-agent) MATH benchmark.
"""

from benchmarks.compiler.evaluate import (
    check_correct,
    compute_accuracy,
    extract_answer,
    extract_boxed_answer,
    normalize_math_answer,
    print_level_summary,
)


def extract_gold(sample_answer: str) -> str:
    """Extract gold answer from MATH-500 sample.

    MATH-500 'answer' field contains the final answer directly
    (e.g. "\\frac{3}{4}", "120", "2^{10}"), unlike GSM8K which uses "#### <number>".
    """
    return sample_answer


__all__ = [
    "extract_gold",
    "extract_answer",
    "extract_boxed_answer",
    "normalize_math_answer",
    "check_correct",
    "compute_accuracy",
    "print_level_summary",
]
