"""GSM8K answer extraction and evaluation for fan-out benchmark.

Reuses the same extraction logic as the original GSM8K benchmark.
"""

from benchmarks.shared.evaluate_common import (
    check_correct,
    compute_accuracy,
    normalize_answer,
)
from benchmarks.gsm8k.evaluate import (
    extract_gold,
    extract_gsm8k_answer,
)

__all__ = [
    "extract_gold",
    "extract_gsm8k_answer",
    "check_correct",
    "compute_accuracy",
    "normalize_answer",
]
