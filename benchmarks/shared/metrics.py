"""GPU memory tracking and result dict building for benchmarks."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch


@contextmanager
def gpu_memory_tracker(device: str):
    """Context manager that tracks peak GPU memory delta.

    Usage:
        with gpu_memory_tracker("cuda") as tracker:
            # ... do work ...
        peak_mb = tracker["peak_memory_mb"]  # None if not CUDA
    """
    result = {"peak_memory_mb": None}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()
    try:
        yield result
    finally:
        if device == "cuda":
            result["peak_memory_mb"] = (
                torch.cuda.max_memory_allocated() - mem_before
            ) / (1024 * 1024)


def build_result_dict(
    *,
    question: str,
    gold: Optional[str],
    prediction: Optional[str],
    raw_output: str,
    correct: bool,
    wall_time: float,
    mode: str,
    peak_memory_mb: Optional[float] = None,
    **extra,
) -> Dict[str, Any]:
    """Build a standard result dict with common fields."""
    result = {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "raw_output": raw_output,
        "correct": correct,
        "wall_time": wall_time,
        "peak_memory_mb": peak_memory_mb,
        "mode": mode,
    }
    result.update(extra)
    return result
