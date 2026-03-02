"""GPU memory tracking for benchmarks."""

from contextlib import contextmanager

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
