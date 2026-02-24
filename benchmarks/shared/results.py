"""Result saving utilities for benchmarks."""

import json
import os
from datetime import datetime
from typing import Any, Dict


def save_results(
    data: Dict[str, Any],
    output_dir: str,
    benchmark_name: str,
    model_name: str,
    mode: str,
    n_samples: int,
) -> str:
    """Save benchmark results to a timestamped JSON file.

    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_short = model_name.split("/")[-1].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{model_short}_{mode}_n{n_samples}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Also write a latest copy for easy access
    latest_path = os.path.join(output_dir, f"{benchmark_name}_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Latest copy: {latest_path}")

    return output_path
