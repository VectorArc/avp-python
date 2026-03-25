"""Modal smoke test: GSM8K 2-agent n=5 all modes on Qwen 7B.

Quick validation that the full benchmark pipeline works after API changes.

Usage:
    modal run benchmarks/modal_smoke_gsm8k.py
"""

import importlib
import os
import sys

import modal

app = modal.App("avp-smoke-gsm8k")

results_volume = modal.Volume.from_name("avp-results", create_if_missing=True)
model_cache = modal.Volume.from_name("avp-model-cache", create_if_missing=True)
adapter_volume = modal.Volume.from_name("avp-adapters", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "datasets",
        "transformers>=4.45",
        "torch>=2.4",
        "accelerate",
        "numpy",
        "pyyaml",
        "sentencepiece",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/vectorarc/avp-python.git /root/avp-python",
        "pip install /root/avp-python",
        "echo 'avp-cache-bust: 2026-03-24-smoke-v1'",
    )
)

RESULTS_DIR = "/results"
MODEL_CACHE_DIR = "/model-cache"
ADAPTER_DIR = "/adapters"
AVP_PYTHON_DIR = "/root/avp-python"


@app.function(
    image=base_image,
    gpu="A100",
    volumes={RESULTS_DIR: results_volume, MODEL_CACHE_DIR: model_cache, ADAPTER_DIR: adapter_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_smoke():
    """Run GSM8K 2-agent n=5 all modes."""
    os.environ["HF_HOME"] = os.path.join(MODEL_CACHE_DIR, "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_CACHE_DIR, "huggingface")
    os.environ["AVP_CACHE_DIR"] = ADAPTER_DIR
    sys.path.insert(0, AVP_PYTHON_DIR)

    # Verify SDK imports cleanly
    import avp
    print(f"AVP version: {avp.__version__}")
    print(f"ThinkResult: {avp.ThinkResult}")
    print(f"GenerateResult: {avp.GenerateResult}")
    print(f"ConfigurationError: {avp.ConfigurationError}")
    print(f"ProjectionError: {avp.ProjectionError}")

    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "mode": "all",
        "max_samples": 5,
        "seed": 42,
        "latent_steps": 20,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "verbose": True,
        "output_dir": RESULTS_DIR,
    }

    module = importlib.import_module("benchmarks.gsm8k_2agent.run_gsm8k_2agent")
    result = module.run_benchmark(config)

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS (GSM8K 2-agent, n=5, Qwen 7B)")
    print("=" * 60)

    for mode_name in ["direct", "text", "latent"]:
        if mode_name not in result:
            continue
        summary = result[mode_name].get("summary", {})
        acc = summary.get("accuracy", 0)
        correct = summary.get("correct", 0)
        total = summary.get("total", 0)
        print(f"  {mode_name}: {acc:.0%} ({correct}/{total})")

    results_volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    result = run_smoke.remote()
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
