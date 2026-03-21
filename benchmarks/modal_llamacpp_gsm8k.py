"""Modal benchmark: llama.cpp GSM8K parity test.

Tests the full latent think → generate pipeline on GGUF models.
Attempts GPU inference; falls back to CPU if CUDA linking fails.

Usage:
    modal run benchmarks/modal_llamacpp_gsm8k.py --n 50
"""

import modal

app = modal.App("avp-llamacpp-gsm8k")

# GPU attempt: install torch first (provides libcudart.so), set LD_LIBRARY_PATH,
# then install CUDA wheel for llama-cpp-python.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "huggingface-hub>=0.20",
        "gguf>=0.6",
        "datasets>=2.14",
    )
    .run_commands(
        # Install CUDA wheel, using torch's bundled CUDA runtime for linking
        'TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0])")/lib && '
        'LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH" '
        'pip install llama-cpp-python>=0.3 '
        '--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 '
        '--no-cache-dir',
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/torch/lib"})
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=7200)
def run_benchmark(n: int = 50):
    import os
    import re
    import time

    # Preload CUDA runtime from torch before llama_cpp imports
    import torch
    import ctypes
    torch_lib = os.path.join(torch.__path__[0], "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    try:
        ctypes.CDLL(os.path.join(torch_lib, "libcudart.so.12"))
    except OSError:
        pass

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    print("=" * 60)
    print(f"Downloading {MODEL_FILE}...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model: {model_path}")

    # Detect GPU support
    try:
        import llama_cpp
        gpu_layers = -1  # All layers on GPU
        print("llama-cpp-python with CUDA loaded successfully")
    except RuntimeError as e:
        if "libcudart" in str(e) or "libcuda" in str(e):
            print(f"CUDA linking failed ({e}), falling back to CPU")
            # Reinstall CPU-only version
            import subprocess
            subprocess.run(
                ["pip", "install", "llama-cpp-python>=0.3", "--force-reinstall",
                 "--no-cache-dir"], check=True,
            )
            import importlib
            importlib.invalidate_caches()
            gpu_layers = 0
        else:
            raise

    from avp.connectors.llamacpp import LlamaCppConnector

    connector = LlamaCppConnector.from_pretrained(
        model_path, n_ctx=2048, n_gpu_layers=gpu_layers, verbose=False,
    )

    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    num_pat = re.compile(r"-?\d[\d,]*")

    questions = [ds[i]["question"] for i in range(n)]
    gold = []
    for i in range(n):
        m = re.search(r"####\s*(-?\d[\d,]*)", ds[i]["answer"])
        gold.append(m.group(1).replace(",", "") if m else "")

    def extract_answer(text):
        nums = num_pat.findall(text)
        return nums[-1].replace(",", "") if nums else ""

    # ==============================================================
    # Latent: think → generate
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"LATENT: think(10 steps) + generate (n={n})")
    print(f"{'='*60}")

    latent_correct = 0
    t0 = time.monotonic()

    for i, q in enumerate(questions):
        # Think
        context = connector.think(
            f"Analyze this math problem carefully: {q}",
            steps=10,
        )

        # Generate
        if context is not None:
            answer = connector.generate(
                f"Solve step by step: {q}",
                context=context,
                max_tokens=1024,
                temperature=0.0,
            )
        else:
            answer = connector.generate(
                f"Solve step by step: {q}",
                max_tokens=1024,
                temperature=0.0,
            )

        pred = extract_answer(answer)
        if pred == gold[i]:
            latent_correct += 1
        else:
            # Log first 3 wrong answers for debugging
            if latent_correct == 0 and i < 3:
                print(f"    [debug] q{i}: gold={gold[i]}, pred={pred!r}")
                print(f"    [debug] answer: {answer[:200]!r}")

        if (i + 1) % 10 == 0 or i == n - 1:
            pct = latent_correct / (i + 1) * 100
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{n}] acc={pct:.1f}%, elapsed={elapsed:.0f}s")

    latent_elapsed = time.monotonic() - t0
    latent_pct = latent_correct / n * 100

    # ==============================================================
    # Text baseline
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"TEXT BASELINE (n={n})")
    print(f"{'='*60}")

    text_correct = 0
    t0 = time.monotonic()

    for i, q in enumerate(questions):
        answer = connector.generate(
            f"Solve step by step: {q}",
            max_tokens=512,
            temperature=0.0,
        )
        pred = extract_answer(answer)
        if pred == gold[i]:
            text_correct += 1

        if (i + 1) % 10 == 0 or i == n - 1:
            pct = text_correct / (i + 1) * 100
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{n}] acc={pct:.1f}%, elapsed={elapsed:.0f}s")

    text_elapsed = time.monotonic() - t0
    text_pct = text_correct / n * 100

    # ==============================================================
    # Results
    # ==============================================================
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Model:   {MODEL_FILE}")
    print(f"  n:       {n}")
    print(f"  GPU:     {'yes' if gpu_layers != 0 else 'no (CPU)'}")
    print(f"  Latent:  {latent_correct}/{n} = {latent_pct:.1f}%")
    print(f"  Text:    {text_correct}/{n} = {text_pct:.1f}%")
    print(f"  Delta:   {latent_pct - text_pct:+.1f}pp")
    print(f"  Latent time: {latent_elapsed:.0f}s ({latent_elapsed/n:.1f}s/problem)")
    print(f"  Text time:   {text_elapsed:.0f}s ({text_elapsed/n:.1f}s/problem)")

    return {
        "latent_correct": latent_correct,
        "latent_pct": latent_pct,
        "text_correct": text_correct,
        "text_pct": text_pct,
        "delta_pp": latent_pct - text_pct,
        "latent_time": latent_elapsed,
        "text_time": text_elapsed,
        "n": n,
        "gpu": gpu_layers != 0,
    }


@app.local_entrypoint()
def main(n: int = 50):
    import json
    result = run_benchmark.remote(n=n)
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
