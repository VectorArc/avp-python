"""Modal benchmark: cross-model rosetta on GGUF models via LlamaCppConnector.

Downloads two GGUF models from HuggingFace Hub and runs cross-model
rosetta (Qwen 7B → Llama 3B and reverse) on GSM8K problems. Validates
the full pipeline: think() on source → generate(context, source, cross_model)
on target.

Usage:
    modal run benchmarks/modal_cross_model_gguf.py
    modal run benchmarks/modal_cross_model_gguf.py --n 20
"""

import modal

app = modal.App("avp-cross-model-gguf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "huggingface-hub>=0.20",
        "gguf>=0.6",
        "datasets>=2.14",
        "jinja2>=3.0",
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


MODELS = {
    "qwen7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q3_k_m.gguf",
        "label": "Qwen2.5-7B Q3_K_M",
    },
    "llama3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "label": "Llama-3.2-3B Q4_K_M",
    },
}


@app.function(image=image, gpu="A100-40GB", timeout=7200)
def run_benchmark(n: int = 10):
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
        print("Warning: could not preload libcudart.so.12")

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    # ==============================================================
    # Download models
    # ==============================================================
    print("=" * 60)
    print("DOWNLOADING MODELS")
    print("=" * 60)

    model_paths = {}
    for key, info in MODELS.items():
        print(f"  Downloading {info['label']}...")
        path = hf_hub_download(repo_id=info["repo"], filename=info["file"])
        model_paths[key] = path
        print(f"  -> {path}")

    # ==============================================================
    # Load connectors (one at a time to manage VRAM)
    # ==============================================================
    from avp.connectors.llamacpp import LlamaCppConnector

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

    results = {}

    # ==============================================================
    # Test 1: Same-model baselines (Qwen 7B)
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"TEST 1: Qwen 7B same-model latent + text (n={n})")
    print(f"{'='*60}")

    qwen = LlamaCppConnector.from_pretrained(
        model_paths["qwen7b"], n_ctx=4096, n_gpu_layers=-1, verbose=False,
    )

    # Latent
    latent_correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        prompt = f"Solve step by step: {q}"
        context = qwen.think(prompt, steps=10)
        if context is not None:
            answer = qwen.generate("", context=context, max_tokens=2048, temperature=0.0)
        else:
            answer = qwen.generate(prompt, max_tokens=2048, temperature=0.0)
        if extract_answer(answer) == gold[i]:
            latent_correct += 1
        if (i + 1) % 5 == 0:
            print(f"  latent [{i+1}/{n}] acc={latent_correct/(i+1)*100:.0f}%")
    latent_time = time.monotonic() - t0

    # Text
    text_correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        answer = qwen.generate(f"Solve step by step: {q}", max_tokens=2048, temperature=0.0)
        if extract_answer(answer) == gold[i]:
            text_correct += 1
        if (i + 1) % 5 == 0:
            print(f"  text [{i+1}/{n}] acc={text_correct/(i+1)*100:.0f}%")
    text_time = time.monotonic() - t0

    results["qwen7b_same"] = {
        "latent": latent_correct, "text": text_correct, "n": n,
        "latent_pct": latent_correct / n * 100,
        "text_pct": text_correct / n * 100,
    }
    print(f"  Qwen 7B: latent={latent_correct}/{n}, text={text_correct}/{n}")

    # ==============================================================
    # Test 2: Cross-model Qwen 7B → Llama 3B (forward rosetta)
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"TEST 2: Cross-model Qwen 7B → Llama 3B (n={n})")
    print(f"{'='*60}")

    llama = LlamaCppConnector.from_pretrained(
        model_paths["llama3b"], n_ctx=4096, n_gpu_layers=-1, verbose=False,
    )

    rosetta_fwd_correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        prompt = f"Solve step by step: {q}"
        context = qwen.think(prompt, steps=10)
        if context is not None:
            answer = llama.generate(
                prompt, context=context, source=qwen, cross_model=True,
                max_tokens=2048, temperature=0.0,
            )
        else:
            answer = llama.generate(prompt, max_tokens=2048, temperature=0.0)
        if extract_answer(answer) == gold[i]:
            rosetta_fwd_correct += 1
        if (i + 1) % 5 == 0:
            print(f"  fwd [{i+1}/{n}] acc={rosetta_fwd_correct/(i+1)*100:.0f}%")
    rosetta_fwd_time = time.monotonic() - t0

    results["qwen7b_to_llama3b"] = {
        "correct": rosetta_fwd_correct, "n": n,
        "pct": rosetta_fwd_correct / n * 100,
    }
    print(f"  Qwen 7B → Llama 3B: {rosetta_fwd_correct}/{n} = {rosetta_fwd_correct/n*100:.0f}%")

    # ==============================================================
    # Test 3: Cross-model Llama 3B → Qwen 7B (reverse rosetta)
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"TEST 3: Cross-model Llama 3B → Qwen 7B (n={n})")
    print(f"{'='*60}")

    rosetta_rev_correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        prompt = f"Solve step by step: {q}"
        context = llama.think(prompt, steps=10)
        if context is not None:
            answer = qwen.generate(
                prompt, context=context, source=llama, cross_model=True,
                max_tokens=2048, temperature=0.0,
            )
        else:
            answer = qwen.generate(prompt, max_tokens=2048, temperature=0.0)
        if extract_answer(answer) == gold[i]:
            rosetta_rev_correct += 1
        if (i + 1) % 5 == 0:
            print(f"  rev [{i+1}/{n}] acc={rosetta_rev_correct/(i+1)*100:.0f}%")
    rosetta_rev_time = time.monotonic() - t0

    results["llama3b_to_qwen7b"] = {
        "correct": rosetta_rev_correct, "n": n,
        "pct": rosetta_rev_correct / n * 100,
    }
    print(f"  Llama 3B → Qwen 7B: {rosetta_rev_correct}/{n} = {rosetta_rev_correct/n*100:.0f}%")

    # ==============================================================
    # Test 4: Llama 3B same-model baseline
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"TEST 4: Llama 3B same-model text baseline (n={n})")
    print(f"{'='*60}")

    llama_text_correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(questions):
        answer = llama.generate(f"Solve step by step: {q}", max_tokens=2048, temperature=0.0)
        if extract_answer(answer) == gold[i]:
            llama_text_correct += 1
        if (i + 1) % 5 == 0:
            print(f"  text [{i+1}/{n}] acc={llama_text_correct/(i+1)*100:.0f}%")
    llama_text_time = time.monotonic() - t0

    results["llama3b_text"] = {
        "correct": llama_text_correct, "n": n,
        "pct": llama_text_correct / n * 100,
    }

    # ==============================================================
    # Summary
    # ==============================================================
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Qwen 7B same-model:  latent={results['qwen7b_same']['latent']}/{n} "
          f"({results['qwen7b_same']['latent_pct']:.0f}%), "
          f"text={results['qwen7b_same']['text']}/{n} "
          f"({results['qwen7b_same']['text_pct']:.0f}%)")
    print(f"  Llama 3B text:       {results['llama3b_text']['correct']}/{n} "
          f"({results['llama3b_text']['pct']:.0f}%)")
    print(f"  Qwen 7B → Llama 3B: {results['qwen7b_to_llama3b']['correct']}/{n} "
          f"({results['qwen7b_to_llama3b']['pct']:.0f}%)")
    print(f"  Llama 3B → Qwen 7B: {results['llama3b_to_qwen7b']['correct']}/{n} "
          f"({results['llama3b_to_qwen7b']['pct']:.0f}%)")

    return results


@app.local_entrypoint()
def main(n: int = 10):
    import json
    result = run_benchmark.remote(n=n)
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
