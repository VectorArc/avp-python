"""Modal sanity: llama.cpp latent think → generate pipeline.

Tests the full pipeline: cb_eval captures hidden state during think(),
batch.embd injects it during generate(). Uses a small GGUF model
(Qwen2.5-1.5B Q4_K_M) for fast validation.

Usage:
    modal run benchmarks/modal_llamacpp_sanity.py
"""

import modal

app = modal.App("avp-llamacpp-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "huggingface-hub>=0.20",
        "gguf>=0.6",
    )
    .run_commands(
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


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def run_test():
    import os
    import time

    # Set LD_LIBRARY_PATH before any llama_cpp import
    import torch
    torch_lib = os.path.join(torch.__path__[0], "lib")
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    # Also set for ctypes to find it
    import ctypes
    try:
        ctypes.CDLL(os.path.join(torch_lib, "libcudart.so.12"))
    except OSError:
        print("Warning: could not preload libcudart.so.12")

    from huggingface_hub import hf_hub_download

    # Download a small GGUF model
    MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

    print("=" * 60)
    print(f"Downloading {MODEL_FILE}...")
    print("=" * 60)

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model path: {model_path}")

    results = {}

    # ==============================================================
    # Test 1: Basic generation (no latent, verify model works)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Basic generation (no latent)")
    print("=" * 60)

    from avp.connectors.llamacpp import LlamaCppConnector

    connector = LlamaCppConnector.from_pretrained(
        model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False,
    )

    t0 = time.monotonic()
    text = connector.generate(
        "What is 2 + 2? Answer with just the number.",
        max_tokens=20,
        temperature=0.0,
    )
    elapsed = time.monotonic() - t0
    print(f"  Output: {text!r}")
    print(f"  Time: {elapsed:.2f}s")
    results["basic"] = {"text": text[:100], "ok": len(text) > 0}

    # ==============================================================
    # Test 1b: GGUF weight extraction diagnostic
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 1b: GGUF embed weight extraction")
    print("=" * 60)
    try:
        from avp.connectors._llamacpp_compat import extract_gguf_embedding_weights
        emb_w = extract_gguf_embedding_weights(model_path)
        print(f"  embed_tokens shape: {emb_w.shape}")
        import numpy as np
        print(f"  embed_tokens norm (mean): {np.linalg.norm(emb_w, axis=1).mean():.3f}")
        results["gguf_extract"] = {"shape": list(emb_w.shape), "ok": True}
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["gguf_extract"] = {"ok": False, "error": str(e)}

    # ==============================================================
    # Test 2: Think (cb_eval captures hidden state)
    # ==============================================================
    # Test 1c: Low-level decode+sample WITHOUT cb_eval (diagnostic)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 1c: Low-level decode+greedy WITHOUT cb_eval")
    print("=" * 60)
    try:
        from llama_cpp import llama_cpp as lc
        model_ptr = connector._model._model.model
        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = 2048
        ctx_params.n_batch = 512
        # NO cb_eval — vanilla context
        diag_ctx = lc.llama_new_context_with_model(model_ptr, ctx_params)
        tokens = connector._model.tokenize(
            "Solve step by step: 24 * 17 + 3".encode("utf-8"), add_bos=True,
        )
        batch = lc.llama_batch_init(len(tokens), 0, 1)
        for i, tok in enumerate(tokens):
            batch.token[i] = tok
            batch.pos[i] = i
            batch.seq_id[i][0] = 0
            batch.n_seq_id[i] = 1
            batch.logits[i] = 1 if i == len(tokens) - 1 else 0
        batch.n_tokens = len(tokens)
        lc.llama_decode(diag_ctx, batch)
        lc.llama_batch_free(batch)

        # Greedy sample 50 tokens
        sampler_params = lc.llama_sampler_chain_default_params()
        sampler = lc.llama_sampler_chain_init(sampler_params)
        greedy_s = lc.llama_sampler_init_greedy()
        lc.llama_sampler_chain_add(sampler, greedy_s)

        gen_tokens = []
        eos = connector._model.token_eos()
        n_cur = len(tokens)
        next_batch = lc.llama_batch_init(1, 0, 1)
        for _ in range(50):
            tid = lc.llama_sampler_sample(sampler, diag_ctx, -1)
            lc.llama_sampler_accept(sampler, tid)
            if tid == eos:
                break
            gen_tokens.append(tid)
            next_batch.token[0] = tid
            next_batch.pos[0] = n_cur
            next_batch.seq_id[0][0] = 0
            next_batch.n_seq_id[0] = 1
            next_batch.logits[0] = 1
            next_batch.n_tokens = 1
            n_cur += 1
            lc.llama_decode(diag_ctx, next_batch)
        lc.llama_batch_free(next_batch)
        lc.llama_sampler_free(sampler)
        lc.llama_free(diag_ctx)

        diag_text = connector._model.detokenize(gen_tokens).decode("utf-8", errors="replace")
        print(f"  Low-level output: {diag_text[:200]!r}")
        results["diag_no_cb"] = {"text": diag_text[:200], "ok": len(diag_text) > 5}

        # Test: can we get embeddings without cb_eval?
        # Recreate context with embeddings=True
        ctx_params2 = lc.llama_context_default_params()
        ctx_params2.n_ctx = 2048
        ctx_params2.n_batch = 512
        ctx_params2.embeddings = True
        emb_ctx = lc.llama_new_context_with_model(model_ptr, ctx_params2)
        batch2 = lc.llama_batch_init(len(tokens), 0, 1)
        for i, tok in enumerate(tokens):
            batch2.token[i] = tok
            batch2.pos[i] = i
            batch2.seq_id[i][0] = 0
            batch2.n_seq_id[i] = 1
            batch2.logits[i] = 1 if i == len(tokens) - 1 else 0
        batch2.n_tokens = len(tokens)
        lc.llama_decode(emb_ctx, batch2)
        lc.llama_batch_free(batch2)

        # Try llama_get_embeddings_ith
        emb_ptr = lc.llama_get_embeddings_ith(emb_ctx, -1)
        if emb_ptr:
            import ctypes
            import numpy as np
            n_embd = connector._n_embd
            emb_arr = (ctypes.c_float * n_embd).from_address(ctypes.addressof(emb_ptr.contents))
            emb_np = np.array(emb_arr, dtype=np.float32, copy=True)
            emb_norm = float(np.linalg.norm(emb_np))
            print(f"  Embeddings via llama_get_embeddings_ith: norm={emb_norm:.3f}, shape=({n_embd},)")
            results["emb_api"] = {"norm": emb_norm, "ok": emb_norm > 0}
        else:
            print("  llama_get_embeddings_ith returned NULL")
            results["emb_api"] = {"ok": False}

        # Now test: greedy sampling AFTER embeddings=True decode
        sampler2 = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
        lc.llama_sampler_chain_add(sampler2, lc.llama_sampler_init_greedy())
        gen2 = []
        n_cur2 = len(tokens)
        nb2 = lc.llama_batch_init(1, 0, 1)
        for _ in range(50):
            tid = lc.llama_sampler_sample(sampler2, emb_ctx, -1)
            lc.llama_sampler_accept(sampler2, tid)
            if tid == eos:
                break
            gen2.append(tid)
            nb2.token[0] = tid
            nb2.pos[0] = n_cur2
            nb2.seq_id[0][0] = 0
            nb2.n_seq_id[0] = 1
            nb2.logits[0] = 1
            nb2.n_tokens = 1
            n_cur2 += 1
            lc.llama_decode(emb_ctx, nb2)
        lc.llama_batch_free(nb2)
        lc.llama_sampler_free(sampler2)
        emb_gen_text = connector._model.detokenize(gen2).decode("utf-8", errors="replace")
        print(f"  Sampling after embeddings=True: {emb_gen_text[:150]!r}")
        results["emb_gen"] = {"text": emb_gen_text[:150], "ok": len(emb_gen_text) > 5}

        lc.llama_free(emb_ctx)
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["diag_no_cb"] = {"ok": False, "error": str(e)}

    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Think (hidden state extraction via cb_eval)")
    print("=" * 60)

    t0 = time.monotonic()
    context = connector.think(
        "Analyze this math problem carefully: 24 * 17 + 3",
        steps=10,
    )
    elapsed = time.monotonic() - t0

    if context is not None:
        hidden = getattr(context, "last_hidden_state", None)
        if hidden is not None:
            import numpy as _np
            h_np = hidden if isinstance(hidden, _np.ndarray) else hidden.detach().cpu().float().numpy()
            h_norm = float(_np.linalg.norm(h_np))
            print(f"  Hidden state shape: {list(h_np.shape)}")
            print(f"  Hidden state norm: {h_norm:.3f}")
            print(f"  Time: {elapsed:.2f}s")
            results["think"] = {
                "shape": list(h_np.shape),
                "norm": h_norm,
                "ok": True,
            }
        else:
            print("  WARNING: context has no hidden_state")
            results["think"] = {"ok": False, "error": "no hidden_state"}
    else:
        print("  WARNING: think() returned None")
        results["think"] = {"ok": False, "error": "returned None"}

    # ==============================================================
    # Test 3: Generate with latent context (batch.embd injection)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Generate with latent context (embedding injection)")
    print("=" * 60)

    if context is not None and getattr(context, "last_hidden_state", None) is not None:
        t0 = time.monotonic()
        answer = connector.generate(
            "Solve step by step: 24 * 17 + 3",
            context=context,
            max_tokens=100,
            temperature=0.0,
        )
        elapsed = time.monotonic() - t0
        print(f"  Answer: {answer[:200]!r}")
        print(f"  Time: {elapsed:.2f}s")
        has_411 = "411" in answer
        print(f"  Contains 411: {has_411}")
        results["generate_latent"] = {
            "text": answer[:200],
            "has_411": has_411,
            "ok": len(answer) > 0,
        }
    else:
        print("  SKIP: No context from think()")
        results["generate_latent"] = {"ok": False, "error": "no context"}

    # ==============================================================
    # Test 4: Text-only baseline (same problem, no latent)
    # ==============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Text baseline (no latent)")
    print("=" * 60)

    t0 = time.monotonic()
    baseline = connector.generate(
        "Solve step by step: 24 * 17 + 3",
        max_tokens=100,
        temperature=0.0,
    )
    elapsed = time.monotonic() - t0
    print(f"  Answer: {baseline[:200]!r}")
    print(f"  Time: {elapsed:.2f}s")
    has_411 = "411" in baseline
    print(f"  Contains 411: {has_411}")
    results["baseline"] = {
        "text": baseline[:200],
        "has_411": has_411,
        "ok": len(baseline) > 0,
    }

    # ==============================================================
    # Summary
    # ==============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, data in results.items():
        status = "OK" if data.get("ok") else "FAILED"
        print(f"  {name}: {status}")

    return results


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
