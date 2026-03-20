"""Modal benchmark: vLLM parity with HuggingFace GSM8K results.

Validates that the vLLM integration reproduces the HuggingFace accuracy
baselines. Supports same-model latent, same-family rosetta, and
cross-family rosetta.

Usage:
    # Quick smoke test (n=5, same-model)
    modal run benchmarks/modal_vllm_gsm8k_parity.py

    # Same-model n=200 (matches HF 90.5% vs 87.0%)
    modal run benchmarks/modal_vllm_gsm8k_parity.py --n 200

    # Cross-model same-family (matches HF 58.5%)
    modal run benchmarks/modal_vllm_gsm8k_parity.py --mode cross-family --n 200

    # Cross-family (matches HF 77.0%, requires HF token for Llama)
    modal run benchmarks/modal_vllm_gsm8k_parity.py --mode cross-family-llama --n 200
"""

import modal

app = modal.App("avp-vllm-gsm8k-parity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm>=0.17.0", "torch>=2.0", "transformers>=4.36",
        "datasets>=2.14", "safetensors",
    )
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)

# HF baselines from CLAUDE.md / BENCHMARKS.md
HF_BASELINES = {
    "same-model": {"latent": 90.5, "text": 87.0, "n": 200},
    "cross-family": {"latent": 58.5, "text": 58.5, "n": 200},  # Qwen 7B → Qwen 1.5B
    "cross-family-llama": {"latent": 77.0, "text": 77.0, "n": 200},  # Qwen 7B → Llama 3B
}


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=18000,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_benchmark(mode: str = "same-model", n: int = 5):
    import gc
    import os
    import re
    import tempfile
    import time

    import torch
    import vllm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    # Model selection by mode
    if mode == "same-model":
        SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        TGT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        LATENT_STEPS = 10
    elif mode == "cross-family":
        SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        TGT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
        LATENT_STEPS = 10
    elif mode == "cross-family-llama":
        SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
        TGT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
        LATENT_STEPS = 10
    else:
        raise ValueError(f"Unknown mode: {mode}")

    is_cross_model = SRC_MODEL != TGT_MODEL
    src_tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    tgt_tokenizer = AutoTokenizer.from_pretrained(TGT_MODEL)

    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    num_pat = re.compile(r"-?\d[\d,]*")

    questions = [ds[i]["question"] for i in range(n)]
    gold = []
    for i in range(n):
        m = re.search(r"####\s*(-?\d[\d,]*)", ds[i]["answer"])
        gold.append(m.group(1).replace(",", "") if m else "")

    from avp.connectors.vllm_kv_connector import (
        compute_request_hash,
        generate_with_rosetta,
        make_sampling_params,
        prepare_latent_prompt,
    )

    def extract_answer(text):
        nums = num_pat.findall(text)
        return nums[-1].replace(",", "") if nums else ""

    results = {
        "mode": mode, "n": n,
        "src_model": SRC_MODEL, "tgt_model": TGT_MODEL,
    }

    with tempfile.TemporaryDirectory() as store_dir:
        # ==============================================================
        # Agent A: Think (latent steps)
        # ==============================================================
        print(f"\n{'='*60}")
        print(f"AGENT A: {SRC_MODEL} (think, {LATENT_STEPS} latent steps)")
        print(f"Mode: {mode}, n={n}")
        print(f"{'='*60}")

        os.environ["AVP_LATENT_STEPS"] = str(LATENT_STEPS)
        if is_cross_model:
            os.environ["AVP_TARGET_MODEL"] = TGT_MODEL

        ktc_a = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": LATENT_STEPS,
                "avp_store_dir": store_dir,
                **({"avp_target_model": TGT_MODEL} if is_cross_model else {}),
            },
        )

        engine_a = vllm.LLM(
            model=SRC_MODEL,
            enforce_eager=True,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            kv_transfer_config=ktc_a,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        # Think on all problems (researcher prompt, different from solver)
        store_keys = []
        t0_think = time.monotonic()

        for i, q in enumerate(questions):
            store_key = f"gsm8k-{i}"
            store_keys.append(store_key)

            researcher_prompt = f"Analyze this math problem carefully: {q}"
            ids = src_tokenizer.apply_chat_template(
                [{"role": "user", "content": researcher_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            padded = prepare_latent_prompt(list(ids), LATENT_STEPS)

            think_params = make_sampling_params(store_key, max_tokens=1, temperature=0.0)
            engine_a.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded)], think_params,
            )
            if (i + 1) % 50 == 0 or i == n - 1:
                print(f"  Think: {i+1}/{n}")

        think_elapsed = time.monotonic() - t0_think
        print(f"  Think complete: {think_elapsed:.1f}s ({think_elapsed/n:.2f}s/problem)")

        # Free Agent A
        del engine_a
        gc.collect()
        torch.cuda.empty_cache()
        os.environ.pop("AVP_TARGET_MODEL", None)
        os.environ.pop("AVP_LATENT_STEPS", None)

        # ==============================================================
        # Agent B: Generate (latent-primed)
        # ==============================================================
        print(f"\n{'='*60}")
        print(f"AGENT B: {TGT_MODEL} (generate)")
        print(f"{'='*60}")

        if is_cross_model:
            # Cross-model: use generate_with_rosetta
            engine_b = vllm.LLM(
                model=TGT_MODEL,
                enforce_eager=True,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                max_num_seqs=1,
            )

            gen_params = vllm.SamplingParams(
                max_tokens=2048, temperature=0.0,
            )

            latent_correct = 0
            t0_gen = time.monotonic()
            for i, q in enumerate(questions):
                solver_prompt = f"Solve step by step: {q}"
                solver_ids = tgt_tokenizer.apply_chat_template(
                    [{"role": "user", "content": solver_prompt}],
                    tokenize=True, add_generation_prompt=True,
                )
                outputs = generate_with_rosetta(
                    engine=engine_b,
                    prompt_token_ids=list(solver_ids),
                    store_dir=store_dir,
                    store_key=store_keys[i],
                    sampling_params=gen_params,
                )
                pred = extract_answer(outputs[0].outputs[0].text)
                if pred == gold[i]:
                    latent_correct += 1
                if (i + 1) % 50 == 0 or i == n - 1:
                    pct = latent_correct / (i + 1) * 100
                    print(f"  Latent: {i+1}/{n}, running acc: {pct:.1f}%")

            gen_elapsed = time.monotonic() - t0_gen
        else:
            # Same-model: use KV connector path (Agent B loads Agent A's KV)
            ktc_b = KVTransferConfig(
                kv_connector="AVPKVConnectorV1Dynamic",
                kv_connector_module_path="avp.connectors.vllm_kv_connector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "avp_latent_steps": 0,
                    "avp_store_dir": store_dir,
                },
            )
            engine_b = vllm.LLM(
                model=TGT_MODEL,
                enforce_eager=True,
                max_model_len=2048,
                gpu_memory_utilization=0.85,
                kv_transfer_config=ktc_b,
                max_num_seqs=1,
                enable_prefix_caching=False,  # Avoid conflict with KV connector
            )

            gen_params = vllm.SamplingParams(
                max_tokens=2048, temperature=0.0,
            )

            latent_correct = 0
            t0_gen = time.monotonic()
            for i, q in enumerate(questions):
                # Same-model with explicit store key: Agent B uses a
                # different solver prompt and the KV connector matches
                # via kv_transfer_params, not prompt hash.
                solver_prompt = f"Solve step by step: {q}"
                solver_ids = tgt_tokenizer.apply_chat_template(
                    [{"role": "user", "content": solver_prompt}],
                    tokenize=True, add_generation_prompt=True,
                )
                padded = prepare_latent_prompt(list(solver_ids), LATENT_STEPS)
                gen_params_keyed = make_sampling_params(
                    store_keys[i], max_tokens=2048, temperature=0.0,
                )
                outputs = engine_b.generate(
                    [vllm.TokensPrompt(prompt_token_ids=padded)],
                    gen_params_keyed,
                )
                pred = extract_answer(outputs[0].outputs[0].text)
                if pred == gold[i]:
                    latent_correct += 1
                if (i + 1) % 50 == 0 or i == n - 1:
                    pct = latent_correct / (i + 1) * 100
                    print(f"  Latent: {i+1}/{n}, running acc: {pct:.1f}%")

            gen_elapsed = time.monotonic() - t0_gen

        latent_pct = latent_correct / n * 100
        print(f"  Latent: {latent_correct}/{n} = {latent_pct:.1f}%")
        print(f"  Generation: {gen_elapsed:.1f}s ({gen_elapsed/n:.2f}s/problem)")

        # ==============================================================
        # Text baseline (no latent, no KV transfer)
        # ==============================================================
        print(f"\n{'='*60}")
        print(f"TEXT BASELINE: {TGT_MODEL}")
        print(f"{'='*60}")

        # Reuse engine_b (no KV connector for cross-model, or steps=0 for same-model)
        text_correct = 0
        t0_text = time.monotonic()
        for i, q in enumerate(questions):
            prompt = f"Solve step by step: {q}"
            ids = tgt_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            outputs = engine_b.generate(
                [vllm.TokensPrompt(prompt_token_ids=list(ids))],
                gen_params,
            )
            pred = extract_answer(outputs[0].outputs[0].text)
            if pred == gold[i]:
                text_correct += 1
            if (i + 1) % 50 == 0 or i == n - 1:
                pct = text_correct / (i + 1) * 100
                print(f"  Text: {i+1}/{n}, running acc: {pct:.1f}%")

        text_elapsed = time.monotonic() - t0_text
        text_pct = text_correct / n * 100

        del engine_b

    # ==============================================================
    # Summary
    # ==============================================================
    baseline = HF_BASELINES.get(mode, {})
    hf_latent = baseline.get("latent", "N/A")
    hf_text = baseline.get("text", "N/A")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Mode:    {mode}")
    print(f"  n:       {n}")
    print(f"  Models:  {SRC_MODEL} -> {TGT_MODEL}")
    print(f"  Latent:  {latent_correct}/{n} = {latent_pct:.1f}% (HF: {hf_latent}%)")
    print(f"  Text:    {text_correct}/{n} = {text_pct:.1f}% (HF: {hf_text}%)")
    print(f"  Delta:   {latent_pct - text_pct:+.1f}pp")
    print(f"  Think:   {think_elapsed:.1f}s, Gen: {gen_elapsed:.1f}s, Text: {text_elapsed:.1f}s")

    results.update({
        "latent_correct": latent_correct,
        "latent_pct": latent_pct,
        "text_correct": text_correct,
        "text_pct": text_pct,
        "delta_pp": latent_pct - text_pct,
        "hf_latent": hf_latent,
        "hf_text": hf_text,
        "think_time": think_elapsed,
        "gen_time": gen_elapsed,
        "text_time": text_elapsed,
    })

    return results


@app.local_entrypoint()
def main(mode: str = "same-model", n: int = 5):
    import json
    result = run_benchmark.remote(mode=mode, n=n)
    print("\n\nFinal:", json.dumps(result, indent=2, default=str))
