"""Modal test: Cross-model rosetta via vLLM (single GPU, sequential).

Agent A (Qwen 7B) thinks + projects → save to store → delete engine.
Agent B (Llama 3B) loads projected embedding → generates.

Usage:
    modal run benchmarks/modal_vllm_cross_model_1gpu.py
"""

import modal

app = modal.App("avp-vllm-cross-model-1gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0", "torch>=2.0", "transformers>=4.36", "safetensors")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=3600)
def run_test():
    import gc
    import os
    import tempfile
    import time
    from pathlib import Path

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    TGT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # ungated, same-family (vocab-mediated)

    src_tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    tgt_tokenizer = AutoTokenizer.from_pretrained(TGT_MODEL)

    results = {"status": "started"}

    problems = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?",
    ]

    with tempfile.TemporaryDirectory() as store_dir:
        # ==============================================================
        # AGENT A: Qwen 7B — think + rosetta project to target space
        # ==============================================================
        print("\n" + "=" * 60)
        print(f"AGENT A: {SRC_MODEL}")
        print("=" * 60)

        # Set env vars BEFORE creating the engine — vLLM loads the model
        # before the KV connector, so the model plugin reads env vars
        # before the connector has a chance to set them.
        os.environ["AVP_LATENT_STEPS"] = "10"
        os.environ["AVP_TARGET_MODEL"] = TGT_MODEL

        ktc_a = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": 10,
                "avp_store_dir": store_dir,
                "avp_target_model": TGT_MODEL,
            },
        )

        engine_a = vllm.LLM(
            model=SRC_MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.7,
            kv_transfer_config=ktc_a,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        from avp.connectors.vllm_kv_connector import (
            compute_request_hash,
            prepare_latent_prompt,
        )

        store_keys = []
        for i, problem in enumerate(problems):
            prompt = f"Analyze this math problem step by step: {problem}"
            ids = src_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            padded = prepare_latent_prompt(list(ids), latent_steps=10)
            store_key = compute_request_hash(padded)
            store_keys.append(store_key)

            params = vllm.SamplingParams(max_tokens=1, temperature=0.0)
            t0 = time.monotonic()
            outputs = engine_a.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded)], params,
            )
            elapsed = time.monotonic() - t0

            print(f"\n[{i+1}] Problem: {problem[:60]}...")
            print(f"    Tokens: {len(ids)} (+10 latent), Time: {elapsed:.2f}s")

            # Verify projected embedding
            proj_path = Path(store_dir) / store_key / "projected.pt"
            if proj_path.exists():
                proj = torch.load(proj_path, map_location="cpu", weights_only=True)
                print(f"    PROJECTED: shape={list(proj.shape)}, norm={proj.float().norm():.3f}")
                results[f"proj_{i}_shape"] = list(proj.shape)
                results[f"proj_{i}_norm"] = proj.float().norm().item()
            else:
                print("    WARNING: No projected embedding found!")

        # Free Agent A and clean up env vars
        del engine_a
        gc.collect()
        torch.cuda.empty_cache()
        os.environ.pop("AVP_TARGET_MODEL", None)
        os.environ.pop("AVP_LATENT_STEPS", None)
        print("\n--- Agent A freed, GPU memory released ---")

        # ==============================================================
        # AGENT B: Llama 3B — generate from projected embedding
        # ==============================================================
        print("\n" + "=" * 60)
        print(f"AGENT B: {TGT_MODEL}")
        print("=" * 60)

        engine_b = vllm.LLM(
            model=TGT_MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.7,
            enable_prompt_embeds=True,
            max_num_seqs=1,
        )

        from avp.connectors.vllm_kv_connector import generate_with_rosetta

        latent_answers = []
        text_answers = []

        for i, problem in enumerate(problems):
            solver_prompt = f"Solve step by step: {problem}"
            solver_ids = tgt_tokenizer.apply_chat_template(
                [{"role": "user", "content": solver_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            params = vllm.SamplingParams(max_tokens=256, temperature=0.0)

            # --- Text baseline ---
            t0 = time.monotonic()
            text_out = engine_b.generate(
                [vllm.TokensPrompt(prompt_token_ids=list(solver_ids))], params,
            )
            text_elapsed = time.monotonic() - t0
            text_answer = text_out[0].outputs[0].text.strip()
            text_answers.append(text_answer)

            # --- Latent (one-liner via generate_with_rosetta) ---
            t0 = time.monotonic()
            try:
                latent_out = generate_with_rosetta(
                    engine=engine_b,
                    prompt_token_ids=list(solver_ids),
                    store_dir=store_dir,
                    store_key=store_keys[i],
                    sampling_params=params,
                )
                latent_elapsed = time.monotonic() - t0
                latent_answer = latent_out[0].outputs[0].text.strip()
            except Exception as e:
                latent_answer = f"ERROR: {e}"
                latent_elapsed = time.monotonic() - t0

            latent_answers.append(latent_answer)

            print(f"\n[{i+1}] {problem[:60]}...")
            print(f"    Text ({text_elapsed:.2f}s):   {text_answer[:150]}")
            print(f"    Latent ({latent_elapsed:.2f}s): {latent_answer[:150]}")

        results["latent_answers"] = latent_answers
        results["text_answers"] = text_answers
        results["status"] = "done"

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return results


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print("\n\nResults:", json.dumps(result, indent=2, default=str))
