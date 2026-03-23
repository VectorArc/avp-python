"""Modal sanity: 2 GPUs, both models loaded simultaneously.

Agent A (Qwen 7B, GPU memory share) thinks, projects rosetta embedding.
Agent B (Qwen 1.5B, same GPU memory share) loads and generates.
Both engines exist at the same time — no sequential load/unload.

Usage:
    modal run benchmarks/modal_vllm_multi_gpu_sanity.py
"""

import modal

app = modal.App("avp-vllm-multi-gpu-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm>=0.17.0,<0.19.0", "torch>=2.0", "transformers>=5.0",
        "safetensors",
    )
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@main",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB:2", timeout=3600)
def run_test():
    import os
    import tempfile
    import time

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    TGT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    src_tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    tgt_tokenizer = AutoTokenizer.from_pretrained(TGT_MODEL)

    problems = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?",
    ]

    from avp.connectors.vllm_kv_connector import (
        generate_with_rosetta,
        make_sampling_params,
        prepare_latent_prompt,
    )

    with tempfile.TemporaryDirectory() as store_dir:
        # Pre-set env vars (model loads before connector)
        os.environ["AVP_LATENT_STEPS"] = "10"
        os.environ["AVP_TARGET_MODEL"] = TGT_MODEL

        # ==============================================================
        # Create BOTH engines simultaneously (2 GPUs)
        # ==============================================================
        print("=" * 60)
        print(f"Creating Agent A ({SRC_MODEL}) and Agent B ({TGT_MODEL})")
        print("Both engines loaded simultaneously on 2 GPUs")
        print("=" * 60)

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
            gpu_memory_utilization=0.45,
            kv_transfer_config=ktc_a,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        # Clear cross-model env before Agent B
        os.environ.pop("AVP_TARGET_MODEL", None)
        os.environ.pop("AVP_LATENT_STEPS", None)

        engine_b = vllm.LLM(
            model=TGT_MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.45,
            max_num_seqs=1,
        )

        print(f"\nBoth engines created. GPUs: {torch.cuda.device_count()}")

        # ==============================================================
        # Agent A thinks, Agent B generates (interleaved)
        # ==============================================================
        results = []
        for i, problem in enumerate(problems):
            store_key = f"multi-gpu-{i}"

            # Agent A: think
            researcher_prompt = f"Analyze this math problem: {problem}"
            ids_a = src_tokenizer.apply_chat_template(
                [{"role": "user", "content": researcher_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            padded = prepare_latent_prompt(list(ids_a), 10)
            think_params = make_sampling_params(store_key, max_tokens=1, temperature=0.0)

            t0 = time.monotonic()
            engine_a.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded)], think_params,
            )
            think_time = time.monotonic() - t0

            # Agent B: generate from rosetta embedding
            solver_prompt = f"Solve step by step: {problem}"
            solver_ids = tgt_tokenizer.apply_chat_template(
                [{"role": "user", "content": solver_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            gen_params = vllm.SamplingParams(max_tokens=256, temperature=0.0)

            t0 = time.monotonic()
            outputs = generate_with_rosetta(
                engine=engine_b,
                prompt_token_ids=list(solver_ids),
                store_dir=store_dir,
                store_key=store_key,
                sampling_params=gen_params,
            )
            gen_time = time.monotonic() - t0
            answer = outputs[0].outputs[0].text.strip()

            print(f"\n[{i+1}] {problem[:60]}...")
            print(f"    Think: {think_time:.2f}s, Generate: {gen_time:.2f}s")
            print(f"    Answer: {answer[:200]}")
            results.append({"problem": problem[:60], "answer": answer[:200]})

        del engine_a, engine_b

    print(f"\n{'='*60}")
    print("MULTI-GPU SANITY: PASSED")
    print(f"{'='*60}")
    return {"status": "passed", "results": results}


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print(json.dumps(result, indent=2, default=str))
