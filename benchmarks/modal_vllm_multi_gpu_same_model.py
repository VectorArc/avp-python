"""Modal sanity: 2 GPUs, same-model KV transfer via connector.

Agent A (Qwen 7B, GPU share) thinks with latent steps, saves enriched KV.
Agent B (Qwen 7B, GPU share) loads enriched KV via connector, generates.
Both engines exist simultaneously. Tests the full KV connector load path
(get_num_new_matched_tokens → start_load_kv → _inject_request_kv).

Usage:
    modal run benchmarks/modal_vllm_multi_gpu_same_model.py
"""

import modal

app = modal.App("avp-vllm-multi-gpu-same-model")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0,<0.19.0", "torch>=2.0", "transformers>=5.0")
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

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LATENT_STEPS = 10

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    problems = [
        ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
        ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?", "5"),
        ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
    ]

    from avp.connectors.vllm_kv_connector import (
        make_sampling_params,
        prepare_latent_prompt,
    )

    with tempfile.TemporaryDirectory() as store_dir:
        # Pre-set env vars (model loads before connector)
        os.environ["AVP_LATENT_STEPS"] = str(LATENT_STEPS)

        # ==============================================================
        # Create BOTH engines simultaneously
        # ==============================================================
        print("=" * 60)
        print(f"Creating Agent A and Agent B (both {MODEL})")
        print("Both engines loaded simultaneously, shared FileKVStore")
        print("=" * 60)

        ktc_a = KVTransferConfig(
            kv_connector="AVPKVConnectorV1Dynamic",
            kv_connector_module_path="avp.connectors.vllm_kv_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "avp_latent_steps": LATENT_STEPS,
                "avp_store_dir": store_dir,
            },
        )

        engine_a = vllm.LLM(
            model=MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.45,
            kv_transfer_config=ktc_a,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
        )

        # Agent B: loads KV via connector (same-model path)
        os.environ.pop("AVP_LATENT_STEPS", None)

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
            model=MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.45,
            kv_transfer_config=ktc_b,
            max_num_seqs=1,
        )

        print(f"\nBoth engines created. GPUs: {torch.cuda.device_count()}")

        # ==============================================================
        # Agent A thinks, Agent B generates (interleaved)
        # ==============================================================
        results = []
        for i, (question, gold) in enumerate(problems):
            store_key = f"same-model-{i}"

            # Agent A: think with researcher prompt
            researcher_prompt = f"Analyze this math problem carefully: {question}"
            ids_a = tokenizer.apply_chat_template(
                [{"role": "user", "content": researcher_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            padded = prepare_latent_prompt(list(ids_a), LATENT_STEPS)
            think_params = make_sampling_params(store_key, max_tokens=1, temperature=0.0)

            t0 = time.monotonic()
            engine_a.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded)], think_params,
            )
            think_time = time.monotonic() - t0

            # Agent B: generate with solver prompt, loads KV via connector
            solver_prompt = f"Solve step by step: {question}"
            ids_b = tokenizer.apply_chat_template(
                [{"role": "user", "content": solver_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            padded_b = prepare_latent_prompt(list(ids_b), LATENT_STEPS)
            gen_params = make_sampling_params(store_key, max_tokens=256, temperature=0.0)

            t0 = time.monotonic()
            outputs = engine_b.generate(
                [vllm.TokensPrompt(prompt_token_ids=padded_b)], gen_params,
            )
            gen_time = time.monotonic() - t0
            answer = outputs[0].outputs[0].text.strip()

            # Check for gold answer
            correct = gold in answer
            print(f"\n[{i+1}] {question[:60]}...")
            print(f"    Think: {think_time:.2f}s, Generate: {gen_time:.2f}s")
            print(f"    Gold: {gold}, Found: {'YES' if correct else 'NO'}")
            print(f"    Answer: {answer[:200]}")
            results.append({"correct": correct, "gold": gold, "answer": answer[:200]})

        # ==============================================================
        # Text baseline (no KV transfer, no latent)
        # ==============================================================
        print(f"\n{'='*60}")
        print("TEXT BASELINE (Agent B only, no latent)")
        print("=" * 60)

        for i, (question, gold) in enumerate(problems):
            solver_prompt = f"Solve step by step: {question}"
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": solver_prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            params = vllm.SamplingParams(max_tokens=256, temperature=0.0)
            outputs = engine_b.generate(
                [vllm.TokensPrompt(prompt_token_ids=list(ids))], params,
            )
            answer = outputs[0].outputs[0].text.strip()
            correct = gold in answer
            print(f"  [{i+1}] Gold: {gold}, Found: {'YES' if correct else 'NO'}")

        del engine_a, engine_b

    latent_correct = sum(1 for r in results if r["correct"])
    print(f"\n{'='*60}")
    print(f"SAME-MODEL KV TRANSFER (2 GPUs): {latent_correct}/{len(problems)} correct")
    print("=" * 60)
    return {"status": "passed", "correct": latent_correct, "total": len(problems), "results": results}


@app.local_entrypoint()
def main():
    import json
    result = run_test.remote()
    print(json.dumps(result, indent=2, default=str))
