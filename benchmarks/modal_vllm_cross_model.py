"""Modal test: Cross-model rosetta via vLLM (2 GPUs).

Agent A (Qwen 7B, GPU 1) thinks via latent steps, projects to Llama 3B's
embedding space via rosetta, saves the projected embedding to a shared store.
Agent B (Llama 3B, GPU 2) loads the projected embedding as prompt_embeds
and generates from the primed context.

The rosetta projection runs on Agent A's side (CPU, ~1ms). Agent B only
needs to load a target-space embedding — no rosetta code, no source model.

Usage:
    modal run benchmarks/modal_vllm_cross_model.py
"""

import modal

app = modal.App("avp-vllm-cross-model")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("vllm>=0.17.0", "torch>=2.0", "transformers>=4.36", "safetensors")
    .pip_install(
        "git+https://github.com/VectorArc/avp-python.git@engine_integration",
        force_build=True,
    )
)


@app.function(image=image, gpu="A100-40GB:2", timeout=3600)
def run_cross_model_test():
    import json
    import tempfile
    import time
    from pathlib import Path

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm.config import KVTransferConfig

    SRC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    TGT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    src_tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL)
    tgt_tokenizer = AutoTokenizer.from_pretrained(TGT_MODEL)

    results = {}

    with tempfile.TemporaryDirectory() as store_dir:
        print(f"Store dir: {store_dir}")

        # ==============================================================
        # Agent A: Qwen 7B — think + project to Llama 3B space
        # ==============================================================
        print("\n" + "=" * 60)
        print(f"AGENT A: {SRC_MODEL} (think + rosetta project)")
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
            gpu_memory_utilization=0.4,
            kv_transfer_config=ktc_a,
            hf_overrides={"architectures": ["AVPLatentQwen2ForCausalLM"]},
            max_num_seqs=1,
            tensor_parallel_size=1,
        )

        # GSM8K-style problems for testing
        problems = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make to buy the wallet?",
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
            "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        ]

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

            print(f"\n[{i+1}] Prompt: {problem[:60]}...")
            print(f"    Tokens: {len(ids)} (+10 latent), Time: {elapsed:.2f}s")
            print(f"    Store key: {store_key}")

            # Verify projected embedding was saved
            projected_path = Path(store_dir) / store_key / "projected.pt"
            if projected_path.exists():
                proj = torch.load(projected_path, map_location="cpu", weights_only=True)
                print(f"    Projected shape: {list(proj.shape)}, norm: {proj.float().norm():.2f}")
            else:
                print("    WARNING: No projected embedding found!")

        # Free GPU memory for Agent A
        del engine_a
        torch.cuda.empty_cache()

        # ==============================================================
        # Agent B: Llama 3B — generate from projected embedding
        # ==============================================================
        print("\n" + "=" * 60)
        print(f"AGENT B: {TGT_MODEL} (generate from rosetta embedding)")
        print("=" * 60)

        engine_b = vllm.LLM(
            model=TGT_MODEL,
            enforce_eager=True,
            max_model_len=512,
            gpu_memory_utilization=0.4,
            enable_prompt_embeds=True,
            max_num_seqs=1,
            tensor_parallel_size=1,
        )

        # Load target model's embedding layer for prepending projected embedding
        from avp.connectors.vllm_kv_connector import load_projected_embedding

        latent_answers = []
        for i, problem in enumerate(problems):
            # Load projected embedding from store
            projected = load_projected_embedding(store_dir, store_keys[i])
            if projected is None:
                print(f"\n[{i+1}] SKIP: No projected embedding for {store_keys[i]}")
                latent_answers.append("")
                continue

            # Create solver prompt for Agent B
            solver_prompt = f"Solve step by step: {problem}"
            solver_ids = tgt_tokenizer.apply_chat_template(
                [{"role": "user", "content": solver_prompt}],
                tokenize=True, add_generation_prompt=True,
            )

            # Get the embedding layer to convert token IDs to embeddings
            # Then prepend the projected embedding as a virtual context token
            emb_layer = engine_b.llm_engine.model_executor.driver_worker.model_runner.model.model.embed_tokens
            solver_token_embeds = emb_layer.weight[solver_ids].detach()

            # Prepend projected embedding (unsqueeze to [1, D_tgt] if needed)
            projected_emb = projected.to(
                dtype=solver_token_embeds.dtype,
                device=solver_token_embeds.device,
            )
            if projected_emb.dim() == 1:
                projected_emb = projected_emb.unsqueeze(0)

            combined_embeds = torch.cat([projected_emb, solver_token_embeds], dim=0)

            params = vllm.SamplingParams(max_tokens=256, temperature=0.0)
            t0 = time.monotonic()
            outputs = engine_b.generate(
                [vllm.PromptEmbeds(prompt_embeds=combined_embeds)], params,
            )
            elapsed = time.monotonic() - t0
            answer = outputs[0].outputs[0].text.strip()
            latent_answers.append(answer)

            print(f"\n[{i+1}] Problem: {problem[:60]}...")
            print(f"    Latent answer ({elapsed:.2f}s): {answer[:200]}")

        # ==============================================================
        # Text baseline: Llama 3B generates from text only
        # ==============================================================
        print("\n" + "=" * 60)
        print("BASELINE: Text-only (Llama 3B)")
        print("=" * 60)

        text_answers = []
        for i, problem in enumerate(problems):
            prompt = f"Solve step by step: {problem}"
            ids = tgt_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True,
            )
            params = vllm.SamplingParams(max_tokens=256, temperature=0.0)
            t0 = time.monotonic()
            outputs = engine_b.generate(
                [vllm.TokensPrompt(prompt_token_ids=list(ids))], params,
            )
            elapsed = time.monotonic() - t0
            answer = outputs[0].outputs[0].text.strip()
            text_answers.append(answer)

            print(f"\n[{i+1}] Problem: {problem[:60]}...")
            print(f"    Text answer ({elapsed:.2f}s): {answer[:200]}")

        # ==============================================================
        # Summary
        # ==============================================================
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        results["problems"] = problems
        results["latent_answers"] = latent_answers
        results["text_answers"] = text_answers
        results["store_keys"] = store_keys
        results["src_model"] = SRC_MODEL
        results["tgt_model"] = TGT_MODEL

        for i, problem in enumerate(problems):
            print(f"\n[{i+1}] {problem[:60]}...")
            print(f"  Latent: {latent_answers[i][:100]}")
            print(f"  Text:   {text_answers[i][:100]}")

    return results


@app.local_entrypoint()
def main():
    import json

    result = run_cross_model_test.remote()
    print("\n\nFinal results:", json.dumps(result, indent=2, default=str))
