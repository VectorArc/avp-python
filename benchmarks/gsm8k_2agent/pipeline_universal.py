"""Universal-mode pipeline: 2-agent chain with universal representation transfer.

Researcher runs on model A (latent rollout → encode to universal tokens).
Solver runs on model B (decode universal tokens → KV-cache priming → generate).

Uses learned universal adapters — must be trained before running this pipeline.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_universal_pipeline(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    adapter_a: Any,
    adapter_b: Any,
    question: str,
    gold_solution: str,
    rollout_steps: int = 256,
    k_tokens: int = 64,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent universal pipeline on a single GSM8K problem.

    Researcher (model A): latent rollout → encode to universal tokens
    Solver (model B): decode universal → KV-cache priming → generate text
    """
    from avp.universal.encoder import UniversalEncoder
    from avp.universal.decoder import UniversalDecoder

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        researcher = AGENTS[0]
        solver = AGENTS[1]

        # --- Agent 1: Researcher on model A (encode to universal) ---
        messages = build_latent_prompt(researcher.role, question)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        # Collect hidden states from latent rollout
        _, hidden_states = conn_a.generate_latent_steps(
            input_ids, latent_steps=rollout_steps, attention_mask=attention_mask,
            collect_hidden_states=True,
        )

        # Build and load encoder
        encode_t0 = time.perf_counter()
        encoder = UniversalEncoder.create(adapter_a.d_source, adapter_a.config)
        encoder.load_state_dict(adapter_a.encoder_state_dict)
        encoder = encoder.to(device)
        encoder.eval()

        with torch.no_grad():
            universal_tokens = encoder(hidden_states)

        # Apply affine_out if present
        if adapter_a.affine_out is not None:
            W = adapter_a.affine_out["W"].to(device, universal_tokens.dtype)
            b = adapter_a.affine_out["b"].to(device, universal_tokens.dtype)
            universal_tokens = universal_tokens @ W.T + b

        encode_ms = (time.perf_counter() - encode_t0) * 1000
        wire_bytes = universal_tokens.nelement() * universal_tokens.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": researcher.name,
            "role": researcher.role,
            "prompt_tokens": prompt_tokens,
            "rollout_steps": rollout_steps,
            "encode_ms": encode_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{researcher.name}] rollout={rollout_steps}, "
                  f"encode={encode_ms:.1f}ms, "
                  f"tokens shape={list(universal_tokens.shape)}, "
                  f"wire={wire_bytes:,} bytes")

        # Free encoder and model A state
        del encoder, hidden_states
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Solver on model B (decode + KV-cache priming + generate) ---
        decode_t0 = time.perf_counter()

        # Apply affine_in if present
        if adapter_b.affine_in is not None:
            W = adapter_b.affine_in["W"].to(device, universal_tokens.dtype)
            b = adapter_b.affine_in["b"].to(device, universal_tokens.dtype)
            universal_tokens = universal_tokens @ W.T + b

        decoder = UniversalDecoder.create(adapter_b.d_source, adapter_b.config)
        decoder.load_state_dict(adapter_b.decoder_state_dict)
        decoder = decoder.to(device)
        decoder.eval()

        with torch.no_grad():
            decoded, gate = decoder(universal_tokens, target_norm=adapter_b.target_norm)

        if decoded.dim() == 2:
            decoded = decoded.unsqueeze(0)

        # KV-cache priming
        embed_input = decoded.to(device).to(model_b.dtype) * gate
        embed_mask = torch.ones(
            (1, embed_input.shape[1]), dtype=torch.long, device=device,
        )
        with torch.no_grad():
            prime_out = model_b(
                inputs_embeds=embed_input,
                attention_mask=embed_mask,
                use_cache=True,
                return_dict=True,
            )
        past_kv_b = prime_out.past_key_values
        decode_ms = (time.perf_counter() - decode_t0) * 1000

        del decoder
        if device == "cuda":
            torch.cuda.empty_cache()

        messages = build_latent_prompt(solver.role, question)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        text, _ = generate_text(
            model_b, tokenizer_b, input_ids, attention_mask, device,
            past_key_values=past_kv_b,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer_b(text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": solver.name,
            "role": solver.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{solver.name}] decode={decode_ms:.1f}ms, gate={gate:.3f}, "
                  f"output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + rollout_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    gold = extract_gold(gold_solution)
    prediction = extract_gsm8k_answer(agent_traces[-1]["output"])
    correct = check_correct(prediction, gold)

    return {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"],
        "correct": correct,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_rollout_steps": rollout_steps,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "encode_overhead_ms": encode_ms,
        "decode_overhead_ms": decode_ms,
        "universal_wire_bytes": wire_bytes,
        "k_tokens": k_tokens,
        "gate": gate,
        "agents": agent_traces,
        "mode": "universal",
    }


def run_universal_benchmark(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    adapter_a: Any,
    adapter_b: Any,
    dataset: List[Dict],
    rollout_steps: int = 256,
    k_tokens: int = 64,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run universal-mode pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Universal] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_universal_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, adapter_a, adapter_b,
            question=sample["question"],
            gold_solution=sample["answer"],
            rollout_steps=rollout_steps,
            k_tokens=k_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"encode={result['encode_overhead_ms']:.1f}ms, "
                  f"decode={result['decode_overhead_ms']:.1f}ms)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Universal] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
