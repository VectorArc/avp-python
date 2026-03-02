"""Universal-mode pipeline for HotpotQA: cross-model Finder → Answerer.

Finder runs on model A (reads paragraphs → latent rollout → encode universal).
Answerer runs on model B (decode universal → KV-cache priming → generate).

Critical test: current rosetta = 7.5%. Target with universal: >30%.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS_2, build_latent_prompt, format_paragraphs
from .evaluate import exact_match, extract_answer, token_f1


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
    gold_answer: str,
    paragraphs: List[Dict],
    rollout_steps: int = 256,
    k_tokens: int = 64,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent universal pipeline on a single HotpotQA problem."""
    from avp.universal.encoder import UniversalEncoder
    from avp.universal.decoder import UniversalDecoder

    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        finder = AGENTS_2[0]
        answerer = AGENTS_2[1]

        # --- Agent 1: Finder on model A (encode to universal) ---
        messages = build_latent_prompt(finder.role, question, paragraphs_text=paragraphs_text)
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

        # Encode to universal space
        encode_t0 = time.perf_counter()
        encoder = UniversalEncoder.create(adapter_a.d_source, adapter_a.config)
        encoder.load_state_dict(adapter_a.encoder_state_dict)
        encoder = encoder.to(device)
        encoder.eval()

        with torch.no_grad():
            universal_tokens = encoder(hidden_states)

        if adapter_a.affine_out is not None:
            W = adapter_a.affine_out["W"].to(device, universal_tokens.dtype)
            b = adapter_a.affine_out["b"].to(device, universal_tokens.dtype)
            universal_tokens = universal_tokens @ W.T + b

        encode_ms = (time.perf_counter() - encode_t0) * 1000
        wire_bytes = universal_tokens.nelement() * universal_tokens.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": finder.name,
            "role": finder.role,
            "prompt_tokens": prompt_tokens,
            "rollout_steps": rollout_steps,
            "encode_ms": encode_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{finder.name}] rollout={rollout_steps}, "
                  f"encode={encode_ms:.1f}ms, wire={wire_bytes:,} bytes")

        del encoder, hidden_states
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Answerer on model B (decode + generate) ---
        decode_t0 = time.perf_counter()

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

        messages = build_latent_prompt(answerer.role, question)
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
            "name": answerer.name,
            "role": answerer.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{answerer.name}] decode={decode_ms:.1f}ms, "
                  f"output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + rollout_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer(agent_traces[-1]["output"])
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"],
        "correct": em,
        "exact_match": em,
        "f1": f1,
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run universal-mode pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Universal] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_universal_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, adapter_a, adapter_b,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            rollout_steps=rollout_steps,
            k_tokens=k_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["exact_match"] else "WRONG"
            print(f"  => {status} (pred='{result['prediction']}', "
                  f"gold='{result['gold']}', F1={result['f1']:.2f}, "
                  f"time={result['wall_time']:.1f}s)")
        else:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [Universal] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
