"""Latent-mode pipeline for fan-out benchmark: sequential KV injection.

Topology:
                  +--> Algebraist --+
Input (question) -+                +--> Aggregator --> Answer
                  +--> Arithmetician -+

Latent pipeline uses sequential KV injection (not concatenation):
1. Algebraist: generate_latent_steps(input_ids_a) → past_kv_a
2. AVP roundtrip → restored past_kv_a
3. Arithmetician: generate_latent_steps(input_ids_b, past_key_values=past_kv_a) → combined KV
4. AVP roundtrip → restored combined KV
5. Aggregator: generate_text(input_ids_agg, past_key_values=combined_kv) → answer

Sequential injection naturally extends positions (no RoPE conflicts from
concatenating independent KV-caches).
"""

import time
import uuid
from typing import Any, Dict, List

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGGREGATOR, SPECIALISTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    question: str,
    gold_solution: str,
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run fan-out latent pipeline on a single GSM8K problem.

    Specialists run sequentially (single GPU), each building on the prior KV-cache.
    The Aggregator generates text from the combined KV-cache.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        past_kv = None
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0
        specialist_times: List[float] = []

        # --- Specialists (sequential KV injection) ---
        for idx, specialist in enumerate(SPECIALISTS):
            next_name = (
                SPECIALISTS[idx + 1].name if idx + 1 < len(SPECIALISTS)
                else AGGREGATOR.name
            )

            messages = build_latent_prompt(specialist.role, question)
            prompt_text = render_prompt(tokenizer, messages)
            input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

            agent_t0 = time.perf_counter()
            prompt_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += prompt_tokens
            total_latent_steps += latent_steps

            # Generate latent steps, building on prior KV-cache
            past_kv = connector.generate_latent_steps(
                input_ids,
                latent_steps=latent_steps,
                attention_mask=attention_mask,
                past_key_values=past_kv,
            )

            # AVP codec roundtrip
            past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
                past_kv, session_id, specialist.name, next_name,
                model_name, identity, device,
            )
            total_codec_time_ms += codec_time_ms
            total_wire_bytes += wire_size

            kv_seq_len = get_past_length(past_kv)
            agent_time_ms = (time.perf_counter() - agent_t0) * 1000
            specialist_times.append(agent_time_ms)

            agent_traces.append({
                "name": specialist.name,
                "role": specialist.role,
                "prompt_tokens": prompt_tokens,
                "latent_steps": latent_steps,
                "kv_seq_len_after": kv_seq_len,
                "wire_bytes": wire_size,
                "codec_time_ms": codec_time_ms,
                "agent_time_ms": agent_time_ms,
                "output": "",
            })

            if verbose:
                print(f"  [{specialist.name}] latent steps={latent_steps}, "
                      f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                      f"codec={codec_time_ms:.1f}ms")

            # Free pre-roundtrip tensors to save VRAM
            if device == "cuda":
                torch.cuda.empty_cache()

        # --- Aggregator (text generation with combined KV-cache) ---
        messages = build_latent_prompt(AGGREGATOR.role, question)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        kv_seq_len = get_past_length(past_kv)

        text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer(text, add_special_tokens=False)
        output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += output_tokens
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": AGGREGATOR.name,
            "role": AGGREGATOR.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "kv_seq_len_input": kv_seq_len,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{AGGREGATOR.name}] KV input seq_len={kv_seq_len}, "
                  f"output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Parallel speedup potential: max specialist time instead of sum
    parallel_speedup_potential = (
        sum(specialist_times) / max(specialist_times) if specialist_times else 1.0
    )

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
        "total_latent_steps": total_latent_steps,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
        "parallel_speedup_potential": parallel_speedup_potential,
        "agents": agent_traces,
        "mode": "latent",
    }


def run_latent_benchmark(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    dataset: List[Dict],
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run fan-out latent pipeline on GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            question=sample["question"],
            gold_solution=sample["answer"],
            model_name=model_name,
            latent_steps=latent_steps,
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
                  f"codec={result['codec_overhead_ms']:.1f}ms, "
                  f"parallel_speedup={result['parallel_speedup_potential']:.1f}x)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Latent] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
