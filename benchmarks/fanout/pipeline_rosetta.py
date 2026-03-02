"""Rosetta-mode pipeline for fan-out benchmark: cross-model specialists → aggregator.

Topology:
                  +--> Algebraist --+
Input (question) -+                +--> [project] --> Aggregator (model B) --> Answer
                  +--> Arithmetician -+

Both specialists run on model A with sequential KV injection (same as latent pipeline).
After both specialists, extract the last hidden state, project to model B space,
and inject into model B for the Aggregator to generate.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGGREGATOR, SPECIALISTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_rosetta_pipeline(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    avp_map: Any,
    question: str,
    gold_solution: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
    num_transfer_states: int = 1,
) -> Dict:
    """Run fan-out cross-model pipeline on a single GSM8K problem.

    Both specialists run on model A (sequential KV injection).
    Aggregator runs on model B (inject projected embedding(s)).

    When num_transfer_states > 1, collects hidden states from all specialists'
    latent steps and projects the last N from the concatenated sequence.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0
        specialist_times: List[float] = []
        past_kv = None
        all_hidden_states = []

        # --- Specialists on model A (sequential KV injection) ---
        for specialist in SPECIALISTS:
            messages = build_latent_prompt(specialist.role, question)
            prompt_text = render_prompt(tokenizer_a, messages)
            input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

            agent_t0 = time.perf_counter()
            prompt_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += prompt_tokens
            total_latent_steps += latent_steps

            if num_transfer_states > 1:
                past_kv, hidden_states = conn_a.generate_latent_steps(
                    input_ids,
                    latent_steps=latent_steps,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    collect_hidden_states=True,
                )
                all_hidden_states.append(hidden_states)
            else:
                past_kv = conn_a.generate_latent_steps(
                    input_ids,
                    latent_steps=latent_steps,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                )

            agent_time_ms = (time.perf_counter() - agent_t0) * 1000
            specialist_times.append(agent_time_ms)

            agent_traces.append({
                "name": specialist.name,
                "role": specialist.role,
                "prompt_tokens": prompt_tokens,
                "latent_steps": latent_steps,
                "agent_time_ms": agent_time_ms,
                "output": "",
            })

            if verbose:
                kv_len = get_past_length(past_kv)
                print(f"  [{specialist.name}] latent steps={latent_steps}, "
                      f"KV seq_len={kv_len}, time={agent_time_ms:.0f}ms")

        # --- Extract hidden state(s) and project ---
        proj_t0 = time.perf_counter()

        if num_transfer_states > 1:
            # Concatenate hidden states from all specialists → [total_steps, D]
            combined = torch.cat(all_hidden_states, dim=0)
            # Select last N from concatenated sequence
            selected = combined[-num_transfer_states:]

            projected = conn_a.project_hidden_for_cross_model(
                selected, avp_map, temperature=projection_temperature,
            )
            rosetta_embeds = projected.unsqueeze(0)  # [1, N, D_tgt]
        else:
            past_len = get_past_length(past_kv)
            dummy_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            eos_id = tokenizer_a.eos_token_id or 0
            dummy_ids = torch.tensor([[eos_id]], device=device)
            with torch.no_grad():
                out = model_a(
                    input_ids=dummy_ids,
                    attention_mask=dummy_mask,
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = out.hidden_states[-1][:, -1, :]  # [1, D_src]

            projected = conn_a.project_hidden_for_cross_model(
                last_hidden, avp_map, temperature=projection_temperature,
            )
            rosetta_embeds = projected.unsqueeze(1)  # [1, 1, D_tgt]

        projection_ms = (time.perf_counter() - proj_t0) * 1000

        wire_bytes = rosetta_embeds.nelement() * rosetta_embeds.element_size()

        if verbose:
            print(f"  [Projection] {projection_ms:.1f}ms, "
                  f"shape={list(rosetta_embeds.shape)}, wire={wire_bytes:,} bytes")

        # Free model A state
        del past_kv
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Aggregator on model B ---
        inject_t0 = time.perf_counter()
        embed_input = rosetta_embeds.to(device).to(model_b.dtype)
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
        injection_ms = (time.perf_counter() - inject_t0) * 1000

        messages = build_latent_prompt(AGGREGATOR.role, question)
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
            "name": AGGREGATOR.name,
            "role": AGGREGATOR.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{AGGREGATOR.name}] output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

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
        "projection_overhead_ms": projection_ms,
        "injection_overhead_ms": injection_ms,
        "projection_wire_bytes": wire_bytes,
        "num_transfer_states": num_transfer_states,
        "parallel_speedup_potential": parallel_speedup_potential,
        "agents": agent_traces,
        "mode": "rosetta",
    }


def run_rosetta_benchmark(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    avp_map: Any,
    dataset: List[Dict],
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
    num_transfer_states: int = 1,
) -> List[Dict]:
    """Run rosetta-mode pipeline on GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Rosetta] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_rosetta_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_solution=sample["answer"],
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            projection_temperature=projection_temperature,
            num_transfer_states=num_transfer_states,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"projection={result['projection_overhead_ms']:.1f}ms)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Rosetta] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
