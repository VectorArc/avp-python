"""Soft-vocab pipeline: multi-position cross-model transfer via forward-pass hidden states.

Key insight: hidden states from a forward pass on real tokens are on-distribution —
each position encodes a well-defined next-token prediction. When projected through
vocab-overlap/vocab-mediated projection, they produce on-manifold embeddings
(weighted sums of real target embeddings).

This tests whether N>1 projected embeddings carry more information than the
single-embedding rosetta baseline. Also supports state_source="latent" as a
control (known to fail at N>1 — noise compounds from off-distribution states).

Researcher (model A): forward pass → extract N hidden states → project each
Solver (model B): inject [1, N, D_tgt] via inputs_embeds → generate answer
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def _select_positions(
    hidden_states: torch.Tensor,
    num_states: int,
    strategy: str,
) -> tuple:
    """Select positions from hidden states tensor [1, seq_len, D].

    Returns:
        (selected_hidden, selected_indices) where selected_hidden is [N, D]
        and selected_indices is a list of ints.
    """
    seq_len = hidden_states.shape[1]
    num_states = min(num_states, seq_len)

    if strategy == "stride":
        # Evenly spaced positions across the sequence
        indices = torch.linspace(0, seq_len - 1, num_states).long().tolist()
    else:
        # last_n: take the last N positions (highest context density)
        indices = list(range(seq_len - num_states, seq_len))

    selected = hidden_states[0, indices, :]  # [N, D]
    return selected, indices


def run_soft_vocab_pipeline(
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
    num_transfer_states: int = 16,
    position_strategy: str = "last_n",
    state_source: str = "text",
    latent_steps: int = 10,
    projection_temperature: float = 1.0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent soft-vocab pipeline on a single GSM8K problem.

    Agent 1 (Researcher): extracts hidden states via forward pass (text) or
    latent rollout (latent), selects N positions, projects each through
    vocab-overlap/vocab-mediated projection.

    Agent 2 (Solver): receives [1, N, D_tgt] projected embeddings, primes
    KV-cache via inputs_embeds, generates answer.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        researcher = AGENTS[0]
        solver = AGENTS[1]

        # --- Agent 1: Researcher on model A ---
        messages = build_latent_prompt(researcher.role, question)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        proj_t0 = time.perf_counter()

        if state_source == "text":
            # Forward pass on real tokens — hidden states are on-distribution
            with torch.no_grad():
                out = model_a(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # Last layer hidden states: [1, seq_len, D_src]
            all_hidden = out.hidden_states[-1]
            source_seq_len = all_hidden.shape[1]

            # Select N positions
            selected, selected_positions = _select_positions(
                all_hidden, num_transfer_states, position_strategy,
            )
        else:
            # Latent rollout — known to fail at N>1 (control condition)
            total_latent_steps += latent_steps
            _, hidden_states = conn_a.generate_latent_steps(
                input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
                collect_hidden_states=True,
            )
            source_seq_len = len(hidden_states)

            # Select last N from latent rollout steps
            n = min(num_transfer_states, len(hidden_states))
            selected = hidden_states[-n:]  # list of [1, D] or stacked
            if isinstance(selected, list):
                selected = torch.stack(selected, dim=0).squeeze(1)  # [N, D]
            selected_positions = list(range(len(hidden_states) - n, len(hidden_states)))

        # Project batch to target space → [N, D_tgt]
        projected, proj_metrics = conn_a.project_hidden_for_cross_model(
            selected, avp_map, temperature=projection_temperature,
            return_metrics=True,
        )
        rosetta_embeds = projected.unsqueeze(0)  # [1, N, D_tgt]

        projection_ms = (time.perf_counter() - proj_t0) * 1000

        wire_bytes = rosetta_embeds.nelement() * rosetta_embeds.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": researcher.name,
            "role": researcher.role,
            "prompt_tokens": prompt_tokens,
            "latent_steps": latent_steps if state_source == "latent" else 0,
            "projection_ms": projection_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
            "state_source": state_source,
            "num_transfer_states": num_transfer_states,
            "position_strategy": position_strategy,
            "source_seq_len": source_seq_len,
            "selected_positions": selected_positions,
        })

        if verbose:
            print(f"  [{researcher.name}] state_source={state_source}, "
                  f"positions={position_strategy}, N={len(selected_positions)}, "
                  f"seq_len={source_seq_len}, "
                  f"projection={projection_ms:.1f}ms, "
                  f"shape={list(rosetta_embeds.shape)}, "
                  f"wire={wire_bytes:,} bytes")

        # Free model A state
        if state_source == "text":
            del out
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Solver on model B (inject projected embeds, generate) ---
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
            print(f"  [{solver.name}] output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
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
        "total_latent_steps": total_latent_steps,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "projection_overhead_ms": projection_ms,
        "injection_overhead_ms": injection_ms,
        "projection_wire_bytes": wire_bytes,
        "num_transfer_states": num_transfer_states,
        "state_source": state_source,
        "position_strategy": position_strategy,
        "source_seq_len": source_seq_len,
        "selected_positions": selected_positions,
        "projection_entropy": float(proj_metrics["entropy"].mean()) if "entropy" in proj_metrics else None,
        "projection_max_prob": float(proj_metrics["max_prob"].mean()) if "max_prob" in proj_metrics else None,
        "agents": agent_traces,
        "mode": "soft_vocab",
    }


def run_soft_vocab_benchmark(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    avp_map: Any,
    dataset: List[Dict],
    num_transfer_states: int = 16,
    position_strategy: str = "last_n",
    state_source: str = "text",
    latent_steps: int = 10,
    projection_temperature: float = 1.0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run soft-vocab pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[SoftVocab] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_soft_vocab_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_solution=sample["answer"],
            num_transfer_states=num_transfer_states,
            position_strategy=position_strategy,
            state_source=state_source,
            latent_steps=latent_steps,
            projection_temperature=projection_temperature,
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
                  f"projection={result['projection_overhead_ms']:.1f}ms)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [SoftVocab] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
