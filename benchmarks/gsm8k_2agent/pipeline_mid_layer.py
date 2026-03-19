"""Mid-layer injection pipeline: 2-agent chain with intermediate layer injection.

Researcher runs on model A (latent steps -> extract hidden state -> project).
Solver runs on model B (inject at layer ~75% depth via forward hook -> generate).

Unlike rosetta (injects projected embedding at layer 0 via inputs_embeds),
mid-layer injects at an intermediate layer, operating directly in the
semantic representation space.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_mid_layer_pipeline(
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
    depth_ratio: float = 0.75,
    num_vectors: int = 1,
) -> Dict:
    """Run the 2-agent cross-model pipeline with mid-layer injection.

    Researcher (model A): latent steps -> extract hidden state -> project
    Solver (model B): inject at intermediate layer via forward hook -> generate
    """
    from avp.rosetta.mid_layer import (
        compute_activation_norm,
        compute_injection_layer,
        mid_layer_injection_hook,
        renormalize_to_activation_space,
    )

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        researcher = AGENTS[0]
        solver = AGENTS[1]

        # --- Agent 1: Researcher on model A (latent steps) ---
        messages = build_latent_prompt(researcher.role, question)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens
        total_latent_steps += latent_steps

        # Collect hidden states from all latent steps
        past_kv, hidden_states = conn_a.generate_latent_steps(
            input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
            collect_hidden_states=True,
        )

        # Select hidden states for projection
        if num_vectors == 1:
            selected = [hidden_states[-1].unsqueeze(0)]  # [1, D_src]
        else:
            # Evenly spaced from the latent steps
            n_available = len(hidden_states)
            indices = [int(i * n_available / num_vectors) for i in range(num_vectors)]
            indices[-1] = n_available - 1  # always include last
            selected = [hidden_states[i].unsqueeze(0) for i in indices]

        # Project each hidden state to target model space
        proj_t0 = time.perf_counter()
        projected_list = []
        proj_metrics = {}
        for h in selected:
            p, m = conn_a.project_hidden_for_cross_model(
                h, avp_map, return_metrics=True,
            )
            if p.dim() == 1:
                p = p.unsqueeze(0)
            if p.dim() == 3:
                p = p.squeeze(0)[-1:, :]
            projected_list.append(p)
            if not proj_metrics:
                proj_metrics = m

        # Stack into [N, D] for multi-vector or [1, D] for single
        projected = torch.cat(projected_list, dim=0)  # [N, D]
        projection_ms = (time.perf_counter() - proj_t0) * 1000

        wire_bytes = projected.nelement() * projected.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        # Compute injection layer
        target_num_layers = model_b.config.num_hidden_layers
        injection_layer = compute_injection_layer(target_num_layers, depth_ratio)

        # Renormalize from embedding-space norm to activation-space norm
        activation_norm = compute_activation_norm(
            model_b, tokenizer_b, injection_layer,
        )
        projected = renormalize_to_activation_space(projected, activation_norm)

        if verbose:
            print(f"  [{researcher.name}] activation_norm at layer {injection_layer}: "
                  f"{activation_norm:.1f}")

        agent_traces.append({
            "name": researcher.name,
            "role": researcher.role,
            "prompt_tokens": prompt_tokens,
            "latent_steps": latent_steps,
            "projection_ms": projection_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "injection_layer": injection_layer,
            "target_num_layers": target_num_layers,
            "output": "",
        })

        if verbose:
            print(f"  [{researcher.name}] latent steps={latent_steps}, "
                  f"projection={projection_ms:.1f}ms, "
                  f"inject at layer {injection_layer}/{target_num_layers} "
                  f"({100*injection_layer/target_num_layers:.0f}% depth)")

        # Free model A KV-cache
        del past_kv, hidden_states
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Solver on model B (mid-layer injection + generate) ---
        messages = build_latent_prompt(solver.role, question)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        # For multi-vector: prepend N pad tokens as placeholders for injection
        if num_vectors > 1:
            pad_id = tokenizer_b.pad_token_id or tokenizer_b.eos_token_id
            pad_ids = torch.full((1, num_vectors), pad_id, dtype=input_ids.dtype, device=device)
            pad_mask = torch.ones((1, num_vectors), dtype=attention_mask.dtype, device=device)
            input_ids = torch.cat([pad_ids, input_ids], dim=-1)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=-1)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        # Generate with mid-layer injection hook
        inject_hidden = projected.to(device).to(model_b.dtype)

        with mid_layer_injection_hook(model_b, injection_layer, inject_hidden, num_vectors=num_vectors):
            text, _ = generate_text(
                model_b, tokenizer_b, input_ids, attention_mask, device,
                past_key_values=None,  # No KV-cache priming
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
        "projection_wire_bytes": wire_bytes,
        "injection_layer": injection_layer,
        "depth_ratio": depth_ratio,
        "hidden_state_norm": float(proj_metrics["hidden_state_norm"].mean()) if "hidden_state_norm" in proj_metrics else None,
        "nearest_cos_sim": float(proj_metrics["nearest_cos_sim"].mean()) if "nearest_cos_sim" in proj_metrics else None,
        "agents": agent_traces,
        "mode": "mid_layer",
        "num_vectors": num_vectors,
    }


def run_mid_layer_benchmark(
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
    depth_ratio: float = 0.75,
    num_vectors: int = 1,
) -> List[Dict]:
    """Run mid-layer pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Mid-Layer] Sample {i + 1}/{len(dataset)}: "
                  f"{sample['question'][:80]}...")

        result = run_mid_layer_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_solution=sample["answer"],
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            depth_ratio=depth_ratio,
            num_vectors=num_vectors,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"time={result['wall_time']:.1f}s)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Mid-Layer d={depth_ratio}] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
