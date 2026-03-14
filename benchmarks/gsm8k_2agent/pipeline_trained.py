"""Trained projection pipeline: 2-agent chain with learned per-layer projections.

Researcher runs on model A (latent steps → extract hidden state).
Solver runs on model B (per-layer hooks inject trained projections → generate answer).

Requires a pre-trained AVPMap with layer_weights/layer_biases/layer_gates.
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_trained_pipeline(
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
) -> Dict:
    """Run the 2-agent pipeline with trained per-layer projection.

    Researcher (model A): latent steps → extract hidden state
    Solver (model B): per-layer hooks inject trained projections → generate text
    """
    from avp.rosetta.trained_hooks import trained_multi_layer_hook

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

        past_kv = conn_a.generate_latent_steps(
            input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
        )

        # Extract last hidden state
        proj_t0 = time.perf_counter()
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
        src_hidden = out.hidden_states[-1][:, -1, :].float()  # [1, D_src]

        # Pre-compute per-layer projections
        layer_projections = []
        active_count = 0
        for i, (w, b, gate) in enumerate(
            zip(avp_map.layer_weights, avp_map.layer_biases, avp_map.layer_gates)
        ):
            if gate < 0.01:
                layer_projections.append(None)
                continue
            projected = F.linear(
                src_hidden, w.to(device), b.to(device)
            )  # [1, D_tgt]
            layer_projections.append((projected, gate))
            active_count += 1

        projection_ms = (time.perf_counter() - proj_t0) * 1000

        agent_time_ms = (time.perf_counter() - agent_t0) * 1000
        agent_traces.append({
            "name": researcher.name,
            "role": researcher.role,
            "prompt_tokens": prompt_tokens,
            "latent_steps": latent_steps,
            "projection_ms": projection_ms,
            "active_layers": active_count,
            "total_layers": len(avp_map.layer_gates),
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{researcher.name}] latent steps={latent_steps}, "
                  f"projection={projection_ms:.1f}ms, "
                  f"active layers={active_count}/{len(avp_map.layer_gates)}")

        # Free model A KV-cache
        del past_kv
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Solver on model B (trained hooks, generate) ---
        messages = build_latent_prompt(solver.role, question)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        with trained_multi_layer_hook(model_b, layer_projections):
            text, _ = generate_text(
                model_b, tokenizer_b, input_ids, attention_mask, device,
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
        "active_layers": active_count,
        "total_layers": len(avp_map.layer_gates),
        "agents": agent_traces,
        "mode": "trained",
    }


def run_trained_benchmark(
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
) -> List[Dict]:
    """Run trained projection pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Trained] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_trained_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_solution=sample["answer"],
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
                  f"time={result['wall_time']:.1f}s)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Trained] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
