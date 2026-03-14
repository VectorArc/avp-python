"""Logit-guided pipeline: 2-agent chain with cross-model logit bias.

Researcher runs on model A (latent steps → extract hidden state → compute logits).
Solver runs on model B (generate with logit bias from mapped source distribution).

Unlike rosetta (single virtual token in KV-cache), logit-guided distributes
the source signal across the target's entire autoregressive generation.
"""

import time
from typing import Any, Dict, List

import torch
from transformers import LogitsProcessorList

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import extract_gold, extract_gsm8k_answer, check_correct


def run_logit_guided_pipeline(
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
    logit_bias_alpha: float = 0.5,
    logit_bias_confidence_threshold: float = 0.8,
) -> Dict:
    """Run the 2-agent cross-model pipeline with logit-guided decoding.

    Researcher (model A): latent steps → extract hidden state → compute logits
    Solver (model B): generate with mapped logit bias (no KV-cache priming)
    """
    from avp.rosetta.logit_guided import (
        CrossModelLogitBias,
        compute_cross_model_logit_bias,
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

        # Run latent steps and collect hidden states
        past_kv, hidden_states = conn_a.generate_latent_steps(
            input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
            collect_hidden_states=True,
        )

        # Use last hidden state for logit computation
        last_hidden = hidden_states[-1].unsqueeze(0)  # [1, D_src]

        # Compute logit bias
        bias_t0 = time.perf_counter()

        source_lm_head = model_a.get_output_embeddings()
        if source_lm_head is None:
            source_lm_head = getattr(model_a, "lm_head", None)

        target_vocab_size = model_b.config.vocab_size
        bias = compute_cross_model_logit_bias(
            source_hidden_state=last_hidden,
            source_lm_head_weight=source_lm_head.weight,
            avp_map=avp_map,
            target_vocab_size=target_vocab_size,
        )

        bias_ms = (time.perf_counter() - bias_t0) * 1000
        nonzero_count = int((bias != 0).sum())
        bias_magnitude = float(bias[bias != 0].abs().mean()) if nonzero_count > 0 else 0.0

        # Wire size: just the bias tensor (much smaller than text, larger than rosetta embed)
        wire_bytes = bias.nelement() * bias.element_size()

        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": researcher.name,
            "role": researcher.role,
            "prompt_tokens": prompt_tokens,
            "latent_steps": latent_steps,
            "bias_compute_ms": bias_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{researcher.name}] latent steps={latent_steps}, "
                  f"bias compute={bias_ms:.1f}ms, "
                  f"nonzero={nonzero_count}/{target_vocab_size}, "
                  f"mean_abs_bias={bias_magnitude:.4f}")

        # Free model A KV-cache
        del past_kv, hidden_states
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Solver on model B (generate with logit bias) ---
        messages = build_latent_prompt(solver.role, question)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        # Create logit bias processor
        processor = CrossModelLogitBias(
            bias=bias,
            alpha=logit_bias_alpha,
            confidence_threshold=logit_bias_confidence_threshold,
        )

        text, _ = generate_text(
            model_b, tokenizer_b, input_ids, attention_mask, device,
            past_key_values=None,  # No KV-cache priming
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logits_processor=LogitsProcessorList([processor]),
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
        "bias_compute_ms": bias_ms,
        "logit_bias_alpha": logit_bias_alpha,
        "logit_bias_confidence_threshold": logit_bias_confidence_threshold,
        "bias_nonzero_count": nonzero_count,
        "bias_mean_magnitude": bias_magnitude,
        "wire_bytes": wire_bytes,
        "agents": agent_traces,
        "mode": "logit_guided",
    }


def run_logit_guided_benchmark(
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
    logit_bias_alpha: float = 0.5,
    logit_bias_confidence_threshold: float = 0.8,
) -> List[Dict]:
    """Run logit-guided pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Logit-Guided] Sample {i + 1}/{len(dataset)}: "
                  f"{sample['question'][:80]}...")

        result = run_logit_guided_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_solution=sample["answer"],
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            logit_bias_alpha=logit_bias_alpha,
            logit_bias_confidence_threshold=logit_bias_confidence_threshold,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  => {status} (pred={result['prediction']}, gold={result['gold']}, "
                  f"time={result['wall_time']:.1f}s)")
        else:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [Logit-Guided a={logit_bias_alpha}] {i + 1}/{len(dataset)} "
                  f"({correct}/{i + 1} correct, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
