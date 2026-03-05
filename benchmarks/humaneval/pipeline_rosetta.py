"""Rosetta-mode pipeline: 2-agent chain with cross-model projection.

Coder runs on model A (latent steps -> extract hidden state -> project).
Reviewer runs on model B (inject projected embedding -> generate code).
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import check_correct


def run_rosetta_pipeline(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    avp_map: Any,
    prompt: str,
    test: str,
    entry_point: str,
    task_id: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
    num_transfer_states: int = 1,
) -> Dict:
    """Run the 2-agent cross-model pipeline on a single HumanEval problem."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        coder = AGENTS[0]
        reviewer = AGENTS[1]

        # --- Agent 1: Coder on model A (latent steps) ---
        messages = build_latent_prompt(coder.role, prompt)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens
        total_latent_steps += latent_steps

        attention_entropy = None
        if num_transfer_states > 1:
            past_kv, hidden_states = conn_a.generate_latent_steps(
                input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
                collect_hidden_states=True,
            )

            proj_t0 = time.perf_counter()
            selected = hidden_states[-num_transfer_states:]

            projected, proj_metrics = conn_a.project_hidden_for_cross_model(
                selected, avp_map, temperature=projection_temperature,
                return_metrics=True,
            )
            rosetta_embeds = projected.unsqueeze(0)  # [1, N, D_tgt]
        else:
            past_kv = conn_a.generate_latent_steps(
                input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
            )

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
                    output_attentions=True,
                    return_dict=True,
                )
            last_hidden = out.hidden_states[-1][:, -1, :]  # [1, D_src]

            # Compute attention entropy from last layer
            if out.attentions:
                last_attn = out.attentions[-1][:, :, -1, :]  # [batch, heads, seq_len]
                attn_log = torch.log(last_attn.clamp_min(1e-12))
                attn_ent = -(last_attn * attn_log).sum(dim=-1)  # [batch, heads]
                attention_entropy = float(attn_ent.mean())

            projected, proj_metrics = conn_a.project_hidden_for_cross_model(
                last_hidden, avp_map, temperature=projection_temperature,
                return_metrics=True,
            )
            rosetta_embeds = projected.unsqueeze(1)  # [1, 1, D_tgt]

        projection_ms = (time.perf_counter() - proj_t0) * 1000

        wire_bytes = rosetta_embeds.nelement() * rosetta_embeds.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": coder.name,
            "role": coder.role,
            "prompt_tokens": p_tokens,
            "latent_steps": latent_steps,
            "projection_ms": projection_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{coder.name}] latent steps={latent_steps}, "
                  f"projection={projection_ms:.1f}ms, "
                  f"projected shape={list(rosetta_embeds.shape)}, "
                  f"wire={wire_bytes:,} bytes")

        # Free model A KV-cache
        del past_kv
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Reviewer on model B (inject projected embed, generate) ---
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

        messages = build_latent_prompt(reviewer.role, prompt)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        agent_t0 = time.perf_counter()
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

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
            "name": reviewer.name,
            "role": reviewer.role,
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{reviewer.name}] output ({len(text)} chars): {text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    eval_result = check_correct(
        agent_traces[-1]["output"], prompt, test, entry_point,
    )
    correct = eval_result["passed"]

    return {
        "task_id": task_id,
        "prompt": prompt,
        "raw_output": agent_traces[-1]["output"],
        "extracted_code": eval_result["code"],
        "correct": correct,
        "error": eval_result["error"],
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
        "projection_entropy": float(proj_metrics["entropy"].mean()) if "entropy" in proj_metrics else None,
        "projection_max_prob": float(proj_metrics["max_prob"].mean()) if "max_prob" in proj_metrics else None,
        "projection_logit_gap": float(proj_metrics["logit_gap"].mean()) if "logit_gap" in proj_metrics else None,
        "hidden_state_norm": float(proj_metrics["hidden_state_norm"].mean()) if "hidden_state_norm" in proj_metrics else None,
        "nearest_cos_sim": float(proj_metrics["nearest_cos_sim"].mean()) if "nearest_cos_sim" in proj_metrics else None,
        "attention_entropy": attention_entropy,
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
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
    num_transfer_states: int = 1,
) -> List[Dict]:
    """Run rosetta-mode pipeline on a list of HumanEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Rosetta] Sample {i + 1}/{len(dataset)}: {sample['task_id']}")

        result = run_rosetta_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            prompt=sample["prompt"],
            test=sample["test"],
            entry_point=sample["entry_point"],
            task_id=sample["task_id"],
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
            status = "PASS" if result["correct"] else "FAIL"
            print(f"  => {status} (time={result['wall_time']:.1f}s, "
                  f"projection={result['projection_overhead_ms']:.1f}ms)")
        else:
            passed = sum(1 for r in results if r["correct"])
            print(f"  [Rosetta] {i + 1}/{len(dataset)} "
                  f"({passed}/{i + 1} passed, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
