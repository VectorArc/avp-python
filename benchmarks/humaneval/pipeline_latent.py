"""Latent-mode pipeline: 2-agent chain with KV-cache transfer via AVP codec.

Coder generates latent reasoning -> AVP roundtrip -> Reviewer generates final code.
"""

import time
import uuid
from typing import Any, Dict, List

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt
from .evaluate import check_correct


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    prompt: str,
    test: str,
    entry_point: str,
    task_id: str,
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent latent-mode pipeline on a single HumanEval problem."""
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        # --- Agent 1: Coder (latent steps) ---
        coder = AGENTS[0]
        reviewer = AGENTS[1]

        messages = build_latent_prompt(coder.role, prompt)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

        agent_t0 = time.perf_counter()
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens
        total_latent_steps += latent_steps

        past_kv = connector.generate_latent_steps(
            input_ids,
            latent_steps=latent_steps,
            attention_mask=attention_mask,
        )

        # AVP codec roundtrip
        past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
            past_kv, session_id, coder.name, reviewer.name,
            model_name, identity, device,
        )
        total_codec_time_ms += codec_time_ms
        total_wire_bytes += wire_size

        kv_seq_len = get_past_length(past_kv)
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": coder.name,
            "role": coder.role,
            "prompt_tokens": p_tokens,
            "latent_steps": latent_steps,
            "kv_seq_len_after": kv_seq_len,
            "wire_bytes": wire_size,
            "codec_time_ms": codec_time_ms,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{coder.name}] latent steps={latent_steps}, "
                  f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                  f"codec={codec_time_ms:.1f}ms")

        # --- Agent 2: Reviewer (text generation with KV-cache) ---
        messages = build_latent_prompt(reviewer.role, prompt)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

        agent_t0 = time.perf_counter()
        p_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += p_tokens

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
            "name": reviewer.name,
            "role": reviewer.role,
            "prompt_tokens": p_tokens,
            "output_tokens": output_tokens,
            "kv_seq_len_input": kv_seq_len,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{reviewer.name}] KV input seq_len={kv_seq_len}, "
                  f"output ({len(text)} chars): {text[:200]}...")

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
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
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
    temperature: float = 0.01,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode pipeline on a list of HumanEval samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: {sample['task_id']}")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            prompt=sample["prompt"],
            test=sample["test"],
            entry_point=sample["entry_point"],
            task_id=sample["task_id"],
            model_name=model_name,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "PASS" if result["correct"] else "FAIL"
            print(f"  => {status} (time={result['wall_time']:.1f}s, "
                  f"codec={result['codec_overhead_ms']:.1f}ms, "
                  f"wire={result['avp_wire_bytes']:,} bytes)")

    return results
