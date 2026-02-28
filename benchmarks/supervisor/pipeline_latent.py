"""Latent-mode pipeline for Supervisor benchmark: Router + Specialist via KV-cache.

Topology:
    Router --> select 1 of 4 Specialists --> Answer

Hybrid text+KV pattern:
1. Router generates TEXT (to extract route) + returns KV-cache
2. AVP roundtrip of Router's KV-cache
3. Selected Specialist generates text with Router's KV context
"""

import time
import uuid
from typing import Any, Dict, List

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import (
    ROUTER,
    SPECIALISTS,
    SUBJECT_TO_CATEGORY,
    build_latent_specialist_prompt,
    build_router_prompt,
    extract_route,
)
from .evaluate import check_correct, extract_answer_letter


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    question: str,
    choices: List[str],
    gold_answer: int,
    model_name: str,
    subject: str = "",
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run Router + Specialist latent pipeline on a single MMLU question.

    The Router ALWAYS generates text (to extract routing decision).
    Its KV-cache is then passed to the selected Specialist.
    """
    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_output_tokens = 0

        # --- Step 1: Router generates TEXT + returns KV-cache ---
        router_t0 = time.perf_counter()
        messages = build_router_prompt(question, choices)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        router_text, router_past_kv = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            max_new_tokens=64,
            temperature=temperature,
            top_p=top_p,
        )

        route = extract_route(router_text)
        specialist = SPECIALISTS.get(route, SPECIALISTS["stem"])

        output_encoded = tokenizer(router_text, add_special_tokens=False)
        router_output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += router_output_tokens

        # --- Step 2: AVP roundtrip of Router's KV-cache ---
        if router_past_kv is not None:
            router_past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
                router_past_kv, session_id, ROUTER.name, specialist.name,
                model_name, identity, device,
            )
            total_codec_time_ms += codec_time_ms
            total_wire_bytes += wire_size
        else:
            codec_time_ms = 0.0
            wire_size = 0

        router_time_ms = (time.perf_counter() - router_t0) * 1000

        agent_traces.append({
            "name": ROUTER.name,
            "role": ROUTER.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": router_output_tokens,
            "kv_seq_len_after": get_past_length(router_past_kv),
            "wire_bytes": wire_size,
            "codec_time_ms": codec_time_ms,
            "agent_time_ms": router_time_ms,
            "output": router_text,
            "route": route,
        })

        if verbose:
            print(f"  [Router] route={route}, KV seq_len={get_past_length(router_past_kv)}, "
                  f"wire={wire_size:,} bytes")

        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Step 3: Selected Specialist generates with Router's KV context ---
        specialist_t0 = time.perf_counter()
        messages = build_latent_specialist_prompt(question, choices)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        kv_seq_len = get_past_length(router_past_kv)

        specialist_text, _ = generate_text(
            model, tokenizer, input_ids, attention_mask, device,
            past_key_values=router_past_kv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_encoded = tokenizer(specialist_text, add_special_tokens=False)
        specialist_output_tokens = len(output_encoded["input_ids"])
        total_output_tokens += specialist_output_tokens
        specialist_time_ms = (time.perf_counter() - specialist_t0) * 1000

        agent_traces.append({
            "name": specialist.name,
            "role": specialist.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": specialist_output_tokens,
            "kv_seq_len_input": kv_seq_len,
            "agent_time_ms": specialist_time_ms,
            "output": specialist_text,
        })

        if verbose:
            print(f"  [{specialist.name}] KV input seq_len={kv_seq_len}, "
                  f"output: {specialist_text[:200]}...")

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer_letter(specialist_text)
    correct = check_correct(prediction, gold_answer)
    expected_category = SUBJECT_TO_CATEGORY.get(subject, "unknown")
    routing_correct = (route == expected_category)

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": specialist_text,
        "correct": correct,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
        "agents": agent_traces,
        "subject": subject,
        "expected_category": expected_category,
        "route": route,
        "routing_correct": routing_correct,
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode pipeline on MMLU samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            question=sample["question"],
            choices=sample["choices"],
            gold_answer=sample["answer"],
            model_name=model_name,
            subject=sample.get("subject", ""),
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
                  f"route={result['route']}, time={result['wall_time']:.1f}s, "
                  f"codec={result['codec_overhead_ms']:.1f}ms)")

    return results
