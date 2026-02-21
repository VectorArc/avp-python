"""Hybrid-mode pipeline: agents communicate via KV-cache + text summary through AVP codec.

Each non-judger agent generates latent steps (KV-cache) AND a short text summary,
packing both into a HYBRID AVP message via encode_hybrid(). The receiver decodes
and uses the latent KV-cache for generation; the text_fallback is logged in the
agent trace for observability.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import torch

from .agents import (
    AGENTS,
    build_latent_prompt,
    generate_text,
    render_prompt,
    tokenize_prompt,
)
from .evaluate import extract_gold, extract_gsm8k_answer
from .pipeline_latent import _get_past_length


def run_hybrid_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    question: str,
    gold_solution: str,
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    summary_max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 4-agent hybrid-mode pipeline on a single GSM8K problem.

    Non-judger agents use generate_latent_steps to produce KV-cache, then generate
    a short text summary from the accumulated cache. Both are packed into a HYBRID
    AVP message. The Judger generates text with the accumulated KV-cache.

    Args:
        summary_max_tokens: Max tokens for the text summary per hop (default: 128).
    """
    from avp.codec import decode as avp_decode
    from avp.codec import encode_hybrid
    from avp.kv_cache import (
        deserialize_kv_cache,
        legacy_to_dynamic_cache,
        serialize_kv_cache,
    )
    from avp.types import (
        AVPMetadata,
        CommunicationMode,
        DataType,
        PayloadType,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.max_memory_allocated()

    t0 = time.perf_counter()
    session_id = str(uuid.uuid4())
    past_kv = None
    agent_traces: List[Dict] = []
    total_codec_time_ms = 0.0
    total_wire_bytes = 0
    total_cross_model_ms = 0.0
    total_prompt_tokens = 0
    total_latent_steps = 0
    total_summary_tokens = 0
    total_output_tokens = 0

    for idx, agent in enumerate(AGENTS):
        next_agent = AGENTS[idx + 1] if idx + 1 < len(AGENTS) else None

        # Build and tokenize prompt
        messages = build_latent_prompt(agent.role, question)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        if agent.role != "judger":
            # --- Latent steps: generate KV-cache ---
            total_latent_steps += latent_steps
            past_kv = connector.generate_latent_steps(
                input_ids,
                latent_steps=latent_steps,
                attention_mask=attention_mask,
                past_key_values=past_kv,
            )

            # --- Serialize KV-cache BEFORE summary generation ---
            # model.generate() mutates DynamicCache in place, appending
            # summary tokens. Serialize first to capture only the latent KV.
            kv_bytes, kv_header = serialize_kv_cache(past_kv)

            # --- Text summary: generate short text from accumulated KV ---
            # past_kv is mutated by generate_text (summary tokens appended),
            # but we already have kv_bytes from the clean cache above.
            summary_t0 = time.perf_counter()
            summary_text, _ = generate_text(
                model, tokenizer, input_ids, attention_mask, device,
                past_key_values=past_kv,
                max_new_tokens=summary_max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            summary_time_ms = (time.perf_counter() - summary_t0) * 1000

            summary_encoded = tokenizer(summary_text, add_special_tokens=False)
            summary_tokens = len(summary_encoded["input_ids"])
            total_summary_tokens += summary_tokens

            # --- AVP hybrid codec roundtrip ---
            codec_t0 = time.perf_counter()

            metadata = AVPMetadata(
                session_id=session_id,
                source_agent_id=agent.name,
                target_agent_id=next_agent.name if next_agent else "",
                model_id=model_name,
                hidden_dim=identity.hidden_dim,
                num_layers=identity.num_layers,
                payload_type=PayloadType.KV_CACHE,
                dtype=DataType.FLOAT32,
                mode=CommunicationMode.HYBRID,
            )

            wire_bytes = encode_hybrid(kv_bytes, summary_text, metadata)
            wire_size = len(wire_bytes)
            total_wire_bytes += wire_size

            # Decode on "receiving" side
            avp_msg = avp_decode(wire_bytes)
            assert avp_msg.text_fallback is not None  # sanity check
            legacy_kv, _ = deserialize_kv_cache(avp_msg.payload, device=device)
            past_kv = legacy_to_dynamic_cache(legacy_kv)

            codec_time_ms = (time.perf_counter() - codec_t0) * 1000
            total_codec_time_ms += codec_time_ms

            kv_seq_len = _get_past_length(past_kv)
            agent_time_ms = (time.perf_counter() - agent_t0) * 1000

            agent_traces.append({
                "name": agent.name,
                "role": agent.role,
                "prompt_tokens": prompt_tokens,
                "latent_steps": latent_steps,
                "summary_tokens": summary_tokens,
                "kv_seq_len_after": kv_seq_len,
                "wire_bytes": wire_size,
                "codec_time_ms": codec_time_ms,
                "summary_time_ms": summary_time_ms,
                "agent_time_ms": agent_time_ms,
                "text_fallback": avp_msg.text_fallback,
                "output": "",
            })

            if verbose:
                summary_preview = avp_msg.text_fallback[:80] if avp_msg.text_fallback else ""
                print(f"  [{agent.name}] latent steps={latent_steps}, "
                      f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                      f"codec={codec_time_ms:.1f}ms, "
                      f"summary={summary_time_ms:.0f}ms")
                print(f"    text_fallback: {summary_preview}...")

        else:
            # --- Judger: generate text with accumulated KV-cache ---
            kv_seq_len = _get_past_length(past_kv)

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
                "name": agent.name,
                "role": agent.role,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "kv_seq_len_input": kv_seq_len,
                "agent_time_ms": agent_time_ms,
                "output": text,
            })

            if verbose:
                print(f"  [{agent.name}] KV input seq_len={kv_seq_len}, "
                      f"output ({len(text)} chars): {text[:200]}...")

    wall_time = time.perf_counter() - t0
    total_tokens = total_prompt_tokens + total_latent_steps + total_summary_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    peak_memory_mb = None
    if device == "cuda":
        peak_memory_mb = (torch.cuda.max_memory_allocated() - mem_before) / (1024 * 1024)

    gold = extract_gold(gold_solution)
    prediction = extract_gsm8k_answer(agent_traces[-1]["output"])

    from .evaluate import check_correct
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
        "total_summary_tokens": total_summary_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": peak_memory_mb,
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
        "kv_seq_len_judger": _get_past_length(past_kv) if past_kv else 0,
        "agents": agent_traces,
        "mode": "hybrid",
    }


def run_hybrid_benchmark(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    dataset: List[Dict],
    model_name: str,
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    summary_max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run hybrid-mode pipeline on a list of GSM8K samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Hybrid] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_hybrid_pipeline(
            connector, model, tokenizer, device, identity,
            question=sample["question"],
            gold_solution=sample["answer"],
            model_name=model_name,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            summary_max_tokens=summary_max_tokens,
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
                  f"wire={result['avp_wire_bytes']:,} bytes)")

    return results
