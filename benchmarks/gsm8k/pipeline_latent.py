"""Latent-mode pipeline: agents communicate via KV-cache transferred through AVP codec."""

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


def _get_past_length(past_kv: Any) -> int:
    """Get sequence length from past_key_values (DynamicCache or legacy tuple)."""
    if past_kv is None:
        return 0
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            return past_kv.get_seq_length()
    except ImportError:
        pass
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (tuple, list)) and len(first) > 0:
            return first[0].shape[-2]
    return 0


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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 4-agent latent-mode pipeline on a single GSM8K problem.

    Non-judger agents use generate_latent_steps to produce KV-cache, which is
    serialized through the full AVP codec (serialize → encode → decode → deserialize)
    between each hop. The Judger generates text with the accumulated KV-cache.
    """
    from avp.codec import decode as avp_decode
    from avp.codec import encode_kv_cache
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

    t0 = time.perf_counter()
    session_id = str(uuid.uuid4())
    past_kv = None
    agent_traces: List[Dict] = []
    total_codec_time_ms = 0.0
    total_wire_bytes = 0

    for idx, agent in enumerate(AGENTS):
        next_agent = AGENTS[idx + 1] if idx + 1 < len(AGENTS) else None

        # Build and tokenize prompt
        messages = build_latent_prompt(agent.role, question)
        prompt_text = render_prompt(tokenizer, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

        if agent.role != "judger":
            # --- Latent steps: generate KV-cache ---
            past_kv = connector.generate_latent_steps(
                input_ids,
                latent_steps=latent_steps,
                attention_mask=attention_mask,
                past_key_values=past_kv,
            )

            # --- AVP codec roundtrip: serialize → encode → decode → deserialize ---
            codec_t0 = time.perf_counter()

            kv_bytes, kv_header = serialize_kv_cache(past_kv)

            metadata = AVPMetadata(
                session_id=session_id,
                source_agent_id=agent.name,
                target_agent_id=next_agent.name if next_agent else "",
                model_id=model_name,
                hidden_dim=identity.hidden_dim,
                num_layers=identity.num_layers,
                payload_type=PayloadType.KV_CACHE,
                dtype=DataType.FLOAT32,
                mode=CommunicationMode.LATENT,
            )

            wire_bytes = encode_kv_cache(kv_bytes, metadata)
            wire_size = len(wire_bytes)
            total_wire_bytes += wire_size

            # Decode on "receiving" side
            avp_msg = avp_decode(wire_bytes)
            legacy_kv, _ = deserialize_kv_cache(avp_msg.payload, device=device)
            past_kv = legacy_to_dynamic_cache(legacy_kv)

            codec_time_ms = (time.perf_counter() - codec_t0) * 1000
            total_codec_time_ms += codec_time_ms

            kv_seq_len = _get_past_length(past_kv)

            agent_traces.append({
                "name": agent.name,
                "role": agent.role,
                "prompt_tokens": int(input_ids.shape[-1]),
                "latent_steps": latent_steps,
                "kv_seq_len_after": kv_seq_len,
                "wire_bytes": wire_size,
                "codec_time_ms": codec_time_ms,
                "output": "",
            })

            if verbose:
                print(f"  [{agent.name}] latent steps={latent_steps}, "
                      f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                      f"codec={codec_time_ms:.1f}ms")

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

            agent_traces.append({
                "name": agent.name,
                "role": agent.role,
                "prompt_tokens": int(input_ids.shape[-1]),
                "kv_seq_len_input": kv_seq_len,
                "output": text,
            })

            if verbose:
                print(f"  [{agent.name}] KV input seq_len={kv_seq_len}, "
                      f"output ({len(text)} chars): {text[:200]}...")

    wall_time = time.perf_counter() - t0
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
        "codec_overhead_ms": total_codec_time_ms,
        "avp_wire_bytes": total_wire_bytes,
        "kv_seq_len_judger": _get_past_length(past_kv) if past_kv else 0,
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode pipeline on a list of GSM8K samples."""
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
                  f"wire={result['avp_wire_bytes']:,} bytes)")

    return results
