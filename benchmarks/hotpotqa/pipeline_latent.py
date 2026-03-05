"""Latent-mode pipeline for HotpotQA: Finder → Answerer via KV-cache.

Key insight: The Answerer's prompt omits the context paragraphs entirely.
The Finder's KV-cache carries the model's internal understanding of which
paragraphs matter. This is the ideal latent scenario — transferring
comprehension, not text.

Supports 2-agent (Finder → Answerer) and 3-agent (Decomposer → Finder → Answerer).
"""

import time
import uuid
from typing import Any, Dict, List

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS_2, AGENTS_3, build_latent_prompt, format_paragraphs
from .evaluate import check_correct, exact_match, extract_answer, token_f1


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    question: str,
    gold_answer: str,
    paragraphs: List[Dict],
    model_name: str,
    num_agents: int = 2,
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run latent-mode pipeline on a single HotpotQA problem."""
    agents = AGENTS_2 if num_agents == 2 else AGENTS_3
    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        past_kv = None
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        for idx, agent in enumerate(agents):
            next_agent = agents[idx + 1] if idx + 1 < len(agents) else None
            is_last = next_agent is None

            # Build prompt — Finder gets paragraphs, others don't in latent mode
            needs_paragraphs = agent.role in ("finder", "finder_3agent")
            messages = build_latent_prompt(
                agent.role, question,
                paragraphs_text=paragraphs_text if needs_paragraphs else "",
            )
            prompt_text = render_prompt(tokenizer, messages)
            input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)

            agent_t0 = time.perf_counter()
            prompt_tokens = int(input_ids.shape[-1])
            total_prompt_tokens += prompt_tokens

            if not is_last:
                # Non-terminal agent: generate latent steps
                total_latent_steps += latent_steps
                past_kv = connector.generate_latent_steps(
                    input_ids,
                    latent_steps=latent_steps,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                )

                # AVP codec roundtrip
                past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
                    past_kv, session_id, agent.name,
                    next_agent.name if next_agent else "",
                    model_name, identity, device,
                )
                total_codec_time_ms += codec_time_ms
                total_wire_bytes += wire_size

                kv_seq_len = get_past_length(past_kv)
                agent_time_ms = (time.perf_counter() - agent_t0) * 1000

                agent_traces.append({
                    "name": agent.name,
                    "role": agent.role,
                    "prompt_tokens": prompt_tokens,
                    "latent_steps": latent_steps,
                    "kv_seq_len_after": kv_seq_len,
                    "wire_bytes": wire_size,
                    "codec_time_ms": codec_time_ms,
                    "agent_time_ms": agent_time_ms,
                    "output": "",
                })

                if verbose:
                    print(f"  [{agent.name}] latent steps={latent_steps}, "
                          f"KV seq_len={kv_seq_len}, wire={wire_size:,} bytes, "
                          f"codec={codec_time_ms:.1f}ms")
            else:
                # Terminal agent: generate text answer
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

    total_tokens = total_prompt_tokens + total_latent_steps + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    prediction = extract_answer(agent_traces[-1]["output"])
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"],
        "correct": em,
        "exact_match": em,
        "f1": f1,
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
    num_agents: int = 2,
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            model_name=model_name,
            num_agents=num_agents,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["exact_match"] else "WRONG"
            print(f"  => {status} (pred='{result['prediction']}', "
                  f"gold='{result['gold']}', F1={result['f1']:.2f}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"codec={result['codec_overhead_ms']:.1f}ms)")
        else:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [Latent] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
