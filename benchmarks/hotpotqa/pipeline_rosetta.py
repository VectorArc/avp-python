"""Rosetta-mode pipeline for HotpotQA: cross-model Finder → Answerer.

Finder runs on model A (reads paragraphs → latent steps → extract hidden state → project).
Answerer runs on model B (inject projected embedding → generate answer).

Only supports 2-agent mode (Finder → Answerer). 3-agent rosetta not implemented.
"""

import time
from typing import Any, Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS_2, build_latent_prompt, format_paragraphs
from .evaluate import exact_match, extract_answer, token_f1


def run_rosetta_pipeline(
    conn_a: Any,
    model_a: Any,
    tokenizer_a: Any,
    identity_a: Any,
    model_b: Any,
    tokenizer_b: Any,
    device: str,
    avp_map: Any,
    question: str,
    gold_answer: str,
    paragraphs: List[Dict],
    latent_steps: int = 10,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
) -> Dict:
    """Run the 2-agent cross-model pipeline on a single HotpotQA problem."""
    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_latent_steps = 0
        total_output_tokens = 0

        finder = AGENTS_2[0]
        answerer = AGENTS_2[1]

        # --- Agent 1: Finder on model A (latent steps) ---
        messages = build_latent_prompt(finder.role, question, paragraphs_text=paragraphs_text)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens
        total_latent_steps += latent_steps

        past_kv = conn_a.generate_latent_steps(
            input_ids, latent_steps=latent_steps, attention_mask=attention_mask,
        )

        # Extract last hidden state from source model
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
        last_hidden = out.hidden_states[-1][:, -1, :]  # [1, D_src]

        # Project to target model space
        projected = conn_a.project_hidden_for_cross_model(
            last_hidden, avp_map, temperature=projection_temperature,
        )
        rosetta_embeds = projected.unsqueeze(1)  # [1, 1, D_tgt]
        projection_ms = (time.perf_counter() - proj_t0) * 1000

        wire_bytes = rosetta_embeds.nelement() * rosetta_embeds.element_size()
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": finder.name,
            "role": finder.role,
            "prompt_tokens": prompt_tokens,
            "latent_steps": latent_steps,
            "projection_ms": projection_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
        })

        if verbose:
            print(f"  [{finder.name}] latent steps={latent_steps}, "
                  f"projection={projection_ms:.1f}ms, "
                  f"projected shape={list(rosetta_embeds.shape)}, "
                  f"wire={wire_bytes:,} bytes")

        # Free model A KV-cache
        del past_kv, out
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Answerer on model B (inject projected embed, generate) ---
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

        messages = build_latent_prompt(answerer.role, question)
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
            "name": answerer.name,
            "role": answerer.role,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "agent_time_ms": agent_time_ms,
            "output": text,
        })

        if verbose:
            print(f"  [{answerer.name}] output ({len(text)} chars): {text[:200]}...")

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
        "projection_overhead_ms": projection_ms,
        "projection_wire_bytes": wire_bytes,
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
    projection_temperature: float = 1.0,
) -> List[Dict]:
    """Run rosetta-mode pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Rosetta] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_rosetta_pipeline(
            conn_a, model_a, tokenizer_a, identity_a,
            model_b, tokenizer_b, device, avp_map,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            projection_temperature=projection_temperature,
        )
        results.append(result)

        if verbose:
            status = "CORRECT" if result["exact_match"] else "WRONG"
            print(f"  => {status} (pred='{result['prediction']}', "
                  f"gold='{result['gold']}', F1={result['f1']:.2f}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"projection={result['projection_overhead_ms']:.1f}ms)")
        else:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [Rosetta] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
