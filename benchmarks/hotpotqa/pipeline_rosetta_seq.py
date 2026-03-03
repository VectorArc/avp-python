"""Rosetta Sequence pipeline for HotpotQA: discrete token transfer for cross-model Finder → Answerer.

Finder runs on model A (reads paragraphs → forward pass → lm_head argmax → decode to text).
Answerer runs on model B (receives decoded tokens as regular text context).

No continuous embeddings, no inputs_embeds, no KV-cache priming, no calibration.
Completely on-manifold by construction — decoded tokens are real vocabulary items.
"""

import time
from typing import Dict, List

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS_2, build_latent_prompt, build_text_prompt, format_paragraphs
from .evaluate import exact_match, extract_answer, token_f1


def _select_positions(
    hidden_states: torch.Tensor,
    num_positions: int,
    strategy: str,
) -> tuple:
    """Select positions from hidden states tensor [seq_len, D].

    Returns:
        (selected_hidden, selected_indices) where selected_hidden is [N, D]
        and selected_indices is a list of ints.
    """
    seq_len = hidden_states.shape[0]
    num_positions = min(num_positions, seq_len)

    if strategy == "stride":
        indices = torch.linspace(0, seq_len - 1, num_positions).long().tolist()
    else:
        # last_n: take the last N positions (highest context density)
        indices = list(range(seq_len - num_positions, seq_len))

    selected = hidden_states[indices, :]  # [N, D]
    return selected, indices


def run_rosetta_seq_pipeline(
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    device: str,
    question: str,
    gold_answer: str,
    paragraphs: List[Dict],
    num_positions: int = 16,
    position_strategy: str = "last_n",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run the 2-agent rosetta sequence pipeline on a single HotpotQA problem.

    Agent 1 (Finder on model A): forward pass on paragraphs + question →
        extract hidden states → lm_head argmax → decode to text tokens.
    Agent 2 (Answerer on model B): receives decoded tokens as text context →
        standard text generation.
    """
    paragraphs_text = format_paragraphs(paragraphs)

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        finder = AGENTS_2[0]
        answerer = AGENTS_2[1]

        # --- Agent 1: Finder on model A (forward pass → decode to tokens) ---
        messages = build_latent_prompt(finder.role, question, paragraphs_text=paragraphs_text)
        prompt_text = render_prompt(tokenizer_a, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_a, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        # Forward pass on the prompt — no generation, no latent steps
        with torch.no_grad():
            out = model_a(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last-layer hidden states: [1, seq_len, D]
        all_hidden = out.hidden_states[-1]
        source_seq_len = all_hidden.shape[1]

        # Select N positions
        selected, selected_positions = _select_positions(
            all_hidden[0], num_positions, position_strategy,
        )

        # Decode to discrete tokens via lm_head → argmax
        decode_t0 = time.perf_counter()
        lm_head = model_a.get_output_embeddings()
        logits = lm_head(selected.to(lm_head.weight.dtype))  # [N, V_src]
        token_ids = logits.argmax(dim=-1)  # [N]

        # Decode token IDs to text string
        decoded_text = tokenizer_a.decode(token_ids.tolist(), skip_special_tokens=True)
        decode_ms = (time.perf_counter() - decode_t0) * 1000

        # Wire size is the UTF-8 byte count of the decoded text
        wire_bytes = len(decoded_text.encode("utf-8"))
        agent_time_ms = (time.perf_counter() - agent_t0) * 1000

        agent_traces.append({
            "name": finder.name,
            "role": finder.role,
            "prompt_tokens": prompt_tokens,
            "decode_ms": decode_ms,
            "wire_bytes": wire_bytes,
            "agent_time_ms": agent_time_ms,
            "output": "",
            "decoded_tokens": decoded_text,
            "num_positions": len(selected_positions),
            "position_strategy": position_strategy,
            "source_seq_len": source_seq_len,
            "selected_positions": selected_positions,
        })

        if verbose:
            print(f"  [{finder.name}] positions={position_strategy}, "
                  f"N={len(selected_positions)}, seq_len={source_seq_len}, "
                  f"decode={decode_ms:.1f}ms, "
                  f"wire={wire_bytes} bytes, "
                  f"decoded: '{decoded_text[:100]}...'")

        # Free model A state
        del out, all_hidden, selected, logits
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Agent 2: Answerer on model B (text prompt with decoded tokens as context) ---
        messages = build_text_prompt(answerer.role, question, context=decoded_text)
        prompt_text = render_prompt(tokenizer_b, messages)
        input_ids, attention_mask = tokenize_prompt(tokenizer_b, prompt_text, device)

        agent_t0 = time.perf_counter()
        prompt_tokens = int(input_ids.shape[-1])
        total_prompt_tokens += prompt_tokens

        # Standard text generation — no inputs_embeds, no KV-cache priming
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

    total_tokens = total_prompt_tokens + total_output_tokens
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
        "total_latent_steps": 0,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "decode_overhead_ms": decode_ms,
        "wire_bytes": wire_bytes,
        "decoded_tokens": decoded_text,
        "num_positions": num_positions,
        "position_strategy": position_strategy,
        "source_seq_len": source_seq_len,
        "agents": agent_traces,
        "mode": "rosetta_seq",
    }


def run_rosetta_seq_benchmark(
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    device: str,
    dataset: List[Dict],
    num_positions: int = 16,
    position_strategy: str = "last_n",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run rosetta sequence pipeline on HotpotQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[RosettaSeq] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_rosetta_seq_pipeline(
            model_a, tokenizer_a,
            model_b, tokenizer_b, device,
            question=sample["question"],
            gold_answer=sample["answer"],
            paragraphs=sample["paragraphs"],
            num_positions=num_positions,
            position_strategy=position_strategy,
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
                  f"wire={result['wire_bytes']} bytes)")
        else:
            correct = sum(1 for r in results if r["exact_match"])
            f1s = [r["f1"] for r in results]
            mean_f1 = sum(f1s) / len(f1s)
            print(f"  [RosettaSeq] {i + 1}/{len(dataset)} "
                  f"(EM={correct}/{i + 1}, F1={mean_f1:.2f}, {result['wall_time']:.1f}s)",
                  flush=True)

    return results
