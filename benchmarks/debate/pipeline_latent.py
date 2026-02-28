"""Latent-mode pipeline for Debate benchmark: KV-cache accumulates across rounds.

Topology:
    3 agents x N rounds --> majority vote

All agents generate TEXT every round (needed for yes/no extraction).
The KV-cache from generate_text() accumulates across all agents and rounds,
carrying the full debate history in latent form.

Token savings: In text mode, by round 3 each agent re-processes ~6 prior
text outputs. In latent mode, this is all in KV-cache — zero re-processing.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import torch

from benchmarks.shared.avp_roundtrip import avp_kv_roundtrip
from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.kv_utils import get_past_length
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_latent_prompt, extract_yes_no, majority_vote
from .evaluate import check_correct


def run_latent_pipeline(
    connector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    identity: Any,
    question: str,
    gold_answer: bool,
    model_name: str,
    num_rounds: int = 3,
    num_agents: int = 3,
    latent_steps: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run multi-round debate with KV-cache accumulation.

    Every agent generates text (for vote extraction). The KV-cache returned
    by generate_text() is passed through AVP codec and forwarded to the next
    agent, accumulating the full debate history.
    """
    agents = AGENTS[:num_agents]

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        past_kv = None
        agent_traces: List[Dict] = []
        total_codec_time_ms = 0.0
        total_wire_bytes = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        round_votes: Dict[str, Optional[str]] = {}
        prev_answers: Dict[str, Optional[str]] = {}
        total_flips = 0

        for round_num in range(num_rounds):
            round_answers = []

            for agent_idx, agent in enumerate(agents):
                agent_t0 = time.perf_counter()

                # Determine next agent for AVP metadata
                if agent_idx + 1 < len(agents):
                    next_name = agents[agent_idx + 1].name
                elif round_num + 1 < num_rounds:
                    next_name = agents[0].name  # First agent of next round
                else:
                    next_name = "final"

                messages = build_latent_prompt(
                    agent.role, agent.perspective, question,
                    round_num, num_rounds,
                )
                prompt_text = render_prompt(tokenizer, messages)
                input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
                prompt_tokens = int(input_ids.shape[-1])
                total_prompt_tokens += prompt_tokens

                kv_seq_len_input = get_past_length(past_kv)

                # Generate text WITH KV-cache context
                text, past_kv = generate_text(
                    model, tokenizer, input_ids, attention_mask, device,
                    past_key_values=past_kv,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                output_encoded = tokenizer(text, add_special_tokens=False)
                output_tokens = len(output_encoded["input_ids"])
                total_output_tokens += output_tokens

                # AVP codec roundtrip (skip for very last agent of very last round)
                is_last = (round_num == num_rounds - 1 and agent_idx == len(agents) - 1)
                if past_kv is not None and not is_last:
                    past_kv, codec_time_ms, wire_size = avp_kv_roundtrip(
                        past_kv, session_id, agent.name, next_name,
                        model_name, identity, device,
                    )
                    total_codec_time_ms += codec_time_ms
                    total_wire_bytes += wire_size
                else:
                    codec_time_ms = 0.0
                    wire_size = 0

                kv_seq_len_after = get_past_length(past_kv)
                agent_time_ms = (time.perf_counter() - agent_t0) * 1000

                answer = extract_yes_no(text)
                round_answers.append(answer)

                # Track flips
                if agent.name in prev_answers and prev_answers[agent.name] != answer:
                    total_flips += 1
                prev_answers[agent.name] = answer

                agent_traces.append({
                    "name": agent.name,
                    "role": agent.role,
                    "round_num": round_num,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "kv_seq_len_input": kv_seq_len_input,
                    "kv_seq_len_after": kv_seq_len_after,
                    "wire_bytes": wire_size,
                    "codec_time_ms": codec_time_ms,
                    "agent_time_ms": agent_time_ms,
                    "output": text,
                    "answer": answer,
                })

                if verbose:
                    print(f"  [R{round_num + 1} {agent.name}] answer={answer}, "
                          f"KV in={kv_seq_len_input} out={kv_seq_len_after}, "
                          f"wire={wire_size:,} bytes, "
                          f"output ({len(text)} chars): {text[:100]}...")

                if device == "cuda":
                    torch.cuda.empty_cache()

            vote = majority_vote(round_answers)
            round_votes[str(round_num)] = vote

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Final prediction
    final_round_answers = [
        a["answer"] for a in agent_traces
        if a["round_num"] == num_rounds - 1
    ]
    prediction = majority_vote(final_round_answers)
    correct = check_correct(prediction, gold_answer)

    # Consensus
    valid_final = [a for a in final_round_answers if a is not None]
    consensus = len(set(valid_final)) == 1 if valid_final else False

    # Rounds to consensus
    rounds_to_consensus = None
    for rnd in range(num_rounds):
        rnd_answers = [
            a["answer"] for a in agent_traces if a["round_num"] == rnd
        ]
        valid_rnd = [a for a in rnd_answers if a is not None]
        if valid_rnd and len(set(valid_rnd)) == 1:
            rounds_to_consensus = rnd + 1
            break

    return {
        "question": question,
        "gold": gold_answer,
        "prediction": prediction,
        "raw_output": agent_traces[-1]["output"] if agent_traces else "",
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
        "round_votes": round_votes,
        "consensus": consensus,
        "rounds_to_consensus": rounds_to_consensus,
        "total_flips": total_flips,
        "num_rounds": num_rounds,
        "num_agents": num_agents,
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
    num_rounds: int = 3,
    num_agents: int = 3,
    latent_steps: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run latent-mode debate pipeline on StrategyQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Latent] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_latent_pipeline(
            connector, model, tokenizer, device, identity,
            question=sample["question"],
            gold_answer=sample["answer"],
            model_name=model_name,
            num_rounds=num_rounds,
            num_agents=num_agents,
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
                  f"consensus={result['consensus']}, flips={result['total_flips']}, "
                  f"time={result['wall_time']:.1f}s, "
                  f"codec={result['codec_overhead_ms']:.1f}ms)")

    return results
