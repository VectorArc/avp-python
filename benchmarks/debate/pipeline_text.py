"""Text-mode pipeline for Debate benchmark: agents pass text transcript."""

import time
from typing import Any, Dict, List, Optional

import torch

from benchmarks.shared.generation import generate_text, render_prompt, tokenize_prompt
from benchmarks.shared.metrics import gpu_memory_tracker
from .agents import AGENTS, build_text_prompt, extract_yes_no, majority_vote
from .evaluate import check_correct


def run_text_pipeline(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
    gold_answer: bool,
    num_rounds: int = 3,
    num_agents: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> Dict:
    """Run multi-round debate with text context passing.

    Each agent reads the full transcript of all prior arguments.
    Token count grows linearly per round (the "communication tax").
    """
    agents = AGENTS[:num_agents]

    with gpu_memory_tracker(device) as mem:
        t0 = time.perf_counter()
        transcript = ""
        agent_traces: List[Dict] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_context_tokens = 0
        round_votes: Dict[str, Optional[str]] = {}
        prev_answers: Dict[str, Optional[str]] = {}
        total_flips = 0

        for round_num in range(num_rounds):
            round_answers = []

            for agent in agents:
                agent_t0 = time.perf_counter()

                # Count context tokens
                if transcript:
                    context_encoded = tokenizer(transcript, add_special_tokens=False)
                    context_token_count = len(context_encoded["input_ids"])
                else:
                    context_token_count = 0
                total_context_tokens += context_token_count

                messages = build_text_prompt(
                    agent.role, agent.perspective, question,
                    transcript, round_num, num_rounds,
                )
                prompt_text = render_prompt(tokenizer, messages)
                input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_text, device)
                prompt_tokens = int(input_ids.shape[-1])
                total_prompt_tokens += prompt_tokens

                text, _ = generate_text(
                    model, tokenizer, input_ids, attention_mask, device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                transcript += (
                    f"[Round {round_num + 1}, {agent.name}]: {text}\n\n"
                )

                output_encoded = tokenizer(text, add_special_tokens=False)
                output_tokens = len(output_encoded["input_ids"])
                total_output_tokens += output_tokens
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
                    "context_tokens": context_token_count,
                    "agent_time_ms": agent_time_ms,
                    "output": text,
                    "answer": answer,
                })

                if verbose:
                    print(f"  [R{round_num + 1} {agent.name}] answer={answer}, "
                          f"output ({len(text)} chars): {text[:100]}...")

            # Majority vote for this round
            vote = majority_vote(round_answers)
            round_votes[str(round_num)] = vote

        wall_time = time.perf_counter() - t0

    total_tokens = total_prompt_tokens + total_output_tokens
    tokens_per_sec = total_tokens / wall_time if wall_time > 0 else 0

    # Final prediction is majority vote of the last round
    final_round_answers = [
        a["answer"] for a in agent_traces
        if a["round_num"] == num_rounds - 1
    ]
    prediction = majority_vote(final_round_answers)
    correct = check_correct(prediction, gold_answer)

    # Consensus: all agents agree in last round
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
        "raw_output": transcript,
        "correct": correct,
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_context_tokens": total_context_tokens,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": mem["peak_memory_mb"],
        "agents": agent_traces,
        "round_votes": round_votes,
        "consensus": consensus,
        "rounds_to_consensus": rounds_to_consensus,
        "total_flips": total_flips,
        "num_rounds": num_rounds,
        "num_agents": num_agents,
        "mode": "text",
    }


def run_text_benchmark(
    model: Any,
    tokenizer: Any,
    device: str,
    dataset: List[Dict],
    num_rounds: int = 3,
    num_agents: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    verbose: bool = False,
) -> List[Dict]:
    """Run text-mode debate pipeline on StrategyQA samples."""
    results = []
    for i, sample in enumerate(dataset):
        if verbose:
            print(f"\n[Text] Sample {i + 1}/{len(dataset)}: {sample['question'][:80]}...")

        result = run_text_pipeline(
            model, tokenizer, device,
            question=sample["question"],
            gold_answer=sample["answer"],
            num_rounds=num_rounds,
            num_agents=num_agents,
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
                  f"time={result['wall_time']:.1f}s)")

    return results
