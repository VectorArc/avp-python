"""StrategyQA answer evaluation for Debate benchmark."""

from typing import Dict, List, Optional


def check_correct(prediction_str: Optional[str], gold_bool: bool) -> bool:
    """Check if predicted Yes/No matches gold boolean."""
    if prediction_str is None:
        return False
    pred_bool = prediction_str.lower() == "yes"
    return pred_bool == gold_bool


def compute_accuracy(results: List[Dict]) -> Dict:
    """Compute accuracy with debate-specific metrics."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))

    # Consensus rate: how often all agents agreed in the final round
    consensus_count = sum(1 for r in results if r.get("consensus", False))

    # Mean rounds to consensus (if tracked)
    rounds_to_consensus = [
        r["rounds_to_consensus"] for r in results
        if r.get("rounds_to_consensus") is not None
    ]
    mean_rtc = (
        sum(rounds_to_consensus) / len(rounds_to_consensus)
        if rounds_to_consensus else None
    )

    # Answer flips: how many times agents changed their answer across rounds
    flips = [r.get("total_flips", 0) for r in results]
    mean_flips = sum(flips) / len(flips) if flips else 0.0

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "consensus_rate": consensus_count / total if total > 0 else 0.0,
        "mean_rounds_to_consensus": mean_rtc,
        "mean_flips_per_sample": mean_flips,
    }


def print_debate_summary(results: List[Dict], num_rounds: int) -> None:
    """Print debate-specific metrics."""
    acc = compute_accuracy(results)

    print(f"\n  Consensus rate: {acc['consensus_rate']:.1%}")
    if acc['mean_rounds_to_consensus'] is not None:
        print(f"  Mean rounds to consensus: {acc['mean_rounds_to_consensus']:.1f}")
    print(f"  Mean answer flips per sample: {acc['mean_flips_per_sample']:.1f}")

    # Per-round accuracy (if available)
    per_round = {}
    for r in results:
        round_votes = r.get("round_votes", {})
        for rnd_str, vote in round_votes.items():
            rnd = int(rnd_str)
            if rnd not in per_round:
                per_round[rnd] = {"correct": 0, "total": 0}
            per_round[rnd]["total"] += 1
            gold = r.get("gold")
            if vote is not None and check_correct(vote, gold):
                per_round[rnd]["correct"] += 1

    if per_round:
        print(f"\n  Per-Round Majority Vote Accuracy:")
        print(f"  {'Round':<8} {'Accuracy':>10} {'Correct':>8} {'Total':>6}")
        print(f"  {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 6}")
        for rnd in sorted(per_round.keys()):
            d = per_round[rnd]
            acc_val = d["correct"] / d["total"] if d["total"] > 0 else 0.0
            print(f"  {rnd + 1:<8} {acc_val:>9.1%} {d['correct']:>8} {d['total']:>6}")

    # Token growth per round (text mode)
    per_round_tokens = {}
    for r in results:
        agents = r.get("agents", [])
        for a in agents:
            rnd = a.get("round_num")
            if rnd is not None and "prompt_tokens" in a:
                if rnd not in per_round_tokens:
                    per_round_tokens[rnd] = []
                per_round_tokens[rnd].append(a["prompt_tokens"])

    if per_round_tokens:
        print(f"\n  Mean Prompt Tokens per Agent by Round:")
        print(f"  {'Round':<8} {'Tokens':>10}")
        print(f"  {'-' * 8} {'-' * 10}")
        for rnd in sorted(per_round_tokens.keys()):
            tokens = per_round_tokens[rnd]
            mean_tok = sum(tokens) / len(tokens) if tokens else 0
            print(f"  {rnd + 1:<8} {mean_tok:>10,.0f}")
