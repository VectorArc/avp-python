"""Agent definitions and prompt builders for Debate benchmark.

Topology:
    3 agents x N rounds --> majority vote

Three agents with distinct perspectives debate each question:
1. Analyst — Breaks down the question into verifiable facts
2. Skeptic — Challenges assumptions and looks for edge cases
3. Synthesizer — Connects facts together based on common knowledge

All agents generate text every round (for yes/no extraction).
In latent mode, KV-cache accumulates across all agents and rounds.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Agent:
    """A single agent in the debate."""
    name: str
    role: str
    perspective: str


AGENTS = [
    Agent(
        name="Analyst",
        role="analyst",
        perspective="Focus on breaking down the question into verifiable facts.",
    ),
    Agent(
        name="Skeptic",
        role="skeptic",
        perspective="Challenge assumptions and look for edge cases.",
    ),
    Agent(
        name="Synthesizer",
        role="synthesizer",
        perspective="Connect facts together based on common knowledge.",
    ),
]

SYSTEM_MESSAGE = "You are a helpful assistant participating in a structured debate."

DIRECT_SOLVE_PROMPT = (
    "Answer the following yes/no question. Think step by step, "
    "then give your final answer as either Yes or No.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_latent_prompt(
    role: str,
    perspective: str,
    question: str,
    round_num: int,
    num_rounds: int,
) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior debate history is carried via KV-cache.
    The prompt is minimal — just role + question + round info.
    """
    if round_num == 0:
        round_context = "This is the opening round."
    else:
        round_context = (
            f"This is round {round_num + 1} of {num_rounds}. "
            f"Previous debate context is available in latent KV representation."
        )

    user_prompt = (
        f"You are a {role.title()} in a structured debate. "
        f"Your perspective: {perspective}\n\n"
        f"{round_context}\n\n"
        f"Question: {question}\n\n"
        f"State your position clearly. End with your answer: Yes or No."
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_prompt(
    role: str,
    perspective: str,
    question: str,
    transcript: str,
    round_num: int,
    num_rounds: int,
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, the full debate transcript is included in the prompt.
    """
    if round_num == 0 and not transcript:
        context_section = "This is the opening round — no prior debate."
    else:
        context_section = (
            f"## Debate So Far (Round {round_num + 1} of {num_rounds}):\n"
            f"{transcript}"
        )

    user_content = (
        f"You are a {role.title()} in a structured debate. "
        f"Your perspective: {perspective}\n\n"
        f"Question: {question}\n\n"
        f"{context_section}\n\n"
        f"Consider the prior arguments and state your position. "
        f"You may update your view based on compelling arguments. "
        f"End with your answer: Yes or No."
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def extract_yes_no(text: str) -> Optional[str]:
    """Extract Yes/No answer from agent's output.

    Priority:
    1. Last occurrence of standalone "Yes" or "No" (case-insensitive)
    2. "answer is Yes/No" pattern
    3. First occurrence of Yes/No
    """
    text_stripped = text.strip()

    # 1. Check last line for standalone Yes/No
    last_line = text_stripped.split("\n")[-1].strip().rstrip(".")
    if last_line.lower() in ("yes", "no"):
        return last_line.capitalize()

    # 2. "answer is Yes/No" pattern
    match = re.findall(
        r"(?:my\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(yes|no)\b",
        text, re.IGNORECASE,
    )
    if match:
        return match[-1].capitalize()

    # 3. Find all standalone Yes/No occurrences, take last
    matches = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()

    return None


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Compute majority vote from a list of Yes/No answers."""
    valid = [a for a in answers if a in ("Yes", "No")]
    if not valid:
        return None
    yes_count = sum(1 for a in valid if a == "Yes")
    no_count = sum(1 for a in valid if a == "No")
    if yes_count > no_count:
        return "Yes"
    elif no_count > yes_count:
        return "No"
    # Tie — no majority
    return None
