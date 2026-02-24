"""Agent definitions and prompt builders for the 2-agent GSM8K benchmark.

Two agents:
1. Researcher — Analyzes the problem, identifies approach, shows reasoning
2. Solver — Receives Researcher's work (KV-cache or text), computes final answer
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the 2-agent chain."""
    name: str
    role: str


AGENTS = [
    Agent(name="Researcher", role="researcher"),
    Agent(name="Solver", role="solver"),
]

SYSTEM_MESSAGE = "You are a helpful math assistant."

# Direct single-agent prompt (reused from original GSM8K benchmark)
DIRECT_SOLVE_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your work clearly, then give the final numeric answer "
    "inside \\boxed{{}}.\n\n"
    "Question: {question}\n\n"
    "Solution:"
)


def build_latent_prompt(role: str, question: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache — the prompt only
    contains the agent's role instruction and the question.
    """
    if role == "researcher":
        user_prompt = (
            f"You are a Researcher Agent. Analyze the following math problem. "
            f"Identify the key quantities, relationships, and the approach "
            f"needed to solve it. Show your intermediate reasoning and calculations.\n\n"
            f"Question: {question}\n\n"
            f"Analyze the problem and show your reasoning:"
        )
    elif role == "solver":
        user_prompt = (
            f"You are a Solver Agent. You have received analysis and reasoning "
            f"from a Researcher who already examined this problem. "
            f"Use their work to compute the final answer.\n\n"
            f"Question: {question}\n\n"
            f"Based on the research provided, compute the final numeric answer. "
            f"Show your calculation steps and output the answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}. The answer must be a number."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_prompt(
    role: str, question: str, context: str = ""
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, the Researcher's text output is included in the Solver's prompt.
    """
    if role == "researcher":
        user_content = (
            f"You are a Researcher Agent. Analyze the following math problem. "
            f"Identify the key quantities, relationships, and the approach "
            f"needed to solve it. Show your intermediate reasoning and calculations.\n\n"
            f"## Question:\n{question}\n\n"
            f"## Format your response as follows:\n"
            f"Researcher's Analysis:\n"
            f"[Your detailed analysis and intermediate calculations here]\n\n"
            f"Now output your analysis:"
        )
    elif role == "solver":
        user_content = (
            f"You are a Solver Agent. You are provided with:\n"
            f"(1) the original question, and\n"
            f"(2) the Researcher's analysis in text format.\n\n"
            f"## Question:\n{question}\n\n"
            f"## Researcher's Analysis:\n{context}\n\n"
            f"Using the Researcher's analysis, compute the final answer. "
            f"Show your calculation steps and output the final numeric answer "
            f"inside \\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
