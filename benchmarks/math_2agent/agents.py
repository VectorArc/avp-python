"""Agent definitions and prompt builders for the 2-agent MATH benchmark.

Two agents:
1. Researcher — Analyzes the competition math problem, identifies concepts and approach
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

# Direct single-agent prompt
DIRECT_SOLVE_PROMPT = (
    "Solve the following competition math problem step by step. "
    "Show your work clearly, then give the final answer "
    "inside \\boxed{{}}. The answer may be a number, fraction, "
    "expression, or mathematical object.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


def build_latent_prompt(role: str, problem: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache — the prompt only
    contains the agent's role instruction and the problem.
    """
    if role == "researcher":
        user_prompt = (
            f"You are a Researcher Agent. Analyze the following competition math problem. "
            f"Identify the relevant mathematical concepts, plan the approach, "
            f"and show your intermediate reasoning. The answer may be a number, "
            f"fraction, expression, or mathematical object.\n\n"
            f"Problem: {problem}\n\n"
            f"Analyze the problem and show your reasoning:"
        )
    elif role == "solver":
        user_prompt = (
            f"You are a Solver Agent. You have received analysis and reasoning "
            f"from a Researcher who already examined this problem. "
            f"Use their work to compute the final answer.\n\n"
            f"Problem: {problem}\n\n"
            f"Based on the research provided, compute the final answer. "
            f"Show your work and output the answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_prompt(
    role: str, problem: str, context: str = ""
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, the Researcher's text output is included in the Solver's prompt.
    """
    if role == "researcher":
        user_content = (
            f"You are a Researcher Agent. Analyze the following competition math problem. "
            f"Identify the relevant mathematical concepts, plan the approach, "
            f"and show your intermediate reasoning.\n\n"
            f"## Problem:\n{problem}\n\n"
            f"## Format your response as follows:\n"
            f"Researcher's Analysis:\n"
            f"[Your detailed analysis and intermediate reasoning here]\n\n"
            f"Now output your analysis:"
        )
    elif role == "solver":
        user_content = (
            f"You are a Solver Agent. You are provided with:\n"
            f"(1) the original problem, and\n"
            f"(2) the Researcher's analysis in text format.\n\n"
            f"## Problem:\n{problem}\n\n"
            f"## Researcher's Analysis:\n{context}\n\n"
            f"Using the Researcher's analysis, compute the final answer. "
            f"Show your work and output the answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
