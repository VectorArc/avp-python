"""Agent definitions and prompt builders for the 2-agent HumanEval benchmark.

Two agents:
1. Coder — Completes the Python function
2. Reviewer — Reviews and fixes the Coder's solution
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the 2-agent chain."""
    name: str
    role: str


AGENTS = [
    Agent(name="Coder", role="coder"),
    Agent(name="Reviewer", role="reviewer"),
]

SYSTEM_MESSAGE = "You are an expert Python programmer."

# Direct single-agent prompt
DIRECT_SOLVE_PROMPT = (
    "Complete the following Python function. Only output the function body "
    "— no explanation, no tests, no markdown fences.\n\n"
    "{prompt}"
)


def build_latent_prompt(role: str, prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache — the prompt only
    contains the agent's role instruction and the function signature.
    """
    if role == "coder":
        user_prompt = (
            f"You are a Coder Agent. Complete the following Python function. "
            f"Only output the function body — no explanation, no tests, "
            f"no markdown fences.\n\n"
            f"{prompt}"
        )
    elif role == "reviewer":
        user_prompt = (
            f"You are a Reviewer Agent. A Coder has drafted a solution to "
            f"the function below. Review it for correctness, edge cases, "
            f"and bugs. Output the complete corrected function. "
            f"Only output code — no explanation, no markdown fences.\n\n"
            f"{prompt}"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_prompt(
    role: str, prompt: str, context: str = ""
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, the Coder's text output is included in the Reviewer's prompt.
    """
    if role == "coder":
        user_content = (
            f"You are a Coder Agent. Complete the following Python function. "
            f"Only output the function body — no explanation, no tests, "
            f"no markdown fences.\n\n"
            f"{prompt}"
        )
    elif role == "reviewer":
        user_content = (
            f"You are a Reviewer Agent. You are provided with:\n"
            f"(1) the function signature and docstring, and\n"
            f"(2) the Coder's draft solution.\n\n"
            f"## Function Signature:\n{prompt}\n\n"
            f"## Coder's Draft:\n{context}\n\n"
            f"Review for correctness, edge cases, and bugs. "
            f"Output the complete corrected function. "
            f"Only output code — no explanation, no markdown fences."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
