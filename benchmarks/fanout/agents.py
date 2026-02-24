"""Agent definitions and prompt builders for fan-out aggregation benchmark.

Topology:
                  +--> Algebraist --+
Input (question) -+                +--> Aggregator --> Answer
                  +--> Arithmetician -+

Three agents:
1. Algebraist — Approaches with variables and equations
2. Arithmetician — Approaches with direct step-by-step computation
3. Aggregator — Reviews both approaches, produces final answer
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the pipeline."""
    name: str
    role: str


SPECIALISTS = [
    Agent(name="Algebraist", role="algebraist"),
    Agent(name="Arithmetician", role="arithmetician"),
]

AGGREGATOR = Agent(name="Aggregator", role="aggregator")

SYSTEM_MESSAGE = "You are a helpful math assistant."

# Direct single-agent prompt (same as other GSM8K benchmarks)
DIRECT_SOLVE_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your work clearly, then give the final numeric answer "
    "inside \\boxed{{}}.\n\n"
    "Question: {question}\n\n"
    "Solution:"
)


def build_latent_prompt(role: str, question: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache.
    """
    if role == "algebraist":
        user_prompt = (
            f"You are an Algebraist Agent. Solve the following math problem "
            f"using algebraic methods. Define variables for unknown quantities, "
            f"set up equations, and solve them systematically.\n\n"
            f"Question: {question}\n\n"
            f"Show your algebraic approach step by step:"
        )
    elif role == "arithmetician":
        user_prompt = (
            f"You are an Arithmetician Agent. Solve the following math problem "
            f"using direct step-by-step arithmetic computation. Work through "
            f"the numbers directly without setting up formal equations.\n\n"
            f"Question: {question}\n\n"
            f"Show your step-by-step computation:"
        )
    elif role == "aggregator":
        user_prompt = (
            f"You are an Aggregator Agent. Two specialists have analyzed this "
            f"problem using different approaches — one algebraic, one arithmetic. "
            f"Their work is available to you.\n\n"
            f"Question: {question}\n\n"
            f"Review both approaches, verify their reasoning, and produce the "
            f"final answer. If the approaches disagree, determine which is correct.\n\n"
            f"Output the final numeric answer inside "
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

    In text mode, specialist outputs are included in the Aggregator's prompt.
    """
    if role == "algebraist":
        user_content = (
            f"You are an Algebraist Agent. Solve the following math problem "
            f"using algebraic methods. Define variables, set up equations, "
            f"and solve them systematically.\n\n"
            f"## Question:\n{question}\n\n"
            f"## Format your response as follows:\n"
            f"Algebraist's Solution:\n"
            f"[Your algebraic approach and solution here]\n\n"
            f"Show your algebraic approach step by step:"
        )
    elif role == "arithmetician":
        user_content = (
            f"You are an Arithmetician Agent. Solve the following math problem "
            f"using direct step-by-step arithmetic computation.\n\n"
            f"## Question:\n{question}\n\n"
            f"## Format your response as follows:\n"
            f"Arithmetician's Solution:\n"
            f"[Your step-by-step computation here]\n\n"
            f"Show your step-by-step computation:"
        )
    elif role == "aggregator":
        user_content = (
            f"You are an Aggregator Agent. Two specialists solved this problem "
            f"using different approaches.\n\n"
            f"## Question:\n{question}\n\n"
            f"## Specialist Solutions:\n{context}\n\n"
            f"Review both approaches. If they agree, confirm the answer. "
            f"If they disagree, determine which is correct and explain why.\n\n"
            f"Output the final numeric answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
