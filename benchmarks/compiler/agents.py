"""Agent definitions and prompt builders for Compiler (Plan-Execute) benchmark.

Topology:
    Planner --> Executor --> Verifier --> Answer

Three agents:
1. Planner — Identifies math concepts, breaks problem into steps
2. Executor — Carries out each calculation step
3. Verifier — Checks the solution, outputs final answer in \\boxed{}
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the pipeline."""
    name: str
    role: str


AGENTS = [
    Agent(name="Planner", role="planner"),
    Agent(name="Executor", role="executor"),
    Agent(name="Verifier", role="verifier"),
]

SYSTEM_MESSAGE = "You are a helpful math assistant."

DIRECT_SOLVE_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your work clearly, then give the final answer "
    "inside \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


def build_latent_prompt(role: str, problem: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache — the prompt only
    contains the agent's role instruction and the problem.
    """
    if role == "planner":
        user_prompt = (
            f"You are a Planner Agent. Create a structured solution plan for "
            f"this math problem. Identify the relevant math concepts, break the "
            f"problem into clear steps, and outline the approach.\n\n"
            f"Problem: {problem}\n\n"
            f"Output a concise plan with numbered steps. Do not solve — only plan."
        )
    elif role == "executor":
        user_prompt = (
            f"You are an Executor Agent. A Planner has analyzed this problem "
            f"and created a solution plan (provided in latent KV representation).\n\n"
            f"Problem: {problem}\n\n"
            f"Execute the plan by performing each calculation step. Show all "
            f"intermediate work. Do not skip any steps."
        )
    elif role == "verifier":
        user_prompt = (
            f"You are a Verifier Agent. A Planner and Executor have worked on "
            f"this problem (their work is in latent KV representation).\n\n"
            f"Problem: {problem}\n\n"
            f"Verify the solution by checking each step. If correct, confirm it. "
            f"If there are errors, fix them.\n\n"
            f"Output the final answer inside \\boxed{{YOUR_ANSWER}}."
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

    In text mode, prior agents' text output is included in the prompt.
    """
    if role == "planner":
        user_content = (
            f"You are a Planner Agent. Create a structured solution plan for "
            f"this math problem.\n\n"
            f"## Problem:\n{problem}\n\n"
            f"Identify relevant math concepts, break into clear steps, and "
            f"outline the approach. Output a concise plan with numbered steps. "
            f"Do not solve — only plan."
        )
    elif role == "executor":
        user_content = (
            f"You are an Executor Agent. A Planner has analyzed this problem "
            f"and created a solution plan.\n\n"
            f"## Problem:\n{problem}\n\n"
            f"## Planner's Plan:\n{context}\n\n"
            f"Execute the plan by performing each calculation step. Show all "
            f"intermediate work."
        )
    elif role == "verifier":
        user_content = (
            f"You are a Verifier Agent. A Planner and Executor have worked on "
            f"this problem.\n\n"
            f"## Problem:\n{problem}\n\n"
            f"## Previous Work:\n{context}\n\n"
            f"Verify the solution by checking each step. If correct, confirm it. "
            f"If there are errors, fix them.\n\n"
            f"Output the final answer inside \\boxed{{YOUR_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
