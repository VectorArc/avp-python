"""Agent definitions and prompt builders for the 2-agent DebugBench benchmark.

Two agents:
1. Detector -- Analyzes buggy code and identifies the bug
2. Fixer -- Fixes the buggy code using the Detector's analysis
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the 2-agent chain."""
    name: str
    role: str


DETECTOR = Agent(name="Detector", role="detector")
FIXER = Agent(name="Fixer", role="fixer")
AGENTS = [DETECTOR, FIXER]

SYSTEM_MESSAGE = (
    "You are an expert programmer who specializes in debugging code. "
    "You analyze buggy code, identify bugs precisely, and produce correct fixes."
)

# Direct single-agent prompt
DIRECT_SOLVE_PROMPT = (
    "The following Python code contains a bug. Fix the bug and output the "
    "corrected code. Only output the complete fixed code -- no explanation, "
    "no markdown fences.\n\n"
    "## Problem Description\n{question}\n\n"
    "## Examples\n{examples}\n\n"
    "## Buggy Code\n{buggy_code}"
)


def build_text_prompt(
    role: str, question: str, buggy_code: str, context: str = ""
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, the Detector's text analysis is included in the Fixer's prompt.
    """
    if role == "detector":
        user_content = (
            f"You are a Detector Agent. Analyze the following buggy Python code. "
            f"Identify the bug type, its exact location (line/statement), and "
            f"explain what is wrong and how to fix it. Be precise and concise.\n\n"
            f"## Problem Description\n{question}\n\n"
            f"## Buggy Code\n{buggy_code}"
        )
    elif role == "fixer":
        user_content = (
            f"You are a Fixer Agent. Fix the buggy Python code below. "
            f"A Detector has analyzed the code and found:\n\n"
            f"## Detector's Analysis\n{context}\n\n"
            f"## Problem Description\n{question}\n\n"
            f"## Buggy Code\n{buggy_code}\n\n"
            f"Output the complete corrected code. Only output code -- "
            f"no explanation, no markdown fences."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_latent_prompt(
    role: str, question: str, buggy_code: str
) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache -- the prompt only
    contains the agent's role instruction and the problem details.
    """
    if role == "detector":
        user_prompt = (
            f"You are a Detector Agent. Analyze this buggy Python code carefully. "
            f"Identify the bug type, location, and root cause.\n\n"
            f"## Problem Description\n{question}\n\n"
            f"## Buggy Code\n{buggy_code}"
        )
    elif role == "fixer":
        user_prompt = (
            f"You are a Fixer Agent. A Detector has analyzed the buggy code below. "
            f"Fix the bug and output the complete corrected code. "
            f"Only output code -- no explanation, no markdown fences.\n\n"
            f"## Problem Description\n{question}\n\n"
            f"## Buggy Code\n{buggy_code}"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def format_examples(examples: List[str]) -> str:
    """Format the examples list into a readable string for prompts."""
    if not examples:
        return "(no examples provided)"
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:\n{ex}")
    return "\n\n".join(parts)
