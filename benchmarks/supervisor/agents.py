"""Agent definitions and prompt builders for Supervisor (Router) benchmark.

Topology:
    Router --> select 1 of 4 Specialists --> Answer

Router classifies each question into a category, then the appropriate
specialist answers it. In latent mode, the Router's KV-cache is passed
to the selected specialist.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Agent:
    """A single agent in the pipeline."""
    name: str
    role: str
    subject: str = ""


ROUTER = Agent(name="Router", role="router", subject="")

SPECIALISTS = {
    "stem": Agent(
        name="STEM_Specialist", role="specialist",
        subject="elementary_mathematics",
    ),
    "humanities": Agent(
        name="Humanities_Specialist", role="specialist",
        subject="high_school_us_history",
    ),
    "social": Agent(
        name="Social_Specialist", role="specialist",
        subject="high_school_psychology",
    ),
    "logic": Agent(
        name="Logic_Specialist", role="specialist",
        subject="formal_logic",
    ),
}

# Map MMLU subjects to routing categories
SUBJECT_TO_CATEGORY = {
    "elementary_mathematics": "stem",
    "high_school_us_history": "humanities",
    "high_school_psychology": "social",
    "formal_logic": "logic",
}

SYSTEM_MESSAGE = "You are a helpful assistant."

DIRECT_SOLVE_PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer from the options.\n\n"
    "Question: {question}\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n\n"
    "Output ONLY the letter of your answer (A, B, C, or D)."
)


def format_choices(choices: List[str]) -> str:
    """Format choices as A/B/C/D lines."""
    labels = ["A", "B", "C", "D"]
    return "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def build_router_prompt(question: str, choices: List[str]) -> List[Dict[str, str]]:
    """Build prompt for the Router agent to classify the question."""
    choices_text = format_choices(choices)
    user_prompt = (
        f"You are a Router Agent. Classify the following question into one of "
        f"these categories: STEM, Humanities, Social, or Logic.\n\n"
        f"Question: {question}\n"
        f"{choices_text}\n\n"
        f"Output ONLY the category name (STEM, Humanities, Social, or Logic)."
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_latent_specialist_prompt(
    question: str, choices: List[str]
) -> List[Dict[str, str]]:
    """Build prompt for specialist in latent mode (KV carries Router context)."""
    choices_text = format_choices(choices)
    user_prompt = (
        f"You are a Specialist Agent. A Router has classified this question "
        f"(classification in latent KV representation).\n\n"
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {question}\n"
        f"{choices_text}\n\n"
        f"Output ONLY the letter of your answer (A, B, C, or D)."
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def build_text_specialist_prompt(
    question: str, choices: List[str], router_text: str
) -> List[Dict[str, str]]:
    """Build prompt for specialist in text mode (includes Router's text)."""
    choices_text = format_choices(choices)
    user_prompt = (
        f"You are a Specialist Agent. A Router has classified this question.\n\n"
        f"## Router's Classification:\n{router_text}\n\n"
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {question}\n"
        f"{choices_text}\n\n"
        f"Output ONLY the letter of your answer (A, B, C, or D)."
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


def extract_route(text: str) -> str:
    """Extract routing category from Router's output.

    Returns one of: "stem", "humanities", "social", "logic".
    Falls back to "stem" if no category is found.
    """
    text_lower = text.lower().strip()

    # Check for exact category matches
    for category in ["stem", "humanities", "social", "logic"]:
        if category in text_lower:
            return category

    return "stem"
