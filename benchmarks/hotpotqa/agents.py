"""Agent definitions and prompt builders for HotpotQA benchmark.

Default 2-agent setup:
1. Finder — Reads paragraphs, identifies relevant supporting facts
2. Answerer — Synthesizes final answer from Finder's analysis

Optional 3-agent setup:
1. Decomposer — Breaks multi-hop question into sub-questions
2. Finder — Answers sub-questions using paragraphs
3. Answerer — Synthesizes final answer
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Agent:
    """A single agent in the pipeline."""
    name: str
    role: str


AGENTS_2 = [
    Agent(name="Finder", role="finder"),
    Agent(name="Answerer", role="answerer"),
]

AGENTS_3 = [
    Agent(name="Decomposer", role="decomposer"),
    Agent(name="Finder", role="finder_3agent"),
    Agent(name="Answerer", role="answerer_3agent"),
]

SYSTEM_MESSAGE = "You are a helpful assistant specialized in reading comprehension."


def format_paragraphs(paragraphs: List[Dict]) -> str:
    """Format HotpotQA context paragraphs into readable text."""
    parts = []
    for para in paragraphs:
        title = para["title"]
        sentences = para["sentences"]
        text = " ".join(sentences)
        parts.append(f"[{title}]\n{text}")
    return "\n\n".join(parts)


# --- Direct (single agent) ---

DIRECT_PROMPT = (
    "Read the following paragraphs and answer the question. "
    "Give a short, precise answer (a few words at most).\n\n"
    "## Paragraphs:\n{paragraphs}\n\n"
    "## Question: {question}\n\n"
    "Answer:"
)


# --- Latent prompts ---

def build_latent_prompt(
    role: str,
    question: str,
    paragraphs_text: str = "",
) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache.
    """
    if role == "finder":
        user_prompt = (
            f"You are a Finder Agent. Read the following paragraphs carefully "
            f"and identify the key facts needed to answer the question. "
            f"Focus on finding the relevant supporting facts across paragraphs.\n\n"
            f"## Paragraphs:\n{paragraphs_text}\n\n"
            f"## Question: {question}\n\n"
            f"Identify the relevant facts and reasoning chain:"
        )
    elif role == "answerer":
        # In latent mode, the Answerer gets the question but NOT the paragraphs.
        # The Finder's KV-cache carries the model's understanding of the context.
        user_prompt = (
            f"You are an Answerer Agent. A Finder agent has already analyzed "
            f"the relevant paragraphs for you. Use their analysis to answer "
            f"the question.\n\n"
            f"## Question: {question}\n\n"
            f"Based on the analysis provided, give a short, precise answer "
            f"(a few words at most).\n\n"
            f"Answer:"
        )
    elif role == "decomposer":
        user_prompt = (
            f"You are a Decomposer Agent. Break the following multi-hop question "
            f"into simpler sub-questions that can be answered individually.\n\n"
            f"## Question: {question}\n\n"
            f"List the sub-questions:"
        )
    elif role == "finder_3agent":
        user_prompt = (
            f"You are a Finder Agent. A Decomposer has broken down the question "
            f"into sub-questions. Read the paragraphs and find answers to each "
            f"sub-question.\n\n"
            f"## Paragraphs:\n{paragraphs_text}\n\n"
            f"## Original Question: {question}\n\n"
            f"Find the relevant facts for each sub-question:"
        )
    elif role == "answerer_3agent":
        user_prompt = (
            f"You are an Answerer Agent. Previous agents have decomposed the "
            f"question and found relevant facts. Synthesize a final answer.\n\n"
            f"## Question: {question}\n\n"
            f"Based on the analysis provided, give a short, precise answer "
            f"(a few words at most).\n\n"
            f"Answer:"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]


# --- Text prompts ---

def build_text_prompt(
    role: str,
    question: str,
    paragraphs_text: str = "",
    context: str = "",
) -> List[Dict[str, str]]:
    """Build chat messages for text mode agents.

    In text mode, prior agents' text output is included in the prompt.
    """
    if role == "finder":
        user_content = (
            f"You are a Finder Agent. Read the following paragraphs carefully "
            f"and identify the key facts needed to answer the question.\n\n"
            f"## Paragraphs:\n{paragraphs_text}\n\n"
            f"## Question: {question}\n\n"
            f"## Format your response as follows:\n"
            f"Relevant Facts:\n"
            f"[List the key facts from the paragraphs that help answer the question]\n\n"
            f"Reasoning:\n"
            f"[Explain how these facts connect to answer the question]\n\n"
            f"Now output your analysis:"
        )
    elif role == "answerer":
        user_content = (
            f"You are an Answerer Agent. You are provided with:\n"
            f"(1) the original question, and\n"
            f"(2) the Finder's analysis of relevant paragraphs.\n\n"
            f"## Question: {question}\n\n"
            f"## Finder's Analysis:\n{context}\n\n"
            f"Using the Finder's analysis, give a short, precise answer "
            f"(a few words at most).\n\n"
            f"Answer:"
        )
    elif role == "decomposer":
        user_content = (
            f"You are a Decomposer Agent. Break the following multi-hop question "
            f"into simpler sub-questions.\n\n"
            f"## Question: {question}\n\n"
            f"List the sub-questions that need to be answered:"
        )
    elif role == "finder_3agent":
        user_content = (
            f"You are a Finder Agent. A Decomposer has broken down the question. "
            f"Read the paragraphs and find answers.\n\n"
            f"## Paragraphs:\n{paragraphs_text}\n\n"
            f"## Original Question: {question}\n\n"
            f"## Decomposer's Sub-questions:\n{context}\n\n"
            f"Find facts and answer each sub-question:"
        )
    elif role == "answerer_3agent":
        user_content = (
            f"You are an Answerer Agent. Previous agents have analyzed the question.\n\n"
            f"## Question: {question}\n\n"
            f"## Previous Analysis:\n{context}\n\n"
            f"Give a short, precise answer (a few words at most).\n\n"
            f"Answer:"
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
