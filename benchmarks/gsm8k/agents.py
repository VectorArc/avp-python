"""Agent definitions, prompt builders, and text generation helper for GSM8K benchmark."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Agent:
    """A single agent in the 4-agent chain."""

    name: str
    role: str


AGENTS = [
    Agent(name="Planner", role="planner"),
    Agent(name="Critic", role="critic"),
    Agent(name="Refiner", role="refiner"),
    Agent(name="Judger", role="judger"),
]

SYSTEM_MESSAGE = "You are a helpful assistant."


def build_latent_prompt(role: str, question: str) -> List[Dict[str, str]]:
    """Build chat messages for latent mode agents.

    In latent mode, prior context is carried via KV-cache — the prompt only
    contains the agent's role instruction and the question.
    Adapted from LatentMAS prompts.py (build_agent_message_sequential_latent_mas).
    """
    if role == "planner":
        user_prompt = (
            f"You are a Planner Agent. Given an input question, design a clear, "
            f"step-by-step plan for how to solve the question.\n\n"
            f"Question: {question}\n\n"
            f"Your outlined plan should be concise with a few bullet points for each step. "
            f"Do not produce the final answer.\n"
            f"Now output your plan to solve the question below:"
        )
    elif role == "critic":
        user_prompt = (
            f"Question: {question}\n\n"
            f"You are a Critic Agent to evaluate the correctness of the input plan "
            f"for the given question and provide helpful feedback for improving the plan.\n"
            f"The plan information is provided in latent KV representation format. "
            f"Review the plan and question and output:\n"
            f"(1) original plan contents\n"
            f"(2) constructive feedback on the original plan.\n\n"
            f"Format your response as follows:\n"
            f"Original Plan: [Copy the provided Planner Agent's plan here]\n"
            f"Feedback: [Your detailed feedback to improve the plan here]\n\n"
            f"Now, output your response below:"
        )
    elif role == "refiner":
        user_prompt = (
            f"Question: {question}\n\n"
            f"You are a Refiner Agent to provide a refined step-by-step plan "
            f"for solving the given question.\n"
            f"You are provided with:\n"
            f"(1) latent-format information: a previous plan with feedback\n"
            f"(2) text-format information: the input question you need to solve.\n\n"
            f"Based on the input, write a refined and improved plan to solve the question. "
            f"Make sure your output plan is correct and concise.\n\n"
            f"Now, output your refined plan below:"
        )
    elif role == "judger":
        user_prompt = (
            f"Target Question: {question}\n\n"
            f"You are a helpful assistant. You are provided with latent information "
            f"for reference and a target question to solve.\n\n"
            f"The latent information might contain irrelevant contents. "
            f"Ignore it if it is not helpful for solving the target question.\n\n"
            f"You must reason step-by-step to solve the provided Target Question. "
            f"Calculate all intermediate values explicitly using actual numbers "
            f"(not symbolic expressions). Show each calculation.\n\n"
            f"Output the final NUMERIC answer inside "
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

    In text mode, prior agents' text output is included in the prompt.
    Adapted from LatentMAS prompts.py (build_agent_messages_sequential_text_mas).
    """
    if role == "planner":
        user_content = (
            f"You are a Planner Agent. Given an input question, design a clear, "
            f"step-by-step plan for how to solve the question.\n\n"
            f"## Input Question:\n{question}\n\n"
            f"Your outlined plan should be concise with a few bullet points for each step. "
            f"Do not produce the final answer.\n\n"
            f"## Format your response as follows:\n"
            f"Planner Agent's Output:\n"
            f"[Your detailed plan here]\n\n"
            f"Now output your plan to solve the question below:"
        )
    elif role == "critic":
        user_content = (
            f"You are a Critic Agent. You are provided with:\n"
            f"(1) the original question, and\n"
            f"(2) the Planner Agent's plan in text format.\n\n"
            f"Your job is to carefully evaluate the correctness and completeness "
            f"of the plan and provide helpful feedback.\n\n"
            f"## Input Question:\n{question}\n\n"
            f"## Plan from Planner Agent:\n{context}\n\n"
            f"## Format your response as follows:\n"
            f"Critic Agent's Output:\n"
            f"Original Plan: [Copy the provided Planner Agent's plan here]\n"
            f"Feedback: [Your detailed feedback to improve the plan here]\n\n"
            f"Now, output your response below:"
        )
    elif role == "refiner":
        user_content = (
            f"You are a Refiner Agent. You are provided with:\n"
            f"(1) the original question, and\n"
            f"(2) the Planner Agent's plan together with Critic Agent's feedback "
            f"in text format.\n\n"
            f"Your job is to incorporate the feedback and produce an improved, "
            f"refined step-by-step plan.\n\n"
            f"## Input Question:\n{question}\n\n"
            f"## Original Plan and Critic Feedback:\n{context}\n\n"
            f"## Format your response as follows:\n"
            f"Refiner Agent's Output:\n"
            f"[Your refined and improved plan here]\n\n"
            f"Make sure your output plan is logically correct, concise, and sufficient "
            f"to guide final problem solving.\n"
            f"Now, output your refined plan below:"
        )
    elif role == "judger":
        user_content = (
            f"Target Question: {question}\n\n"
            f"You are the final solver agent in a sequential multi-agent system "
            f"(planner -> critic -> refiner -> solver).\n"
            f"You are provided with the Refiner Agent's plan as reference.\n\n"
            f"Refined Plan from Previous Agents:\n{context}\n\n"
            f"The plan might contain irrelevant or incorrect contents. "
            f"Ignore them if they are not helpful for solving the target question.\n\n"
            f"You must reason step-by-step to solve the provided Target Question "
            f"without outputting other irrelevant information.\n\n"
            f"Now, reason step by step and output the final answer inside "
            f"\\boxed{{YOUR_FINAL_ANSWER}}."
        )
    else:
        raise ValueError(f"Unknown role: {role}")

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Render chat messages to a string using the tokenizer's chat template."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for models without a chat template
    segments = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
    segments.append("<|assistant|>")
    return "\n".join(segments)


def tokenize_prompt(
    tokenizer: Any, prompt_text: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a rendered prompt string, returning (input_ids, attention_mask)."""
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return input_ids, attention_mask


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    past_key_values: Optional[Any] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Tuple[str, Optional[Any]]:
    """Generate text from input_ids, optionally with a pre-filled KV-cache.

    This is the shared text generation helper used by both pipeline modes.
    Follows the pattern from LatentMAS models.py:generate_text_batch (lines 216-267).
    """
    prompt_len = attention_mask.sum(dim=1).tolist()[0]

    cache_position = None
    if past_key_values is not None:
        # Get past sequence length
        try:
            from transformers.cache_utils import Cache
            if isinstance(past_key_values, Cache):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[-2]
        except ImportError:
            past_len = past_key_values[0][0].shape[-2]

        cache_position = torch.arange(
            past_len,
            past_len + input_ids.shape[-1],
            dtype=torch.long,
            device=device,
        )
        # Extend attention mask to cover past tokens
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        past_key_values=past_key_values,
        cache_position=cache_position,
    )

    generated_ids = outputs.sequences[0, prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, getattr(outputs, "past_key_values", None)
