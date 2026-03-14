"""Text generation, prompt rendering, and tokenization utilities for benchmarks."""

from typing import Any, Dict, List, Optional, Tuple

import torch


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
    logits_processor: Optional[Any] = None,
) -> Tuple[str, Optional[Any]]:
    """Generate text from input_ids, optionally with a pre-filled KV-cache.

    Returns (generated_text, past_key_values).
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

    gen_kwargs = dict(
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
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor
    outputs = model.generate(**gen_kwargs)

    generated_ids = outputs.sequences[0, prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, getattr(outputs, "past_key_values", None)
