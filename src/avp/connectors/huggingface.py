"""HuggingFace Transformers connector for AVP.

Wraps a HuggingFace PreTrainedModel + tokenizer to provide the EngineConnector
interface for latent communication.

Requires torch and transformers — uses lazy imports.
"""

from typing import Any, List, Optional, Tuple

from ..errors import EngineNotAvailableError, RealignmentError
from ..handshake import compute_model_hash, extract_model_identity
from ..realign import (
    apply_realignment,
    compute_target_norm,
    get_or_compute_realignment,
    needs_realignment,
    normalize_to_target,
    project_to_embedding_space,
)
from ..types import ModelIdentity
from .base import EngineConnector


def _require_deps():
    """Check that torch and transformers are available."""
    try:
        import torch
    except ImportError:
        raise EngineNotAvailableError(
            "huggingface (requires torch). Install with: pip install avp[latent]"
        )
    try:
        import transformers
    except ImportError:
        raise EngineNotAvailableError(
            "huggingface (requires transformers). Install with: pip install avp[latent]"
        )
    return torch, transformers


def _past_length(past_kv: Any) -> int:
    """Get sequence length from past_key_values."""
    if past_kv is None:
        return 0
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            return past_kv.get_seq_length()
    except ImportError:
        pass
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (tuple, list)) and len(first) > 0:
            return first[0].shape[-2]
    return 0


class HuggingFaceConnector(EngineConnector):
    """Engine connector for HuggingFace Transformers models.

    Accepts either a pre-loaded model+tokenizer or loads from model_id.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        >>> connector = HuggingFaceConnector(model=model, tokenizer=tokenizer)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        torch, transformers = _require_deps()

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_id is not None:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        else:
            raise ValueError("Provide either (model, tokenizer) or model_id")

        if device:
            self.device = device
            self.model = self.model.to(device)
        else:
            self.device = str(next(self.model.parameters()).device)

        # Ensure pad token (following LatentMAS: try eos first, then add <pad>)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        self._model_hash = compute_model_hash(self.model.config.to_dict())
        self._identity = extract_model_identity(self.model)
        self._w_realign = None
        self._target_norm = None

    def get_model_identity(self) -> ModelIdentity:
        return self._identity

    def extract_hidden_state(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any]:
        """Run forward pass and extract hidden states + KV-cache.

        Returns:
            Tuple of (last_hidden_state [B, D], all_hidden_states, past_key_values).
        """
        import torch

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        # Prepend past attention if using KV-cache
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]
        return last_hidden, outputs.hidden_states, outputs.past_key_values

    def inject_and_generate(
        self,
        inputs_embeds: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[List[str], Any]:
        """Generate text from injected embeddings."""
        import torch

        if attention_mask is None:
            batch_size = inputs_embeds.shape[0]
            total_len = inputs_embeds.shape[1]
            if past_key_values is not None:
                total_len += _past_length(past_key_values)
            attention_mask = torch.ones(
                (batch_size, total_len), dtype=torch.long, device=self.device
            )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
            )

        # Decode generated tokens (skip prompt tokens)
        prompt_len = inputs_embeds.shape[1]
        if past_key_values is not None:
            prompt_len += _past_length(past_key_values)
        texts = []
        for seq in outputs.sequences:
            generated_ids = seq[prompt_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            texts.append(text)

        return texts, getattr(outputs, "past_key_values", None)

    def get_embedding_weights(self) -> Tuple[Any, Any]:
        input_embeds = self.model.get_input_embeddings()
        output_embeds = self.model.get_output_embeddings()
        if output_embeds is None:
            output_embeds = getattr(self.model, "lm_head", None)
        return (
            input_embeds.weight if input_embeds else None,
            output_embeds.weight if output_embeds else None,
        )

    def tokenize(self, text: str) -> Any:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    def needs_realignment(self) -> bool:
        return needs_realignment(self.model)

    def _ensure_realignment(self) -> Tuple[Any, Any]:
        """Load or compute realignment matrix."""
        if self._w_realign is None:
            self._w_realign, self._target_norm = get_or_compute_realignment(
                self.model, self._model_hash, device=self.device
            )
        return self._w_realign, self._target_norm

    def _ensure_target_norm(self) -> Any:
        """Get target norm, computing if needed.

        Target norm is needed for ALL models (tied and untied) because hidden
        states from the last transformer layer have different norms than input
        embeddings. LatentMAS always normalizes, even without the projection.
        """
        if self._target_norm is None:
            self._target_norm = compute_target_norm(self.model, device=self.device)
        return self._target_norm

    def generate_latent_steps(
        self,
        input_ids: Any,
        latent_steps: int,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> Any:
        """Run the LatentMAS latent generation loop.

        Performs `latent_steps` forward passes, each time:
        1. Extract hidden state from last token
        2. Apply realignment projection (untied models) or just normalize (tied models)
        3. Feed aligned hidden state as inputs_embeds for next step
        4. Accumulate KV-cache

        Normalization to target_norm is ALWAYS applied, even for tied-weight
        models. This matches LatentMAS behavior: hidden states from the last
        transformer layer have different norms than input embeddings, and
        injecting un-normalized vectors gives the model out-of-distribution inputs.

        Ported from LatentMAS models.py:276-350.

        Args:
            input_ids: Token IDs [batch, seq_len].
            latent_steps: Number of latent thinking steps.
            attention_mask: Optional attention mask.
            past_key_values: Optional prior KV-cache.

        Returns:
            Updated past_key_values after all latent steps.
        """
        import torch

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        # Prepend past attention
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Initial forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]

        # Determine realignment strategy
        do_realign = self.needs_realignment()
        if do_realign:
            w_realign, target_norm = self._ensure_realignment()
            embed_weight = None
        else:
            # Tied models: project hidden → logits → softmax → soft embedding.
            # Simple normalization doesn't work because hidden state directions
            # have very low cosine similarity (~0.24) to embedding vectors.
            target_norm = None
            embed_weight = self.model.get_input_embeddings().weight

        # Latent loop
        for step in range(latent_steps):
            if do_realign:
                latent_vec = apply_realignment(last_hidden, w_realign, target_norm)
            else:
                # Tied models: project through vocabulary to get valid embedding
                latent_vec = project_to_embedding_space(
                    last_hidden, embed_weight, temperature=1.0
                )

            latent_embed = latent_vec.unsqueeze(1)  # [B, 1, D]

            # Build attention mask for this step
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
