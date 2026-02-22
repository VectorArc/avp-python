"""vLLM engine connector for AVP.

Wraps a vLLM LLM engine to provide the EngineConnector interface for
handshake, identity extraction, and embedding injection.

vLLM is a serving engine — it cannot expose per-step hidden states.
This connector supports:
- Model identity extraction (from HF config inside vLLM)
- Tokenization
- Text generation (vLLM's sweet spot)
- Embedding injection via prompt_embeds API
- Tied/untied weight detection

It does NOT support:
- extract_hidden_state() — raises EngineNotAvailableError
- generate_latent_steps() — not applicable to serving engines

Requires vllm — uses lazy imports so the core SDK works without it.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..errors import EngineNotAvailableError
from ..handshake import compute_model_hash, extract_model_identity
from ..types import ModelIdentity
from .base import EngineConnector


def _require_vllm():
    """Check that vllm is available."""
    try:
        import vllm
        return vllm
    except ImportError:
        raise EngineNotAvailableError(
            "vllm (requires vllm>=0.8.0). Install with: pip install avp[vllm]"
        )


class VLLMConnector(EngineConnector):
    """Engine connector for vLLM serving engine.

    Accepts a vLLM LLM instance and provides the EngineConnector interface
    for AVP handshake and communication.

    Example:
        >>> from vllm import LLM
        >>> engine = LLM(model="Qwen/Qwen2.5-7B-Instruct")
        >>> connector = VLLMConnector(engine=engine)
        >>> identity = connector.get_model_identity()
    """

    def __init__(
        self,
        engine: Any = None,
        model_id: Optional[str] = None,
    ):
        """Initialize VLLMConnector.

        Args:
            engine: A vLLM LLM instance.
            model_id: Model ID to load (creates a new LLM instance).
                Provide either engine or model_id, not both.
        """
        if engine is not None:
            self._engine = engine
        elif model_id is not None:
            vllm = _require_vllm()
            self._engine = vllm.LLM(model=model_id)
        else:
            raise ValueError("Provide either engine or model_id")

        # Extract HF config from vLLM engine
        self._hf_config = self._get_hf_config()
        self._config_dict = self._hf_config_to_dict()
        self._tokenizer = self._get_tokenizer()

        self._model_hash = compute_model_hash(self._config_dict)
        self._identity = extract_model_identity(
            self._config_dict, tokenizer=self._tokenizer
        )

    def _get_hf_config(self) -> Any:
        """Extract HuggingFace config from vLLM engine."""
        # vLLM >= 0.8: engine.llm_engine.model_config.hf_config
        if hasattr(self._engine, "llm_engine"):
            engine = self._engine.llm_engine
            if hasattr(engine, "model_config"):
                return engine.model_config.hf_config
        # vLLM direct config access
        if hasattr(self._engine, "model_config"):
            return self._engine.model_config.hf_config
        # Fallback: check if it's already a config-like object
        if hasattr(self._engine, "config"):
            return self._engine.config
        raise EngineNotAvailableError(
            "vllm (cannot extract HF config from engine)"
        )

    def _hf_config_to_dict(self) -> Dict[str, Any]:
        """Convert HF config to dict."""
        if hasattr(self._hf_config, "to_dict"):
            return self._hf_config.to_dict()
        if isinstance(self._hf_config, dict):
            return self._hf_config
        # Fallback: extract known fields
        return {
            "model_type": getattr(self._hf_config, "model_type", ""),
            "_name_or_path": getattr(self._hf_config, "_name_or_path", ""),
            "hidden_size": getattr(self._hf_config, "hidden_size", 0),
            "num_hidden_layers": getattr(self._hf_config, "num_hidden_layers", 0),
            "num_attention_heads": getattr(self._hf_config, "num_attention_heads", 0),
            "num_key_value_heads": getattr(
                self._hf_config, "num_key_value_heads",
                getattr(self._hf_config, "num_attention_heads", 0)
            ),
            "head_dim": getattr(self._hf_config, "head_dim", 0),
            "tie_word_embeddings": getattr(self._hf_config, "tie_word_embeddings", False),
        }

    def _get_tokenizer(self) -> Any:
        """Get tokenizer from vLLM engine."""
        if hasattr(self._engine, "get_tokenizer"):
            return self._engine.get_tokenizer()
        if hasattr(self._engine, "llm_engine"):
            engine = self._engine.llm_engine
            if hasattr(engine, "get_tokenizer"):
                return engine.get_tokenizer()
            if hasattr(engine, "tokenizer"):
                tok = engine.tokenizer
                # vLLM wraps tokenizer in TokenizerGroup
                if hasattr(tok, "tokenizer"):
                    return tok.tokenizer
                return tok
        raise EngineNotAvailableError(
            "vllm (cannot extract tokenizer from engine)"
        )

    # ----- EngineConnector interface -----

    def get_model_identity(self) -> ModelIdentity:
        return self._identity

    def extract_hidden_state(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any]:
        """Not supported by vLLM — it's a serving engine without hidden state access.

        Raises:
            EngineNotAvailableError: Always. vLLM does not expose per-step hidden states.
        """
        raise EngineNotAvailableError(
            "vllm (extract_hidden_state not supported — vLLM is a serving engine "
            "that does not expose per-step hidden states. Use HuggingFaceConnector "
            "for hidden state extraction, or use KV-cache transfer via "
            "AVPKVConnectorV1Dynamic plugin.)"
        )

    def inject_and_generate(
        self,
        inputs_embeds: Any,
        attention_mask: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Tuple[List[str], Any]:
        """Generate text from injected embeddings via vLLM's prompt_embeds API.

        Uses vLLM's prompt_embeds feature (merged in PR #24278) to inject
        pre-computed embeddings directly, bypassing tokenization.

        Args:
            inputs_embeds: Embedding tensor [batch, seq_len, hidden_dim].
            attention_mask: Ignored (vLLM handles masking internally).
            past_key_values: Ignored (vLLM manages KV-cache internally).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Tuple of (generated_text_list, None).
            past_key_values is always None since vLLM manages its own cache.
        """
        _require_vllm()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Build prompt_embeds inputs
        # vLLM expects list of dicts with "prompt_embeds" key
        batch_size = inputs_embeds.shape[0]
        inputs = [
            {"prompt_embeds": inputs_embeds[i]}
            for i in range(batch_size)
        ]

        outputs = self._engine.generate(inputs, sampling_params)

        texts = [output.outputs[0].text for output in outputs]
        return texts, None

    def get_embedding_weights(self) -> Tuple[Any, Any]:
        """Get input and output embedding weight matrices.

        Loads embedding weights lazily from safetensors to avoid loading
        the full model a second time. Falls back to extracting from the
        vLLM model worker if safetensors are unavailable.

        Returns:
            Tuple of (input_embedding_weight, output_embedding_weight).
        """
        model_id = self._config_dict.get("_name_or_path", "")
        if not model_id:
            return None, None

        try:
            return self._load_embeddings_from_safetensors(model_id)
        except Exception:
            return self._load_embeddings_from_engine()

    def _load_embeddings_from_safetensors(self, model_id: str) -> Tuple[Any, Any]:
        """Load only embedding layers from safetensors (avoids full model load)."""
        import torch
        from safetensors import safe_open

        from huggingface_hub import hf_hub_download

        # Common embedding weight names across model families
        embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "gpt_neox.embed_in.weight",
        ]
        lm_head_names = [
            "lm_head.weight",
            "transformer.lm_head.weight",
        ]

        # Try to find the safetensors index or single file
        try:
            index_path = hf_hub_download(model_id, "model.safetensors.index.json")
            import json
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
        except Exception:
            # Single file model
            weight_map = None

        input_embed = None
        output_embed = None

        if weight_map:
            for name in embed_names + lm_head_names:
                if name in weight_map:
                    shard = weight_map[name]
                    shard_path = hf_hub_download(model_id, shard)
                    with safe_open(shard_path, framework="pt") as f:
                        tensor = f.get_tensor(name)
                        if name in embed_names:
                            input_embed = tensor
                        else:
                            output_embed = tensor
        else:
            try:
                path = hf_hub_download(model_id, "model.safetensors")
                with safe_open(path, framework="pt") as f:
                    keys = f.keys()
                    for name in embed_names:
                        if name in keys:
                            input_embed = f.get_tensor(name)
                            break
                    for name in lm_head_names:
                        if name in keys:
                            output_embed = f.get_tensor(name)
                            break
            except Exception:
                raise

        # For tied weights, output == input
        if output_embed is None and self._config_dict.get("tie_word_embeddings", False):
            output_embed = input_embed

        return input_embed, output_embed

    def _load_embeddings_from_engine(self) -> Tuple[Any, Any]:
        """Fallback: try to extract embeddings from vLLM's model worker."""
        # This is engine-internal and may not always work
        try:
            worker = self._engine.llm_engine.model_executor.driver_worker
            model = worker.model_runner.model
            input_embed = model.get_input_embeddings()
            output_embed = model.get_output_embeddings()
            return (
                input_embed.weight if input_embed else None,
                output_embed.weight if output_embed else None,
            )
        except (AttributeError, RuntimeError):
            return None, None

    def tokenize(self, text: str) -> Any:
        """Tokenize text into input IDs."""
        import torch

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def needs_realignment(self) -> bool:
        """Check if this model needs realignment (untied weights)."""
        return not self._config_dict.get("tie_word_embeddings", False)

    # ----- vLLM-specific methods -----

    def generate_text(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        """Native text-in/text-out generation (vLLM's sweet spot).

        Args:
            prompts: List of text prompts.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            List of generated text strings.
        """
        _require_vllm()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self._engine.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    @property
    def engine(self) -> Any:
        """Access the underlying vLLM engine."""
        return self._engine
