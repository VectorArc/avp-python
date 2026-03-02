"""Configuration for Universal Representation Space adapters."""

from dataclasses import dataclass


@dataclass
class UniversalConfig:
    """Configuration for universal encoder/decoder adapters.

    Default values follow Vision Wormhole architecture:
    D=512, 6 cross-attention layers, 8 heads, K=64 semantic tokens.
    """

    d_universal: int = 512
    """Dimension of the universal representation space."""

    k_tokens: int = 64
    """Number of semantic universal tokens (K). Excludes 2 special tokens
    (global + style) which are added automatically."""

    num_layers: int = 6
    """Number of cross-attention (encoder) or self-attention (decoder) layers."""

    num_heads: int = 8
    """Number of attention heads."""

    dropout: float = 0.1
    """Dropout rate for attention and FFN layers."""

    rollout_steps: int = 256
    """Default number of latent rollout steps for hidden state collection."""

    def __post_init__(self) -> None:
        if self.d_universal <= 0:
            raise ValueError(f"d_universal must be positive, got {self.d_universal}")
        if self.d_universal % self.num_heads != 0:
            raise ValueError(
                f"d_universal ({self.d_universal}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.k_tokens <= 0:
            raise ValueError(f"k_tokens must be positive, got {self.k_tokens}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
