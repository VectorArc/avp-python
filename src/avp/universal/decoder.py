"""Universal decoder — maps universal tokens back to target model space.

Architecture:
- 6 self-attention layers with FFN and LayerNorm
- Output projection: Linear(D_universal → D_target)
- Gate head: Linear(D_universal → 1) + sigmoid — confidence gate
- Strips special tokens (global + style), outputs K semantic tokens
- NormMatch: scales output to target model's mean embedding norm
"""

from .._torch_compat import require_torch as _require_torch
from .config import UniversalConfig


def _build_decoder(d_target: int, config: UniversalConfig):
    """Build and return a UniversalDecoder module.

    Lazy-imports torch to keep the module importable without GPU deps.
    """
    torch = _require_torch()
    nn = torch.nn

    class _SelfAttentionBlock(nn.Module):
        """Single self-attention block: MHA + FFN + LayerNorm."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.norm_attn = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.norm_ff = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            # Self-attention
            normed = self.norm_attn(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out

            # FFN
            x = x + self.ffn(self.norm_ff(x))
            return x

    class UniversalDecoder(nn.Module):
        """Decoder: [K+2, D_universal] → ([K, D_target], gate).

        Args:
            d_target: Target model hidden/embedding dimension.
            config: UniversalConfig with architecture hyperparameters.
        """

        def __init__(self, d_target: int, config: UniversalConfig):
            super().__init__()
            self.config = config
            self.d_target = d_target
            d = config.d_universal

            # Self-attention layers
            self.layers = nn.ModuleList([
                _SelfAttentionBlock(d, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ])

            self.pre_proj_norm = nn.LayerNorm(d)

            # Output projection to target model space
            self.output_projection = nn.Linear(d, d_target)

            # Gate head: scalar confidence from pooled tokens
            self.gate_head = nn.Sequential(
                nn.Linear(d, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            universal_tokens: "torch.Tensor",
            target_norm=None,
        ) -> tuple:
            """Decode universal tokens to target model space.

            Args:
                universal_tokens: [K+2, D_universal] or [B, K+2, D_universal].
                    Token layout: [K semantic, global, style].
                target_norm: If provided, scale output embeddings to this norm.

            Returns:
                Tuple of (decoded, gate):
                - decoded: [K, D_target] or [B, K, D_target] semantic embeddings.
                - gate: float, scalar confidence in (0, 1).
            """
            squeeze_batch = False
            if universal_tokens.dim() == 2:
                universal_tokens = universal_tokens.unsqueeze(0)
                squeeze_batch = True

            K = self.config.k_tokens

            # Self-attention over all tokens (including global + style)
            x = universal_tokens
            for layer in self.layers:
                x = layer(x)

            # Compute gate from mean-pooled representation
            pooled = x.mean(dim=1)  # [B, D]
            gate = self.gate_head(pooled).squeeze(-1)  # [B]

            # Strip special tokens: keep only first K semantic tokens
            semantic = x[:, :K, :]  # [B, K, D]

            # Project to target space
            normed = self.pre_proj_norm(semantic)
            decoded = self.output_projection(normed)  # [B, K, D_target]

            # NormMatch: scale to target model's mean embedding norm
            if target_norm is not None:
                current_norm = decoded.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if isinstance(target_norm, (int, float)):
                    decoded = decoded * (target_norm / current_norm)
                else:
                    decoded = decoded * (target_norm.to(decoded.device) / current_norm)

            if squeeze_batch:
                decoded = decoded.squeeze(0)
                gate_val = float(gate.squeeze(0).detach())
            else:
                gate_val = float(gate.mean().detach())

            return decoded, gate_val

    return UniversalDecoder(d_target, config)


class UniversalDecoder:
    """Factory for UniversalDecoder nn.Module instances.

    Usage::

        decoder = UniversalDecoder.create(d_target=3072, config=UniversalConfig())
        decoded, gate = decoder(universal_tokens)  # [K+2, 512] → ([K, 3072], float)

    The actual nn.Module is returned by ``create()``. This wrapper avoids
    importing torch at module level.
    """

    @staticmethod
    def create(d_target: int, config: "UniversalConfig | None" = None):
        """Create a UniversalDecoder nn.Module.

        Args:
            d_target: Target model hidden/embedding dimension.
            config: Optional config (defaults to UniversalConfig()).

        Returns:
            An nn.Module with forward(universal_tokens) → (decoded, gate).
        """
        if config is None:
            config = UniversalConfig()
        return _build_decoder(d_target, config)
