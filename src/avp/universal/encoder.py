"""Universal encoder — compresses model hidden states to universal tokens.

Architecture (Perceiver-style, following Vision Wormhole):
- Input projection: Linear(D_source → D_universal)
- K+2 learned query tokens: K semantic + 1 global + 1 style
- 6 cross-attention layers with FFN and LayerNorm
- Global token initialized as mean-pool of projected hidden states
- Style token initialized from rollout statistics MLP
"""

from .._torch_compat import require_torch as _require_torch
from .config import UniversalConfig


def _build_encoder(d_source: int, config: UniversalConfig):
    """Build and return a UniversalEncoder module.

    Lazy-imports torch to keep the module importable without GPU deps.
    """
    torch = _require_torch()
    nn = torch.nn

    class _CrossAttentionBlock(nn.Module):
        """Single cross-attention block: MHA + FFN + LayerNorm."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)
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

        def forward(self, query, key_value):
            # Cross-attention: queries attend to key_value
            q = self.norm_q(query)
            kv = self.norm_kv(key_value)
            attn_out, _ = self.attn(q, kv, kv)
            x = query + attn_out

            # FFN
            x = x + self.ffn(self.norm_ff(x))
            return x

    class UniversalEncoder(nn.Module):
        """Perceiver-style encoder: [T, D_src] → [K+2, D_universal].

        Args:
            d_source: Source model hidden dimension.
            config: UniversalConfig with architecture hyperparameters.
        """

        def __init__(self, d_source: int, config: UniversalConfig):
            super().__init__()
            self.config = config
            self.d_source = d_source
            d = config.d_universal
            k_total = config.k_tokens + 2  # K semantic + 1 global + 1 style

            # Input projection
            self.input_projection = nn.Linear(d_source, d)

            # Learned query tokens (K semantic only — global and style are computed)
            self.query_tokens = nn.Parameter(torch.randn(config.k_tokens, d) * 0.02)

            # Style MLP: [mean, std, avg_norm] → D_universal
            self.style_mlp = nn.Sequential(
                nn.Linear(3, d),
                nn.GELU(),
                nn.Linear(d, d),
            )

            # Cross-attention layers
            self.layers = nn.ModuleList([
                _CrossAttentionBlock(d, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ])

            self.final_norm = nn.LayerNorm(d)

        def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
            """Encode model hidden states to universal tokens.

            Args:
                hidden_states: [T, D_source] or [B, T, D_source] tensor of
                    hidden states from latent rollout steps.

            Returns:
                [K+2, D_universal] or [B, K+2, D_universal] universal tokens.
                Token layout: [K semantic tokens, global token, style token].
            """
            # Handle 2D input (no batch dim)
            squeeze_batch = False
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
                squeeze_batch = True

            B, T, _ = hidden_states.shape

            # Project to universal space
            projected = self.input_projection(hidden_states)  # [B, T, D]

            # Compute global token as mean-pool of projected hidden states
            global_token = projected.mean(dim=1, keepdim=True)  # [B, 1, D]

            # Compute style token from rollout statistics
            mean_val = projected.mean(dim=(1, 2), keepdim=False)  # [B]
            std_val = projected.std(dim=(1, 2), keepdim=False)    # [B]
            avg_norm = projected.norm(dim=-1).mean(dim=1)         # [B]
            style_input = torch.stack([mean_val, std_val, avg_norm], dim=-1)  # [B, 3]
            style_token = self.style_mlp(style_input).unsqueeze(1)  # [B, 1, D]

            # Assemble query tokens: [B, K+2, D]
            semantic = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
            queries = torch.cat([semantic, global_token, style_token], dim=1)

            # Cross-attention layers: queries attend to projected hidden states
            for layer in self.layers:
                queries = layer(queries, projected)

            output = self.final_norm(queries)

            if squeeze_batch:
                output = output.squeeze(0)

            return output

    return UniversalEncoder(d_source, config)


class UniversalEncoder:
    """Factory for UniversalEncoder nn.Module instances.

    Usage::

        encoder = UniversalEncoder.create(d_source=3584, config=UniversalConfig())
        tokens = encoder(hidden_states)  # [T, 3584] → [K+2, 512]

    The actual nn.Module is returned by ``create()``. This wrapper avoids
    importing torch at module level.
    """

    @staticmethod
    def create(d_source: int, config: "UniversalConfig | None" = None):
        """Create a UniversalEncoder nn.Module.

        Args:
            d_source: Source model hidden dimension.
            config: Optional config (defaults to UniversalConfig()).

        Returns:
            An nn.Module with forward(hidden_states) → universal_tokens.
        """
        if config is None:
            config = UniversalConfig()
        return _build_encoder(d_source, config)
