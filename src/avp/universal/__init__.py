"""Universal Representation Space — learned cross-model latent projection for AVP.

Per-model encoder/decoder adapters map between model hidden states and
K universal tokens in a shared representation space. Enables cross-model
latent communication with higher fidelity than single-embedding Rosetta.
"""

from .adapter_registry import UniversalAdapter, find_adapter, load_adapter, save_adapter
from .config import UniversalConfig
from .decoder import UniversalDecoder
from .encoder import UniversalEncoder

__all__ = [
    "UniversalConfig",
    "UniversalEncoder",
    "UniversalDecoder",
    "UniversalAdapter",
    "save_adapter",
    "load_adapter",
    "find_adapter",
]
