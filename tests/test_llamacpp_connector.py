"""Tests for LlamaCppConnector (mock-based, no model required)."""

import pytest

try:
    import llama_cpp  # noqa: F401

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False

pytestmark = [
    pytest.mark.skipif(HAS_LLAMACPP, reason="Tests for when llama-cpp-python is NOT installed"),
]


class TestLlamaCppImportGuard:
    """Test that the connector raises ImportError without llama-cpp-python."""

    def test_import_raises_without_llamacpp(self):
        from avp.connectors.llamacpp import LlamaCppConnector
        with pytest.raises(ImportError, match="llama-cpp-python"):
            LlamaCppConnector("nonexistent.gguf")


