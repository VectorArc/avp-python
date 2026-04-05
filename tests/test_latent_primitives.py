"""Tests for latent primitives: create_inference_context, run_latent_steps, generate_on_context.

These test the public API contracts without requiring a GPU or model.
Mock-based for unit testing; integration tests with real models are in test_high_level_api.py.
"""

import sys
from unittest.mock import MagicMock, patch


# --- tokenize(add_bos=) tests ---

class TestTokenizeAddBos:
    """Test add_bos parameter on tokenize() across connectors."""

    def test_base_abc_signature_accepts_add_bos(self):
        """The ABC signature includes add_bos parameter."""
        from avp.connectors.base import EngineConnector
        import inspect
        sig = inspect.signature(EngineConnector.tokenize)
        assert "add_bos" in sig.parameters
        assert sig.parameters["add_bos"].default is False

    def test_llamacpp_tokenize_passes_add_bos(self):
        """LlamaCppConnector.tokenize forwards add_bos to llama-cpp-python."""
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 100, 200]

        mock_lc = MagicMock()
        with patch.dict(sys.modules, {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_cpp": mock_lc,
        }):
            with patch("avp.connectors.llamacpp.HAS_LLAMACPP", True):
                from avp.connectors.llamacpp import LlamaCppConnector
                conn = LlamaCppConnector.__new__(LlamaCppConnector)
                conn._model = mock_model

                conn.tokenize("hello", add_bos=False)
                mock_model.tokenize.assert_called_with(
                    b"hello", add_bos=False, special=True,
                )

                conn.tokenize("hello", add_bos=True)
                mock_model.tokenize.assert_called_with(
                    b"hello", add_bos=True, special=True,
                )

    def test_huggingface_add_bos_prepends_bos_only(self):
        """HuggingFaceConnector.tokenize(add_bos=True) prepends BOS, not all special tokens."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [100, 200]
        mock_tokenizer.bos_token_id = 1

        with patch.dict(sys.modules, {
            "torch": MagicMock(),
            "transformers": MagicMock(),
        }):
            from avp.connectors.huggingface import HuggingFaceConnector
            conn = HuggingFaceConnector.__new__(HuggingFaceConnector)
            conn.tokenizer = mock_tokenizer

            # add_bos=False: no BOS
            result = conn.tokenize("hello", add_bos=False)
            mock_tokenizer.encode.assert_called_with("hello", add_special_tokens=False)
            assert result == [100, 200]

            # add_bos=True: BOS prepended, but encode still called without special tokens
            result = conn.tokenize("hello", add_bos=True)
            mock_tokenizer.encode.assert_called_with("hello", add_special_tokens=False)
            assert result == [1, 100, 200]

    def test_default_add_bos_is_false(self):
        """Default behavior unchanged — add_bos=False by default."""
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [100, 200]

        with patch.dict(sys.modules, {
            "llama_cpp": MagicMock(),
            "llama_cpp.llama_cpp": MagicMock(),
        }):
            with patch("avp.connectors.llamacpp.HAS_LLAMACPP", True):
                from avp.connectors.llamacpp import LlamaCppConnector
                conn = LlamaCppConnector.__new__(LlamaCppConnector)
                conn._model = mock_model

                conn.tokenize("hello")  # no add_bos argument
                mock_model.tokenize.assert_called_with(
                    b"hello", add_bos=False, special=True,
                )


# --- generate_on_context() tests ---

class TestGenerateOnContext:
    """Test generate_on_context() — the third latent primitive."""

    def _make_connector_and_lc(self):
        """Create a LlamaCppConnector + mock lc, with HAS_LLAMACPP patched.

        Returns (connector, mock_lc, context_manager) — caller must
        enter the context manager to activate sys.modules + HAS_LLAMACPP patches.
        """
        from contextlib import ExitStack
        from avp.connectors.llamacpp import LlamaCppConnector

        conn = LlamaCppConnector.__new__(LlamaCppConnector)
        conn._model = MagicMock()
        conn._model._model = MagicMock()
        conn._model._model.model = MagicMock()
        conn._n_embd = 64
        conn._n_vocab = 1000
        conn._n_layer = 4
        conn._model_path = "mock.gguf"
        conn._model_hash = "mock_hash"
        conn._n_ctx = 4096
        conn._cached_stop_tokens = {151643}
        conn._cached_stop_strings = ["<|im_end|>"]
        conn._cached_chat_template = None

        mock_lc = MagicMock()

        # Patch both sys.modules (for `from llama_cpp import`) and
        # HAS_LLAMACPP (for guards in tokenize/detokenize)
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {
            "llama_cpp": MagicMock(llama_cpp=mock_lc),
            "llama_cpp.llama_cpp": mock_lc,
        }))
        stack.enter_context(patch("avp.connectors.llamacpp.HAS_LLAMACPP", True))

        return conn, mock_lc, stack

    def test_returns_text_n_past_and_generated_ids(self):
        """generate_on_context returns (text, new_n_past, generated_ids) tuple."""
        conn, mock_lc, stack = self._make_connector_and_lc()

        call_count = [0]
        def mock_sample(sampler, ctx, idx):
            call_count[0] += 1
            if call_count[0] >= 4:
                return 151643
            return 100 + call_count[0]
        mock_lc.llama_sampler_sample.side_effect = mock_sample
        mock_lc.llama_decode.return_value = 0
        conn._model.detokenize.return_value = b"tok"

        with stack:
            text, n_past, ids = conn.generate_on_context(
                MagicMock(), n_past=10, max_tokens=100,
            )

        assert isinstance(text, str)
        assert isinstance(n_past, int)
        assert n_past > 10
        assert isinstance(ids, list)
        assert len(ids) == 3
        assert ids == [101, 102, 103]

    def test_token_callback_fires_for_each_token(self):
        """token_callback receives each generated token piece."""
        conn, mock_lc, stack = self._make_connector_and_lc()

        call_count = [0]
        def mock_sample(sampler, ctx, idx):
            call_count[0] += 1
            if call_count[0] >= 4:
                return 151643
            return 100 + call_count[0]
        mock_lc.llama_sampler_sample.side_effect = mock_sample
        mock_lc.llama_decode.return_value = 0
        conn._model.detokenize.return_value = b"x"

        callback_tokens = []

        with stack:
            text, _, ids = conn.generate_on_context(
                MagicMock(), n_past=0, max_tokens=100,
                token_callback=lambda t: callback_tokens.append(t),
            )

        assert len(callback_tokens) == 3
        assert all(isinstance(t, str) for t in callback_tokens)
        assert len(ids) == len(callback_tokens)

    def test_no_callback_still_works(self):
        """generate_on_context works without token_callback."""
        conn, mock_lc, stack = self._make_connector_and_lc()
        mock_lc.llama_sampler_sample.return_value = 151643

        with stack:
            text, n_past, ids = conn.generate_on_context(
                MagicMock(), n_past=5,
            )

        assert text == ""
        assert n_past == 5
        assert ids == []

    def test_accepts_inference_context(self):
        """generate_on_context accepts LlamaCppInferenceContext."""
        from avp.connectors.llamacpp import LlamaCppInferenceContext

        conn, mock_lc, stack = self._make_connector_and_lc()
        mock_lc.llama_sampler_sample.return_value = 151643

        inf_ctx = LlamaCppInferenceContext(
            ptr=MagicMock(), n_ctx=4096, embeddings=True,
        )

        with stack:
            text, n_past, ids = conn.generate_on_context(inf_ctx, n_past=0)
            inf_ctx._closed = True  # prevent __del__ segfault on mock

        assert mock_lc.llama_sampler_sample.called

    def test_raises_on_closed_context(self):
        """generate_on_context raises ValueError on closed LlamaCppInferenceContext."""
        import pytest
        from avp.connectors.llamacpp import LlamaCppInferenceContext

        conn, mock_lc, stack = self._make_connector_and_lc()
        closed_ctx = LlamaCppInferenceContext(
            ptr=None, n_ctx=4096, embeddings=True, _closed=True,
        )

        with stack:
            with pytest.raises(ValueError, match="closed"):
                conn.generate_on_context(closed_ctx, n_past=0)

    def test_accepts_raw_pointer(self):
        """generate_on_context accepts a raw C pointer."""
        conn, mock_lc, stack = self._make_connector_and_lc()
        mock_lc.llama_sampler_sample.return_value = 151643

        with stack:
            text, n_past, ids = conn.generate_on_context(MagicMock(), n_past=0)

        assert mock_lc.llama_sampler_sample.called

    def test_prompt_decoded_before_generation(self):
        """When prompt is provided, tokens are decoded before generation starts."""
        conn, mock_lc, stack = self._make_connector_and_lc()
        mock_lc.llama_decode.return_value = 0
        mock_lc.llama_sampler_sample.return_value = 151643
        conn._apply_chat_template = MagicMock(return_value=[10, 20, 30])

        with stack:
            text, n_past, ids = conn.generate_on_context(
                MagicMock(), n_past=5, prompt="Hello",
            )

        conn._apply_chat_template.assert_called_once_with("Hello")
        assert n_past == 8

    def test_extra_stop_strings(self):
        """extra_stop_strings adds custom stop conditions."""
        conn, mock_lc, stack = self._make_connector_and_lc()

        call_count = [0]
        def mock_sample(sampler, ctx, idx):
            call_count[0] += 1
            return 200 + call_count[0]
        mock_lc.llama_sampler_sample.side_effect = mock_sample
        mock_lc.llama_decode.return_value = 0

        pieces = ["hel", "lo ", "STOP"]
        piece_idx = [0]
        def mock_detok(ids, special=True):
            p = pieces[min(piece_idx[0], len(pieces) - 1)]
            piece_idx[0] += 1
            return p.encode()
        conn._model.detokenize.side_effect = mock_detok

        with stack:
            text, _, ids = conn.generate_on_context(
                MagicMock(), n_past=0, max_tokens=100,
                extra_stop_strings=["STOP"],
            )

        assert "STOP" not in text

    def test_max_tokens_zero_returns_empty(self):
        """max_tokens=0 produces no output."""
        conn, mock_lc, stack = self._make_connector_and_lc()

        with stack:
            text, n_past, ids = conn.generate_on_context(
                MagicMock(), n_past=10, max_tokens=0,
            )

        assert text == ""
        assert n_past == 10
        assert ids == []
