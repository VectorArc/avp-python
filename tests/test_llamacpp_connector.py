"""Tests for LlamaCppConnector (mock-based, no model required)."""

import ctypes

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


class TestCompatModule:
    """Test _llamacpp_compat ctypes helpers (no llama-cpp-python needed)."""

    def test_tensor_name_extraction(self):
        """get_tensor_name should extract name from a mock tensor struct."""
        from avp.connectors._llamacpp_compat import (
            _TENSOR_NAME_OFFSET,
            get_tensor_name,
        )

        # Create a fake buffer simulating a ggml_tensor struct
        buf = bytearray(320)  # Large enough for the struct
        name = b"l_out-15\x00"
        buf[_TENSOR_NAME_OFFSET:_TENSOR_NAME_OFFSET + len(name)] = name

        # Get the address of the buffer
        c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
        addr = ctypes.addressof(c_buf)

        assert get_tensor_name(addr) == "l_out-15"

    def test_tensor_shape_extraction(self):
        """get_tensor_shape should extract ne[0..3] from a mock tensor."""
        from avp.connectors._llamacpp_compat import (
            _TENSOR_NE_OFFSET,
            get_tensor_shape,
        )

        buf = bytearray(320)
        # Write ne[0]=4096, ne[1]=128, ne[2]=1, ne[3]=1
        import struct
        struct.pack_into("<qqqq", buf, _TENSOR_NE_OFFSET, 4096, 128, 1, 1)

        c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
        addr = ctypes.addressof(c_buf)

        shape = get_tensor_shape(addr)
        assert shape == (4096, 128, 1, 1)

    def test_tensor_data_ptr(self):
        """get_tensor_data_ptr should extract the data pointer."""
        from avp.connectors._llamacpp_compat import (
            _TENSOR_DATA_OFFSET,
            get_tensor_data_ptr,
        )

        buf = bytearray(320)
        # Write a fake data pointer at the data offset
        import struct
        struct.pack_into("<Q", buf, _TENSOR_DATA_OFFSET, 0xDEADBEEF)

        c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
        addr = ctypes.addressof(c_buf)

        assert get_tensor_data_ptr(addr) == 0xDEADBEEF

    def test_tensor_type(self):
        """get_tensor_type should extract ggml_type enum."""
        from avp.connectors._llamacpp_compat import (
            _TENSOR_TYPE_OFFSET,
            GGML_TYPE_F32,
            get_tensor_type,
        )

        buf = bytearray(320)
        import struct
        struct.pack_into("<i", buf, _TENSOR_TYPE_OFFSET, GGML_TYPE_F32)

        c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
        addr = ctypes.addressof(c_buf)

        assert get_tensor_type(addr) == GGML_TYPE_F32

    def test_batch_embd_offset(self):
        """set_batch_embeddings should write to the correct offset."""
        from avp.connectors._llamacpp_compat import (
            _BATCH_EMBD_OFFSET,
            set_batch_embeddings,
        )

        buf = bytearray(64)
        c_buf = (ctypes.c_char * len(buf)).from_buffer(buf)
        addr = ctypes.addressof(c_buf)

        # Set embd pointer
        set_batch_embeddings(addr, 0x12345678)

        # Read it back
        import struct
        embd_ptr = struct.unpack_from("<Q", buf, _BATCH_EMBD_OFFSET)[0]
        assert embd_ptr == 0x12345678

    def test_eval_callback_type_none_without_llamacpp(self):
        """EVAL_CALLBACK_TYPE should be None when llama-cpp-python not installed."""
        from avp.connectors._llamacpp_compat import HAS_LLAMACPP
        if not HAS_LLAMACPP:
            from avp.connectors._llamacpp_compat import EVAL_CALLBACK_TYPE
            assert EVAL_CALLBACK_TYPE is None
