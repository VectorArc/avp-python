"""Tests that validate SDK against published AVP spec test vectors.

The spec defines exact hex baselines in avp-spec/examples/test-vectors.md.
These tests verify decode compatibility with those baselines, ensuring any
AVP implementation can interoperate with the SDK.

Note: The SDK now writes payload_checksum (proto field 15) on encode, so
SDK-encoded output differs from the pre-checksum spec vectors. These tests
verify DECODE of spec vectors (backward compat) and encode ROUNDTRIP
(new messages include checksum).
"""

import numpy as np
import pytest

import avp
from avp.codec import decode, encode
from avp.types import (
    AVPMetadata,
    CommunicationMode,
    DataType,
    MAGIC,
    PayloadType,
    PROTOCOL_VERSION,
)


# Spec test vector hex strings (from avp-spec/examples/test-vectors.md)
VECTOR_1_HEX = "415601002a0000001a00000012076167656e742d61220a746573742d6d6f64656c28044a0104cdcccc3dcdcc4c3e9a99993ecdcccc3e"
VECTOR_2_HEX = "415601000d0000000500000028024a01020000803f00000040"


class TestSpecVectorDecode:
    """Decode published spec test vectors — backward compatibility."""

    def test_vector1_decodes(self):
        """Vector 1: 4D float32 hidden state with agent/model metadata."""
        data = bytes.fromhex(VECTOR_1_HEX)
        msg = decode(data)

        assert msg.header.magic == MAGIC
        assert msg.header.version == PROTOCOL_VERSION
        assert msg.header.flags == 0
        assert msg.metadata.source_agent_id == "agent-a"
        assert msg.metadata.model_id == "test-model"
        assert msg.metadata.hidden_dim == 4
        assert msg.metadata.payload_type == PayloadType.HIDDEN_STATE
        assert msg.metadata.dtype == DataType.FLOAT32
        assert msg.metadata.tensor_shape == (4,)
        assert msg.raw_size == 54

        # Verify tensor payload
        arr = np.frombuffer(msg.payload, dtype=np.float32)
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)

    def test_vector2_decodes(self):
        """Vector 2: 2D minimal (no model_id, no agent_id)."""
        data = bytes.fromhex(VECTOR_2_HEX)
        msg = decode(data)

        assert msg.header.magic == MAGIC
        assert msg.header.version == PROTOCOL_VERSION
        assert msg.header.flags == 0
        assert msg.metadata.model_id == ""
        assert msg.metadata.source_agent_id == ""
        assert msg.metadata.hidden_dim == 2
        assert msg.metadata.payload_type == PayloadType.HIDDEN_STATE
        assert msg.metadata.dtype == DataType.FLOAT32
        assert msg.metadata.tensor_shape == (2,)
        assert msg.raw_size == 25

        arr = np.frombuffer(msg.payload, dtype=np.float32)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    def test_vector1_no_checksum_skips_verification(self):
        """Old messages without payload_checksum decode without error."""
        data = bytes.fromhex(VECTOR_1_HEX)
        # Should not raise — checksum field is absent, verification skipped
        msg = decode(data)
        assert msg.metadata.model_id == "test-model"

    def test_vector_header_constants(self):
        """Both vectors start with correct magic and version."""
        for hex_str in [VECTOR_1_HEX, VECTOR_2_HEX]:
            data = bytes.fromhex(hex_str)
            assert data[:2] == b"\x41\x56"  # Magic "AV"
            assert data[2] == 0x01           # Version 1


class TestSpecVectorRoundtrip:
    """Encode with current SDK, decode back — verifies checksum works."""

    def test_vector1_roundtrip(self):
        """Encode Vector 1 inputs, decode, verify values match."""
        payload = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
        metadata = AVPMetadata(
            source_agent_id="agent-a",
            model_id="test-model",
            hidden_dim=4,
            payload_type=PayloadType.HIDDEN_STATE,
            dtype=DataType.FLOAT32,
            tensor_shape=(4,),
            mode=CommunicationMode.LATENT,
        )
        data = encode(payload, metadata)
        msg = decode(data)

        assert msg.metadata.source_agent_id == "agent-a"
        assert msg.metadata.model_id == "test-model"
        assert msg.metadata.hidden_dim == 4
        arr = np.frombuffer(msg.payload, dtype=np.float32)
        np.testing.assert_allclose(arr, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)

    def test_vector2_roundtrip(self):
        """Encode Vector 2 inputs, decode, verify values match."""
        payload = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        metadata = AVPMetadata(
            hidden_dim=2,
            payload_type=PayloadType.HIDDEN_STATE,
            dtype=DataType.FLOAT32,
            tensor_shape=(2,),
        )
        data = encode(payload, metadata)
        msg = decode(data)

        assert msg.metadata.hidden_dim == 2
        arr = np.frombuffer(msg.payload, dtype=np.float32)
        np.testing.assert_allclose(arr, [1.0, 2.0])

    def test_roundtrip_includes_checksum(self):
        """SDK-encoded messages include payload_checksum."""
        payload = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        metadata = AVPMetadata(
            hidden_dim=2,
            payload_type=PayloadType.HIDDEN_STATE,
            dtype=DataType.FLOAT32,
            tensor_shape=(2,),
        )
        data = encode(payload, metadata)

        # Decode raw protobuf to verify checksum field is present
        from avp import avp_pb2
        from avp.types import HEADER_SIZE
        import struct

        _, _, _, payload_length, metadata_length = struct.unpack(
            "<2sBBII", data[:HEADER_SIZE]
        )
        meta_pb = avp_pb2.Metadata()
        meta_pb.ParseFromString(data[HEADER_SIZE:HEADER_SIZE + metadata_length])
        assert meta_pb.HasField("payload_checksum")

    def test_corrupted_payload_detected(self):
        """Flipping a bit in the payload triggers checksum mismatch."""
        payload = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        metadata = AVPMetadata(
            hidden_dim=2,
            payload_type=PayloadType.HIDDEN_STATE,
            dtype=DataType.FLOAT32,
            tensor_shape=(2,),
        )
        data = bytearray(encode(payload, metadata))

        # Flip a bit in the last byte of the message (tensor payload area)
        data[-1] ^= 0x01

        with pytest.raises(avp.DecodeError, match="checksum mismatch"):
            decode(bytes(data))
