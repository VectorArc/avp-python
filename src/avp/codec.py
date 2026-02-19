"""AVP binary codec — encode and decode latent payloads.

Wire format:
    Bytes 0-1:    Magic (0x4156)
    Byte 2:       Version (0x01)
    Byte 3:       Flags (bit 0=compressed, bit 1=hybrid, bit 2=has_map, bit 3=kv_cache)
    Bytes 4-7:    Payload length (uint32 LE) — metadata + tensor bytes
    Bytes 8-11:   Metadata length (uint32 LE)
    Bytes 12..N:  Protobuf-encoded Metadata
    Bytes N..:    Raw tensor bytes (optionally zstd-compressed)
"""

import struct
from typing import Optional

import numpy as np

from . import avp_pb2
from .compression import compress, decompress
from .errors import DecodeError, InvalidMagicError, UnsupportedVersionError
from .types import (
    FLAG_COMPRESSED,
    FLAG_HAS_MAP,
    FLAG_HYBRID,
    FLAG_KV_CACHE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPHeader,
    AVPMessage,
    AVPMetadata,
    CommunicationMode,
    CompressionLevel,
    DataType,
    PayloadType,
)
from .utils import embedding_to_bytes

# struct format: 2s magic, B version, B flags, I payload_len, I metadata_len
_HEADER_FMT = "<2sBBII"


def encode(
    payload: bytes,
    metadata: AVPMetadata,
    compression: CompressionLevel = CompressionLevel.NONE,
) -> bytes:
    """Encode a binary payload + metadata into an AVP message.

    Args:
        payload: Raw tensor bytes (hidden state, KV-cache, or embedding).
        metadata: AVPMetadata with all fields set.
        compression: Compression level for the payload.

    Returns:
        Raw bytes of the AVP message.
    """
    # Build protobuf metadata
    meta_pb = avp_pb2.Metadata(
        session_id=metadata.session_id,
        source_agent_id=metadata.source_agent_id,
        target_agent_id=metadata.target_agent_id,
        model_id=metadata.model_id,
        hidden_dim=metadata.hidden_dim,
        num_layers=metadata.num_layers,
        payload_type=int(metadata.payload_type),
        dtype=int(metadata.dtype),
        tensor_shape=list(metadata.tensor_shape),
        mode=int(metadata.mode),
        confidence_score=metadata.confidence_score,
        avp_map_id=metadata.avp_map_id,
    )
    if compression != CompressionLevel.NONE:
        meta_pb.compression = "zstd"
    if metadata.extra:
        for k, v in metadata.extra.items():
            meta_pb.extra[k] = v

    meta_bytes = meta_pb.SerializeToString()

    # Payload, optionally compressed
    payload_bytes = payload
    if compression != CompressionLevel.NONE:
        payload_bytes = compress(payload_bytes, level=compression)

    # Total payload = metadata + tensor bytes
    total_payload_length = len(meta_bytes) + len(payload_bytes)

    # Flags
    flags = 0
    if compression != CompressionLevel.NONE:
        flags |= FLAG_COMPRESSED
    if metadata.mode == CommunicationMode.HYBRID:
        flags |= FLAG_HYBRID
    if metadata.avp_map_id:
        flags |= FLAG_HAS_MAP
    if metadata.payload_type == PayloadType.KV_CACHE:
        flags |= FLAG_KV_CACHE

    # Pack header
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        PROTOCOL_VERSION,
        flags,
        total_payload_length,
        len(meta_bytes),
    )

    return header + meta_bytes + payload_bytes


def decode(data: bytes) -> AVPMessage:
    """Decode an AVP binary message into an AVPMessage.

    Args:
        data: Raw bytes of an AVP message.

    Returns:
        Decoded AVPMessage with header, metadata, and payload.

    Raises:
        InvalidMagicError: If magic bytes don't match.
        UnsupportedVersionError: If protocol version is unknown.
        DecodeError: If the message is malformed.
    """
    if len(data) < HEADER_SIZE:
        raise DecodeError(f"Message too short: {len(data)} bytes, need at least {HEADER_SIZE}")

    magic, version, flags, payload_length, metadata_length = struct.unpack(
        _HEADER_FMT, data[:HEADER_SIZE]
    )

    if magic != MAGIC:
        raise InvalidMagicError(magic)
    if version != PROTOCOL_VERSION:
        raise UnsupportedVersionError(version)

    expected = HEADER_SIZE + payload_length
    if len(data) < expected:
        raise DecodeError(
            f"Message truncated: expected {expected} bytes, got {len(data)}"
        )

    # Parse protobuf metadata
    meta_start = HEADER_SIZE
    meta_end = meta_start + metadata_length
    meta_pb = avp_pb2.Metadata()
    meta_pb.ParseFromString(data[meta_start:meta_end])

    # Extract raw payload bytes
    raw_payload = data[meta_end:meta_start + payload_length]

    # Decompress if needed
    is_compressed = bool(flags & FLAG_COMPRESSED)
    if is_compressed:
        raw_payload = decompress(raw_payload)

    dtype_enum = DataType(meta_pb.dtype) if meta_pb.dtype in DataType.__members__.values() else DataType.FLOAT32

    metadata = AVPMetadata(
        session_id=meta_pb.session_id,
        source_agent_id=meta_pb.source_agent_id,
        target_agent_id=meta_pb.target_agent_id,
        model_id=meta_pb.model_id,
        hidden_dim=meta_pb.hidden_dim,
        num_layers=meta_pb.num_layers,
        payload_type=PayloadType(meta_pb.payload_type),
        dtype=dtype_enum,
        tensor_shape=tuple(meta_pb.tensor_shape),
        mode=CommunicationMode(meta_pb.mode),
        compression="zstd" if is_compressed else None,
        confidence_score=meta_pb.confidence_score,
        avp_map_id=meta_pb.avp_map_id,
        extra=dict(meta_pb.extra),
    )

    header = AVPHeader(
        magic=magic,
        version=version,
        flags=flags,
        payload_length=payload_length,
        metadata_length=metadata_length,
    )

    return AVPMessage(
        header=header,
        metadata=metadata,
        payload=raw_payload,
        raw_size=len(data),
    )


# --- Convenience encoders ---


def encode_hidden_state(
    hidden_state: np.ndarray,
    metadata: AVPMetadata,
    compression: CompressionLevel = CompressionLevel.NONE,
) -> bytes:
    """Encode a hidden state tensor into an AVP message."""
    metadata.payload_type = PayloadType.HIDDEN_STATE
    metadata.tensor_shape = hidden_state.shape
    payload = embedding_to_bytes(hidden_state)
    return encode(payload, metadata, compression)


def encode_kv_cache(
    kv_bytes: bytes,
    metadata: AVPMetadata,
    compression: CompressionLevel = CompressionLevel.NONE,
) -> bytes:
    """Encode serialized KV-cache bytes into an AVP message."""
    metadata.payload_type = PayloadType.KV_CACHE
    return encode(kv_bytes, metadata, compression)
