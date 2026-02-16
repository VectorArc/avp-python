"""AVP binary codec — encode and decode embeddings.

Wire format:
    Bytes 0-1:    Magic (0x4156)
    Byte 2:       Version (0x01)
    Byte 3:       Flags (bit 0 = compressed)
    Bytes 4-7:    Payload length (uint32 LE) — metadata + embedding bytes
    Bytes 8-11:   Metadata length (uint32 LE)
    Bytes 12..N:  Protobuf-encoded Metadata
    Bytes N..:    Raw embedding bytes (optionally zstd-compressed)
"""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as np

from . import avp_pb2
from .compression import compress, decompress
from .errors import DecodeError, InvalidMagicError, UnsupportedVersionError
from .types import (
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPHeader,
    AVPMessage,
    AVPMetadata,
    CompressionLevel,
)
from .utils import bytes_to_embedding, embedding_to_bytes

# struct format: 2s magic, B version, B flags, I payload_len, I metadata_len
_HEADER_FMT = "<2sBBII"


def encode(
    embedding: np.ndarray,
    model_id: str = "",
    data_type: Optional[str] = None,
    compression: CompressionLevel = CompressionLevel.NONE,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> bytes:
    """Encode a numpy embedding into an AVP binary message.

    Args:
        embedding: 1-D numpy array (float32 or float16).
        model_id: Identifier of the model that produced the embedding.
        data_type: Override dtype string; defaults to embedding.dtype.
        compression: Compression level (NONE, FAST, BALANCED, MAX).
        agent_id: Optional sender agent ID.
        task_id: Optional correlation ID.
        extra: Optional extra key-value metadata.

    Returns:
        Raw bytes of the AVP message.
    """
    if embedding.ndim != 1:
        raise ValueError(f"Embedding must be 1-D, got {embedding.ndim}-D")

    dtype_str = data_type or str(embedding.dtype)

    # Build protobuf metadata
    meta_pb = avp_pb2.Metadata(
        model_id=model_id,
        embedding_dim=embedding.shape[0],
        data_type=dtype_str,
    )
    if compression != CompressionLevel.NONE:
        meta_pb.compression = "zstd"
    if agent_id:
        meta_pb.agent_id = agent_id
    if task_id:
        meta_pb.task_id = task_id
    if extra:
        for k, v in extra.items():
            meta_pb.extra[k] = v

    meta_bytes = meta_pb.SerializeToString()

    # Embedding → raw bytes, optionally compressed
    emb_bytes = embedding_to_bytes(embedding)
    if compression != CompressionLevel.NONE:
        emb_bytes = compress(emb_bytes, level=compression)

    # Payload = metadata + embedding
    payload_length = len(meta_bytes) + len(emb_bytes)

    # Flags
    flags = 0x01 if compression != CompressionLevel.NONE else 0x00

    # Pack header
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        PROTOCOL_VERSION,
        flags,
        payload_length,
        len(meta_bytes),
    )

    return header + meta_bytes + emb_bytes


def decode(data: bytes) -> AVPMessage:
    """Decode an AVP binary message into an AVPMessage.

    Args:
        data: Raw bytes of an AVP message.

    Returns:
        Decoded AVPMessage with header, metadata, and embedding.

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

    # Parse metadata
    meta_start = HEADER_SIZE
    meta_end = meta_start + metadata_length
    meta_pb = avp_pb2.Metadata()
    meta_pb.ParseFromString(data[meta_start:meta_end])

    metadata = AVPMetadata(
        model_id=meta_pb.model_id,
        embedding_dim=meta_pb.embedding_dim,
        data_type=meta_pb.data_type,
        compression=meta_pb.compression if meta_pb.HasField("compression") else None,
        agent_id=meta_pb.agent_id if meta_pb.HasField("agent_id") else None,
        task_id=meta_pb.task_id if meta_pb.HasField("task_id") else None,
        extra=dict(meta_pb.extra),
    )

    # Extract embedding bytes
    emb_bytes = data[meta_end:meta_start + payload_length]

    # Decompress if needed
    is_compressed = bool(flags & 0x01)
    if is_compressed:
        emb_bytes = decompress(emb_bytes)

    embedding = bytes_to_embedding(emb_bytes, metadata.data_type, metadata.embedding_dim)

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
        embedding=embedding,
        raw_size=len(data),
    )


# --- Convenience wrappers ---


def encode_simple(
    embedding: np.ndarray,
    model_id: str = "",
    compress: bool = False,
) -> bytes:
    """Simplified encode: just embedding + model_id + optional compression."""
    level = CompressionLevel.BALANCED if compress else CompressionLevel.NONE
    return encode(embedding, model_id=model_id, compression=level)


def decode_simple(data: bytes) -> Tuple[np.ndarray, dict]:
    """Simplified decode: returns (embedding, metadata_dict)."""
    msg = decode(data)
    meta_dict = {
        "model_id": msg.metadata.model_id,
        "embedding_dim": msg.metadata.embedding_dim,
        "data_type": msg.metadata.data_type,
    }
    if msg.metadata.compression:
        meta_dict["compression"] = msg.metadata.compression
    if msg.metadata.agent_id:
        meta_dict["agent_id"] = msg.metadata.agent_id
    if msg.metadata.task_id:
        meta_dict["task_id"] = msg.metadata.task_id
    if msg.metadata.extra:
        meta_dict["extra"] = msg.metadata.extra
    return msg.embedding, meta_dict
