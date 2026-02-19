"""Zstandard compression wrappers for AVP."""

import zstandard as zstd

from .types import CompressionLevel

# Pre-built compressors for each level
_COMPRESSORS = {
    CompressionLevel.FAST: zstd.ZstdCompressor(level=CompressionLevel.FAST.value),
    CompressionLevel.BALANCED: zstd.ZstdCompressor(level=CompressionLevel.BALANCED.value),
    CompressionLevel.MAX: zstd.ZstdCompressor(level=CompressionLevel.MAX.value),
}

_DECOMPRESSOR = zstd.ZstdDecompressor()


def compress(data: bytes, level: CompressionLevel = CompressionLevel.BALANCED) -> bytes:
    """Compress data using zstd at the given level."""
    if level == CompressionLevel.NONE:
        return data
    return _COMPRESSORS[level].compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress zstd-compressed data."""
    return _DECOMPRESSOR.decompress(data)
