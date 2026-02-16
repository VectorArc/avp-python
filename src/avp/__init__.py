"""AVP — Agent Vector Protocol Python SDK."""

from .codec import decode, decode_simple, encode, encode_simple
from .compression import compress, decompress
from .errors import AVPError, DecodeError, InvalidMagicError, TransportError, UnsupportedVersionError
from .transport import AVPAsyncClient, AVPClient, create_app
from .types import (
    AVP_VERSION_HEADER,
    CONTENT_TYPE,
    HEADER_SIZE,
    MAGIC,
    PROTOCOL_VERSION,
    AVPHeader,
    AVPMessage,
    AVPMetadata,
    CompressionLevel,
    DataType,
)
from .version import __version__

__all__ = [
    # Codec
    "encode",
    "decode",
    "encode_simple",
    "decode_simple",
    # Compression
    "compress",
    "decompress",
    # Transport
    "AVPClient",
    "AVPAsyncClient",
    "create_app",
    # Types
    "AVPHeader",
    "AVPMessage",
    "AVPMetadata",
    "CompressionLevel",
    "DataType",
    # Constants
    "MAGIC",
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "CONTENT_TYPE",
    "AVP_VERSION_HEADER",
    # Errors
    "AVPError",
    "InvalidMagicError",
    "UnsupportedVersionError",
    "DecodeError",
    "TransportError",
    # Version
    "__version__",
]
