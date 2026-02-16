"""AVP protocol errors."""

from __future__ import annotations


class AVPError(Exception):
    """Base error for all AVP operations."""


class InvalidMagicError(AVPError):
    """Raised when the magic bytes don't match 0x4156."""

    def __init__(self, got: bytes):
        self.got = got
        super().__init__(f"Invalid magic bytes: expected 0x4156, got {got.hex()}")


class UnsupportedVersionError(AVPError):
    """Raised when the protocol version is not supported."""

    def __init__(self, version: int):
        self.version = version
        super().__init__(f"Unsupported protocol version: {version}")


class DecodeError(AVPError):
    """Raised when a message cannot be decoded."""


class TransportError(AVPError):
    """Raised on transport-layer failures (HTTP errors, timeouts)."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)
