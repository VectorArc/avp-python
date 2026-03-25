"""AVP protocol errors."""

from typing import Optional


class AVPError(Exception):
    """Base error for all AVP operations."""


class ConfigurationError(AVPError):
    """Raised for invalid configuration or arguments to AVP functions."""


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

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class HandshakeError(AVPError):
    """Raised when handshake negotiation fails."""


class SessionError(AVPError):
    """Raised on session-related failures."""


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session expired: {session_id}")


class ShapeMismatchError(AVPError):
    """Raised when tensor shapes don't match expectations."""

    def __init__(self, expected: tuple, got: tuple):
        self.expected = expected
        self.got = got
        super().__init__(f"Shape mismatch: expected {expected}, got {got}")


class ProjectionError(AVPError):
    """Raised when cross-model projection or realignment fails."""


# Deprecated alias — use ProjectionError instead
RealignmentError = ProjectionError



class IncompatibleModelsError(HandshakeError):
    """Raised when models are incompatible for latent communication."""


class EngineNotAvailableError(AVPError):
    """Raised when a required engine backend is not available."""

    def __init__(self, engine: str):
        self.engine = engine
        super().__init__(f"Engine not available: {engine}")
