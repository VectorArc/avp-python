"""Tests for AVP error types and hierarchy."""

import pytest

from avp.errors import (
    AVPError,
    DecodeError,
    EngineNotAvailableError,
    FallbackRequested,
    HandshakeError,
    IncompatibleModelsError,
    InvalidMagicError,
    RealignmentError,
    SessionError,
    SessionExpiredError,
    ShapeMismatchError,
    TransportError,
    UnsupportedVersionError,
)


# --- Hierarchy ---


def test_all_errors_inherit_from_avp_error():
    errors = [
        InvalidMagicError(b"\x00\x00"),
        UnsupportedVersionError(99),
        DecodeError("bad"),
        TransportError("fail", 500),
        HandshakeError("no"),
        SessionError("expired"),
        SessionExpiredError("sess-1"),
        ShapeMismatchError((10,), (20,)),
        RealignmentError("oops"),
        FallbackRequested("reason", 10.0),
        IncompatibleModelsError("mismatch"),
        EngineNotAvailableError("vllm"),
    ]
    for err in errors:
        assert isinstance(err, AVPError)


def test_session_expired_is_session_error():
    err = SessionExpiredError("s1")
    assert isinstance(err, SessionError)


def test_incompatible_models_is_handshake_error():
    err = IncompatibleModelsError("diff arch")
    assert isinstance(err, HandshakeError)


# --- Error attributes ---


def test_invalid_magic_attributes():
    err = InvalidMagicError(b"\xde\xad")
    assert err.got == b"\xde\xad"
    assert "dead" in str(err)


def test_unsupported_version_attributes():
    err = UnsupportedVersionError(99)
    assert err.version == 99


def test_transport_error_attributes():
    err = TransportError("fail", status_code=503)
    assert err.status_code == 503


def test_session_expired_attributes():
    err = SessionExpiredError("sess-abc")
    assert err.session_id == "sess-abc"
    assert "sess-abc" in str(err)


def test_shape_mismatch_attributes():
    err = ShapeMismatchError((10, 20), (10, 30))
    assert err.expected == (10, 20)
    assert err.got == (10, 30)


def test_fallback_requested_attributes():
    err = FallbackRequested("high perplexity", perplexity_score=15.5)
    assert err.reason == "high perplexity"
    assert err.perplexity_score == 15.5
    assert "15.5" in str(err)


def test_engine_not_available_attributes():
    err = EngineNotAvailableError("vllm")
    assert err.engine == "vllm"
    assert "vllm" in str(err)
