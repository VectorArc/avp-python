"""Tests for OllamaConnector (mock-based, no Ollama or model required)."""

import json
import os
from unittest import mock
from urllib.error import URLError

import pytest


# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------


class TestParseModelName:
    """Test _parse_model_name helper."""

    def test_bare_name(self):
        from avp.connectors.ollama import _parse_model_name
        assert _parse_model_name("qwen2.5") == ("qwen2.5", "latest")

    def test_name_with_tag(self):
        from avp.connectors.ollama import _parse_model_name
        assert _parse_model_name("qwen2.5:7b") == ("qwen2.5", "7b")

    def test_name_with_latest_tag(self):
        from avp.connectors.ollama import _parse_model_name
        assert _parse_model_name("qwen2.5:latest") == ("qwen2.5", "latest")

    def test_fully_qualified_name(self):
        from avp.connectors.ollama import _parse_model_name
        name, tag = _parse_model_name(
            "registry.ollama.ai/library/qwen2.5:7b",
        )
        assert name == "qwen2.5"
        assert tag == "7b"

    def test_library_prefix(self):
        from avp.connectors.ollama import _parse_model_name
        name, tag = _parse_model_name("library/llama3.2:3b")
        assert name == "llama3.2"
        assert tag == "3b"

    def test_complex_tag(self):
        from avp.connectors.ollama import _parse_model_name
        name, tag = _parse_model_name("qwen2.5:7b-instruct-q4_K_M")
        assert name == "qwen2.5"
        assert tag == "7b-instruct-q4_K_M"


# ---------------------------------------------------------------------------
# Model resolution (filesystem, no HTTP)
# ---------------------------------------------------------------------------


def _create_fake_ollama_store(tmp_path, model_name, tag, digest):
    """Create a minimal Ollama model store structure for testing."""
    models_dir = tmp_path / "models"
    manifest_dir = (
        models_dir / "manifests" / "registry.ollama.ai" / "library"
        / model_name
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{digest}",
                "size": 12345,
            },
            {
                "mediaType": "application/vnd.ollama.image.template",
                "digest": "sha256:template123",
                "size": 100,
            },
        ],
    }
    manifest_file = manifest_dir / tag
    manifest_file.write_text(json.dumps(manifest))

    # Create blob
    blobs_dir = models_dir / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    blob_file = blobs_dir / f"sha256-{digest}"
    blob_file.write_bytes(b"fake gguf data")

    return models_dir, str(blob_file.resolve())


class TestResolveOllamaModel:
    """Test resolve_ollama_model with fake filesystem."""

    def test_basic_resolution(self, tmp_path):
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, expected_path = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "abc123",
        )
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = resolve_ollama_model("qwen2.5:7b")
        assert result == expected_path

    def test_default_tag_latest(self, tmp_path):
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, expected_path = _create_fake_ollama_store(
            tmp_path, "llama3.2", "latest", "def456",
        )
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = resolve_ollama_model("llama3.2")
        assert result == expected_path

    def test_digest_dash_not_colon(self, tmp_path):
        """Verify sha256:hash in manifest maps to sha256-hash filename."""
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, expected_path = _create_fake_ollama_store(
            tmp_path, "test", "latest", "deadbeef",
        )
        # The blob filename should use dash, not colon
        assert "sha256-deadbeef" in expected_path
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = resolve_ollama_model("test")
        assert result == expected_path

    def test_model_not_downloaded(self, tmp_path):
        from avp.connectors.ollama import resolve_ollama_model

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with pytest.raises(FileNotFoundError, match="ollama pull"):
                resolve_ollama_model("nonexistent:latest")

    def test_missing_blob(self, tmp_path):
        """Manifest exists but blob file is missing."""
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, blob_path = _create_fake_ollama_store(
            tmp_path, "broken", "latest", "missing123",
        )
        # Delete the blob
        os.unlink(blob_path)
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with pytest.raises(FileNotFoundError, match="corrupted"):
                resolve_ollama_model("broken")

    def test_no_model_layer_in_manifest(self, tmp_path):
        """Manifest has no layer with the model media type."""
        from avp.connectors.ollama import resolve_ollama_model

        models_dir = tmp_path / "models"
        manifest_dir = (
            models_dir / "manifests" / "registry.ollama.ai" / "library"
            / "empty"
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "layers": [
                {"mediaType": "application/vnd.ollama.image.template"},
            ],
        }
        (manifest_dir / "latest").write_text(json.dumps(manifest))

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with pytest.raises(ValueError, match="No model layer"):
                resolve_ollama_model("empty")

    def test_custom_models_dir(self, tmp_path):
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, expected_path = _create_fake_ollama_store(
            tmp_path, "custom_model", "v1", "custom789",
        )
        # Override models_dir location
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = resolve_ollama_model("custom_model:v1")
        assert result == expected_path

    def test_fully_qualified_name(self, tmp_path):
        from avp.connectors.ollama import resolve_ollama_model

        models_dir, expected_path = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "fqn999",
        )
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = resolve_ollama_model(
                "registry.ollama.ai/library/qwen2.5:7b",
            )
        assert result == expected_path


# ---------------------------------------------------------------------------
# Ollama server interaction
# ---------------------------------------------------------------------------


class TestOllamaHost:
    """Test _get_ollama_host helper."""

    def test_default_host(self):
        from avp.connectors.ollama import _get_ollama_host

        with mock.patch.dict(os.environ, {}, clear=False):
            # Remove OLLAMA_HOST if set
            os.environ.pop("OLLAMA_HOST", None)
            assert _get_ollama_host() == "http://127.0.0.1:11434"

    def test_custom_host(self):
        from avp.connectors.ollama import _get_ollama_host

        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "http://gpu:8080"}):
            assert _get_ollama_host() == "http://gpu:8080"

    def test_bare_host_gets_http_prefix(self):
        from avp.connectors.ollama import _get_ollama_host

        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "gpu-server:11434"}):
            assert _get_ollama_host() == "http://gpu-server:11434"

    def test_trailing_slash_stripped(self):
        from avp.connectors.ollama import _get_ollama_host

        with mock.patch.dict(
            os.environ, {"OLLAMA_HOST": "http://localhost:11434/"},
        ):
            assert _get_ollama_host() == "http://localhost:11434"


class TestIsOllamaRunning:
    """Test is_ollama_running with mocked HTTP."""

    def test_running(self):
        from avp.connectors.ollama import is_ollama_running

        with mock.patch("avp.connectors.ollama.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = mock.Mock()
            mock_urlopen.return_value.__exit__ = mock.Mock(
                return_value=False,
            )
            assert is_ollama_running() is True

    def test_not_running(self):
        from avp.connectors.ollama import is_ollama_running

        with mock.patch("avp.connectors.ollama.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            assert is_ollama_running() is False


class TestIsModelLoaded:
    """Test is_model_loaded with mocked HTTP."""

    def _mock_ps_response(self, models):
        """Create a mock /api/ps response."""
        data = json.dumps({"models": models}).encode()
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = data
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        return mock_resp

    def test_model_loaded(self):
        from avp.connectors.ollama import is_model_loaded

        resp = self._mock_ps_response([{"name": "qwen2.5:7b"}])
        with mock.patch("avp.connectors.ollama.urlopen", return_value=resp):
            assert is_model_loaded("qwen2.5:7b") is True

    def test_model_not_loaded(self):
        from avp.connectors.ollama import is_model_loaded

        resp = self._mock_ps_response([{"name": "llama3.2:3b"}])
        with mock.patch("avp.connectors.ollama.urlopen", return_value=resp):
            assert is_model_loaded("qwen2.5:7b") is False

    def test_no_models_loaded(self):
        from avp.connectors.ollama import is_model_loaded

        resp = self._mock_ps_response([])
        with mock.patch("avp.connectors.ollama.urlopen", return_value=resp):
            assert is_model_loaded("qwen2.5:7b") is False

    def test_server_not_running(self):
        from avp.connectors.ollama import is_model_loaded

        with mock.patch("avp.connectors.ollama.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            assert is_model_loaded("qwen2.5:7b") is False


class TestUnloadModel:
    """Test unload_model with mocked HTTP."""

    def test_unload_success(self):
        from avp.connectors.ollama import unload_model

        mock_resp = mock.MagicMock()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("avp.connectors.ollama.urlopen", return_value=mock_resp):
            assert unload_model("qwen2.5:7b") is True

    def test_unload_server_down(self):
        from avp.connectors.ollama import unload_model

        with mock.patch("avp.connectors.ollama.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            assert unload_model("qwen2.5:7b") is False


# ---------------------------------------------------------------------------
# Connector class
# ---------------------------------------------------------------------------


class TestOllamaConnectorInit:
    """Test OllamaConnector initialization (mocked LlamaCppConnector)."""

    def test_resolves_and_delegates(self, tmp_path):
        """Verify init resolves name and calls super().__init__ with path."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, blob_path = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "init123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ) as mock_init:
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                    return_value=False,
                ):
                    connector = OllamaConnector("qwen2.5:7b")

        mock_init.assert_called_once_with(
            model_path=blob_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        assert connector._ollama_model_name == "qwen2.5:7b"

    def test_auto_unload_when_loaded(self, tmp_path):
        """Verify auto-unload is called when server running + model loaded."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "unload123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ):
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                    return_value=True,
                ):
                    with mock.patch(
                        "avp.connectors.ollama.is_model_loaded",
                        return_value=True,
                    ) as mock_loaded:
                        with mock.patch(
                            "avp.connectors.ollama.unload_model",
                        ) as mock_unload:
                            OllamaConnector("qwen2.5:7b")

        mock_loaded.assert_called_once_with("qwen2.5:7b")
        mock_unload.assert_called_once_with("qwen2.5:7b")

    def test_no_unload_when_disabled(self, tmp_path):
        """Verify auto_unload=False skips server interaction."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "nounload123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ):
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                ) as mock_running:
                    OllamaConnector("qwen2.5:7b", auto_unload=False)

        mock_running.assert_not_called()

    def test_no_unload_when_server_down(self, tmp_path):
        """Verify graceful skip when Ollama server is not running."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "noserver123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ):
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                    return_value=False,
                ):
                    with mock.patch(
                        "avp.connectors.ollama.unload_model",
                    ) as mock_unload:
                        OllamaConnector("qwen2.5:7b")

        mock_unload.assert_not_called()

    def test_model_not_found_raises(self, tmp_path):
        """Verify clear error when model not downloaded."""
        from avp.connectors.ollama import OllamaConnector

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with pytest.raises(FileNotFoundError, match="ollama pull"):
                OllamaConnector("nonexistent:latest")

    def test_from_ollama_factory(self, tmp_path):
        """Verify from_ollama class method works."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "test", "latest", "factory123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ):
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                    return_value=False,
                ):
                    connector = OllamaConnector.from_ollama("test")

        assert connector._ollama_model_name == "test"

    def test_inherits_can_think(self, tmp_path):
        """Verify can_think is inherited from LlamaCppConnector."""
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "test", "latest", "think123",
        )

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            with mock.patch.object(
                OllamaConnector.__bases__[0], "__init__", return_value=None,
            ):
                with mock.patch(
                    "avp.connectors.ollama.is_ollama_running",
                    return_value=False,
                ):
                    connector = OllamaConnector("test")

        assert connector.can_think is True


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    """Test OllamaConnector.list_models."""

    def test_list_models(self, tmp_path):
        from avp.connectors.ollama import OllamaConnector

        models_dir, _ = _create_fake_ollama_store(
            tmp_path, "qwen2.5", "7b", "list1",
        )
        _create_fake_ollama_store(tmp_path, "llama3.2", "3b", "list2")

        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            result = OllamaConnector.list_models()

        names = [r["name"] for r in result]
        assert "llama3.2:3b" in names
        assert "qwen2.5:7b" in names

    def test_list_models_empty(self, tmp_path):
        from avp.connectors.ollama import OllamaConnector

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with mock.patch.dict(os.environ, {"OLLAMA_MODELS": str(models_dir)}):
            assert OllamaConnector.list_models() == []
