# Tests for the single-backend LLM module.
#
# External services (Gemini API, Ollama) are mocked: no network calls.

import pytest

import app.llm as llm


class FakeUsageMetadata:
    def __init__(self, prompt=10, candidates=5, total=15):
        self.prompt_eval_count = prompt
        self.eval_count = candidates
        self.total_token_count = total


class FakeMessage:
    """Minimal stand-in for an ollama message."""

    def __init__(self, content):
        self.content = content


class FakeResponse:
    """Minimal stand-in for an ollama ChatResponse."""

    def __init__(self, content="ok", usage=None):
        self.message = FakeMessage(content)
        self.usage = usage


class FakeUsage:
    """Minimal stand-in for an ollama usage object."""

    def __init__(self, prompt=10, candidates=5, total=15):
        self.prompt_eval_count = prompt
        self.eval_count = candidates
        self.total_token_count = total


@pytest.fixture(autouse=True)
def _frozen_config(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemma-4-31b-it")

    # Startup-validation results are cached per configuration in the
    # module; clear them so tests stay independent of each other.
    llm.reset_validation_cache()

    yield

    llm._client = None
    llm._probe_client = None
    llm.reset_validation_cache()
    llm._last_request.update({"duration_ms": None, "usage": None})


def _client_with(chat=lambda **kwargs: FakeResponse(FakeMessage("ok"))):
    """Install a fake ollama.Client with a single chat() call.

    Mirrors the real ollama SDK contract: chat() returns one object
    with a ``message`` and an optional usage payload, and streaming
    chat() returns an iterable of those objects.
    """

    class _FakeMessage:
        def __init__(self, content):
            self.content = content

    class _FakeUsage:
        def __init__(self, prompt=10, candidates=5):
            self.prompt_eval_count = prompt
            self.eval_count = candidates

    class _FakeResponse:
        def __init__(self, content="ok", usage=None):
            self.message = _FakeMessage(content)
            self.usage = usage

    class _FakeClient:
        def chat(self, **kwargs):
            return chat(**kwargs)

        def list(self):
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

        def close(self):
            pass

    llm._client = _FakeClient()
    return llm._client


class _FakeModel:
    name = "models/gemma4:31b"


class _FakeShow:
    def __init__(self):
        self.capabilities = ["vision"]


# --------------------------------------------------
# Text generation
# --------------------------------------------------

def test_generate_response_returns_text_and_metadata():
    _client_with(
        chat=lambda **kwargs: FakeResponse(
            content="hello there",
            usage=FakeUsage(prompt=10, candidates=5),
        )
    )

    answer = llm.generate_response("prompt text")

    assert answer == "hello there"

    metadata = llm.get_last_request_metadata()

    assert metadata["usage"]["input_tokens"] == 10
    assert metadata["usage"]["output_tokens"] == 5
    assert metadata["duration_ms"] is not None


def test_generate_response_without_usage_records_none():
    llm._client = _client_with(
        chat=lambda **kwargs: FakeResponse(content="hi", usage=None)
    )

    llm.generate_response("p")

    assert llm.get_last_request_metadata()["usage"] is None


def test_generate_response_empty_text_raises_llm_error():
    llm._client = _client_with(
        chat=lambda **kwargs: FakeResponse(content="", usage=None)
    )

    with pytest.raises(llm.LLMError, match="empty response"):
        llm.generate_response("p")


def test_generate_response_transport_error_raises_llm_error():
    def _boom(**kwargs):
        raise ConnectionError("down")

    llm._client = _client_with(chat=_boom)

    with pytest.raises(llm.LLMError, match="Text generation failed"):
        llm.generate_response("p")


def test_generate_response_blank_prompt_rejected():
    with pytest.raises(ValueError):
        llm.generate_response("   ")


# --------------------------------------------------
# Streaming
# --------------------------------------------------

def test_stream_yields_text_deltas_and_final_usage():
    chunks = [
        FakeResponse(content="hel"),
        FakeResponse(
            content="lo ",
            usage=FakeUsage(prompt=20, candidates=8),
        ),
    ]

    def stream_factory(**kwargs):
        return iter(chunks)

    class _FakeClient:
        def chat(self, **kwargs):
            return stream_factory(**kwargs)

        def list(self):
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

        def close(self):
            pass

    llm._client = _FakeClient()

    deltas = list(llm.stream_response("p"))

    assert "".join(deltas) == "hello "

    metadata = llm.get_last_request_metadata()

    assert metadata["usage"]["input_tokens"] == 20
    assert metadata["usage"]["output_tokens"] == 8


def test_stream_empty_raises_llm_error():
    class _FakeClient:
        def chat(self, **kwargs):
            return iter([])

        def list(self):
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

        def close(self):
            pass

    llm._client = _FakeClient()

    with pytest.raises(llm.LLMError, match="empty response"):
        list(llm.stream_response("p"))


# --------------------------------------------------
# Multimodal (local Ollama vision model)
# --------------------------------------------------

def test_multimodal_requires_existing_image(tmp_path):
    with pytest.raises(ValueError):
        llm.generate_multimodal_response(
            "describe", [str(tmp_path / "missing.png")]
        )


def test_multimodal_uses_ollama_vision_model(tmp_path, monkeypatch):
    image = tmp_path / "page_1.png"

    image.write_bytes(b"png")

    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        response = type("FakeResponse", (), {})()
        response.message = type("FakeMessage", (), {})()
        response.message.content = "a chart"
        return response

    class _FakeClient:
        def chat(self, **kwargs):
            return fake_chat(**kwargs)

        def list(self):
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

        def close(self):
            pass

    llm.reset_client()
    llm._client = _FakeClient()

    monkeypatch.setenv("OLLAMA_VISION_MODEL", "qwen3.5:4b")

    answer = llm.generate_multimodal_response("describe", [str(image)])

    assert answer == "a chart"
    assert captured.get("model") == "qwen3.5:4b"
    assert captured.get("messages", [{}])[0].get("images") == [str(image)]


def test_multimodal_failure_is_llm_error(tmp_path, monkeypatch):
    class _FakeClient:
        def chat(self, **kwargs):
            raise RuntimeError("connection refused")

        def list(self):
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

        def close(self):
            pass

    llm._client = _FakeClient()

    image = tmp_path / "page_1.png"

    image.write_bytes(b"png")

    with pytest.raises(llm.LLMError, match="Image question answering"):
        llm.generate_multimodal_response("describe", [str(image)])


# --------------------------------------------------
# Configuration and description
# --------------------------------------------------

def test_describe_active_provider_reports_model():
    info = llm.describe_active_provider()

    assert info["provider"] == "ollama"
    assert info["model"] == "gemma4:31b"
    assert info["vision_provider"] == "ollama"


def test_validate_llm_config_reports_unavailable_model(monkeypatch):
    class _Model:
        name = "models/some-other-model"

    class _Models:
        @staticmethod
        def list():
            return [_Model()]

    class _FakeClient:
        def list(self):
            return _Models().list()

        def show(self, model=None):
            return _FakeShow()

        def chat(self, **kwargs):
            return FakeResponse(content="ok")

        def close(self):
            pass

    # The startup probe uses its own short-timeout client.
    monkeypatch.setattr(llm, "_get_probe_client", lambda: _FakeClient())

    validation = llm.validate_llm_config()

    assert validation["configured"] is True
    assert validation["model_available"] is False
    assert "gemma4:31b" in validation["warning"]


def test_validate_llm_config_reports_available_model(monkeypatch):
    # This test intentionally configures the Ollama module only.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_TEXT_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "my-model:latest")

    class _AvailableModel:
        # The backend reports exactly the configured model tag.
        name = "models/my-model"

    class _FakeClient:
        def list(self):
            return [_AvailableModel()]

        def show(self, model=None):
            return _FakeShow()

        def chat(self, **kwargs):
            response = type("FakeResponse", (), {})()
            response.message = type("FakeMessage", (), {})()
            response.message.content = "ok"
            return response

        def close(self):
            pass

    monkeypatch.setattr(llm, "_get_probe_client", lambda: _FakeClient())

    validation = llm.validate_llm_config()

    assert validation["model_available"] is True
    assert validation["warning"] is None


def test_validate_llm_config_caches_a_definitive_result(monkeypatch):
    calls = []

    class _FakeClient:
        def list(self):
            calls.append("list")
            return [_FakeModel()]

        def show(self, model=None):
            return _FakeShow()

    monkeypatch.setattr(llm, "_get_probe_client", lambda: _FakeClient())
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "gemma4:31b")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    first = llm.validate_llm_config()
    second = llm.validate_llm_config()

    # The probe ran once; the second call reused the cached result.
    assert calls == ["list"]
    assert first == second

    # force=True probes again.
    llm.validate_llm_config(force=True)
    assert calls == ["list", "list"]


def test_transient_probe_failure_is_not_cached(monkeypatch):
    calls = []

    class _FailingClient:
        def list(self):
            calls.append("list")
            raise ConnectionError("down")

    monkeypatch.setattr(
        llm, "_get_probe_client", lambda: _FailingClient()
    )

    first = llm.validate_llm_config()
    llm.validate_llm_config()

    assert first["model_available"] is None
    assert "could not be checked" in first["warning"]
    # A transient failure must not be cached: the backend may recover.
    assert calls == ["list", "list"]


def test_probe_client_uses_a_short_timeout(monkeypatch):
    captured = {}

    def fake_build_client(timeout):
        captured["timeout"] = timeout
        return object()

    monkeypatch.setattr(llm, "_build_client", fake_build_client)
    monkeypatch.setattr(llm, "_probe_client", None)
    monkeypatch.setattr(llm, "_probe_client_key", None)

    llm._get_probe_client()

    assert captured["timeout"] == llm.PROBE_TIMEOUT_SECONDS
    assert llm.PROBE_TIMEOUT_SECONDS < llm.DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_request_timeout_is_configurable(monkeypatch):
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)
    assert llm.get_request_timeout() == llm.DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "45")
    assert llm.get_request_timeout() == 45.0

    # An invalid or non-positive value falls back to the default.
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "not-a-number")
    assert llm.get_request_timeout() == llm.DEFAULT_REQUEST_TIMEOUT_SECONDS

    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "0")
    assert llm.get_request_timeout() == llm.DEFAULT_REQUEST_TIMEOUT_SECONDS



def test_validate_llm_config_requires_api_key(monkeypatch):
    # The unit layer must fail closed on a missing/blank model tag.
    monkeypatch.setattr(llm, "get_text_model", lambda: "")

    with pytest.raises(llm.LLMError, match="OLLAMA_TEXT_MODEL cannot be empty"):
        llm.validate_llm_config()
