"""Tests for the shared Gemini generation layer.

All provider calls are mocked; these tests do not access external services.
"""

import base64
import sys
from types import SimpleNamespace

import pytest

import app.llm as llm


class FakeMessage:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class FakeChatModel:
    def __init__(self, response=None, chunks=(), error=None):
        self.response = response or FakeMessage("ok")
        self.chunks = list(chunks)
        self.error = error
        self.invocations = []
        self.streamed = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.error:
            raise self.error
        return self.response

    def stream(self, messages):
        self.streamed.append(messages)
        if self.error:
            raise self.error
        yield from self.chunks


@pytest.fixture(autouse=True)
def _configured_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-test-model")
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    llm.reset_client()
    yield
    llm.reset_client()


def test_generate_response_sends_system_and_user_and_records_usage(monkeypatch):
    model = FakeChatModel(
        FakeMessage(
            "hello there",
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)

    assert llm.generate_response("prompt text", system="Be concise") == "hello there"
    messages = model.invocations[0]
    assert [message.type for message in messages] == ["system", "human"]
    assert messages[0].content == "Be concise"
    assert messages[1].content == "prompt text"
    assert llm.get_last_request_metadata()["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_generate_response_without_usage_records_none(monkeypatch):
    monkeypatch.setattr(
        llm,
        "get_chat_model",
        lambda: FakeChatModel(FakeMessage("hi")),
    )

    assert llm.generate_response("prompt") == "hi"
    assert llm.get_last_request_metadata()["usage"] is None


def test_generate_response_normalizes_masked_tokens(monkeypatch):
    monkeypatch.setattr(
        llm,
        "get_chat_model",
        lambda: FakeChatModel(FakeMessage("Contact [PERSON_5] at [EMAIL_1].")),
    )

    assert llm.generate_response("prompt") == (
        "Contact Person E at Email address A."
    )


def test_generate_response_rejects_empty_result_and_blank_prompt(monkeypatch):
    monkeypatch.setattr(
        llm,
        "get_chat_model",
        lambda: FakeChatModel(FakeMessage("  ")),
    )
    with pytest.raises(llm.LLMError, match="empty response"):
        llm.generate_response("prompt")

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        llm.generate_response(" ")


def test_provider_failures_are_actionable_and_do_not_leak(monkeypatch):
    class QuotaError(RuntimeError):
        status_code = 429

    model = FakeChatModel(error=QuotaError("private-key quota details"))
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)

    with pytest.raises(llm.LLMError, match="quota or rate limit") as caught:
        llm.generate_response("prompt")
    assert "private-key" not in str(caught.value)


def test_quota_503_retries_with_a_bound_and_succeeds(monkeypatch):
    class QuotaUnavailable(RuntimeError):
        status_code = 503

    class EventuallyAvailable:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls < 3:
                raise QuotaUnavailable("private quota detail")
            return FakeMessage("recovered")

    model = EventuallyAvailable()
    delays = []
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)
    monkeypatch.setattr(llm.time, "sleep", delays.append)
    monkeypatch.setattr(llm.random, "uniform", lambda low, _high: low)

    assert llm.generate_response("prompt") == "recovered"
    assert model.calls == 3
    assert len(delays) == 2
    assert sum(delays) <= llm.MAX_RETRY_ELAPSED_SECONDS


def test_retry_after_beyond_limit_stops_without_sleep(monkeypatch):
    class DelayedQuotaError(RuntimeError):
        status_code = 503
        retry_after = 60

    class AlwaysUnavailable:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise DelayedQuotaError("quota")

    model = AlwaysUnavailable()
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)
    monkeypatch.setattr(
        llm.time,
        "sleep",
        lambda _delay: pytest.fail("over-limit Retry-After must not sleep"),
    )

    with pytest.raises(llm.LLMError) as caught:
        llm.generate_response("prompt")
    assert caught.value.code == "gemini_quota_exhausted"
    assert model.calls == 1


def test_ollama_is_used_only_after_quota_and_only_when_enabled(monkeypatch):
    class QuotaError(RuntimeError):
        status_code = 429

    class GeminiUnavailable:
        def invoke(self, _messages):
            raise QuotaError("quota")

    class LocalModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return FakeMessage("local answer")

    monkeypatch.setattr(llm, "get_chat_model", GeminiUnavailable)
    monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
    local = LocalModel()
    monkeypatch.delenv("LLM_QUOTA_FALLBACK", raising=False)

    with pytest.raises(llm.LLMError) as disabled:
        llm.generate_response("prompt")
    assert disabled.value.code == "gemini_quota_exhausted"
    assert "private" not in str(disabled.value)

    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "ollama")
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "local-test-model")
    monkeypatch.setattr(llm, "_configured_ollama_model", lambda: local)
    assert llm.generate_response("prompt", system="Be concise") == "local answer"
    assert [message.type for message in local.messages] == ["system", "human"]
    metadata = llm.get_last_request_metadata()
    assert metadata["provider"] == "ollama"
    assert metadata["model"] == "local-test-model"


def test_quota_fallback_failure_is_sanitized(monkeypatch):
    class QuotaError(RuntimeError):
        status_code = 429

    monkeypatch.setattr(
        llm,
        "get_chat_model",
        lambda: FakeChatModel(error=QuotaError("secret quota detail")),
    )
    monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "missing-local-model")
    monkeypatch.setattr(
        llm,
        "_configured_ollama_model",
        lambda: (_ for _ in ()).throw(
            llm.LLMError(
                "local details",
                code="ollama_model_not_found",
                provider="ollama",
            )
        ),
    )

    with pytest.raises(llm.LLMError) as caught:
        llm.generate_response("prompt")
    assert caught.value.code == "gemini_quota_exhausted"
    assert "secret" not in str(caught.value)
    assert "local details" not in str(caught.value)


def test_stream_retries_before_output_but_not_after_partial_output(monkeypatch):
    class TemporaryFailure(RuntimeError):
        status_code = 503

    class StreamModel:
        def __init__(self, fail_after_output=False):
            self.calls = 0
            self.fail_after_output = fail_after_output

        def stream(self, _messages):
            self.calls += 1
            if self.calls == 1:
                if self.fail_after_output:
                    yield FakeMessage("partial")
                raise TemporaryFailure("private service detail")
            yield FakeMessage("complete")

    monkeypatch.setattr(llm.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(llm.random, "uniform", lambda low, _high: low)

    recoverable = StreamModel()
    monkeypatch.setattr(llm, "get_chat_model", lambda: recoverable)
    assert "".join(llm.stream_response("prompt")) == "complete"
    assert recoverable.calls == 2

    partial = StreamModel(fail_after_output=True)
    monkeypatch.setattr(llm, "get_chat_model", lambda: partial)
    stream = llm.stream_response("prompt")
    assert next(stream) == "partial"
    with pytest.raises(llm.LLMError):
        next(stream)
    assert partial.calls == 1


def test_ollama_health_check_is_mocked_and_model_must_be_present(
    monkeypatch,
):
    import httpx

    class TagsResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "local-test-model"}]}

    requests = []
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, timeout: (
            requests.append((url, timeout)) or TagsResponse()
        ),
    )
    constructed = []

    class FakeOllama:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        SimpleNamespace(ChatOllama=FakeOllama),
    )
    monkeypatch.setenv("LLM_QUOTA_FALLBACK", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "local-test-model")

    model = llm._configured_ollama_model()

    assert isinstance(model, FakeOllama)
    assert requests == [("http://localhost:11434/api/tags", 3.0)]
    assert constructed[0]["model"] == "local-test-model"
    assert constructed[0]["temperature"] == 0


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (type("AuthError", (RuntimeError,), {"status_code": 401})("secret"), "authentication"),
        (type("PermissionError", (RuntimeError,), {"status_code": 403})("secret"), "authentication"),
        (type("TimeoutError", (RuntimeError,), {"status_code": 504})("secret"), "timed out"),
        (type("NotFoundError", (RuntimeError,), {"status_code": 404})("secret"), "model is unavailable"),
        (type("InvalidRequest", (RuntimeError,), {"status_code": 400})("secret"), "rejected the request"),
        (type("ServerFailure", (RuntimeError,), {"status_code": 503})("secret"), "temporary service error"),
    ],
)
def test_http_failures_are_sanitized(error, message):
    translated = llm._translate_error(error, "Text generation")
    assert message in str(translated)
    assert "secret" not in str(translated)


def test_stream_yields_live_deltas_and_records_final_usage(monkeypatch):
    model = FakeChatModel(
        chunks=[
            FakeMessage("hello "),
            FakeMessage(
                "world",
                {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            ),
        ]
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)

    deltas = list(llm.stream_response("prompt"))

    assert deltas == ["hello ", "world"]
    assert llm.get_last_request_metadata()["usage"] == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }


def test_stream_normalizes_mask_tokens_split_across_chunks(monkeypatch):
    model = FakeChatModel(
        chunks=[
            FakeMessage("Assigned to [PER"),
            FakeMessage("SON_2] yesterday."),
        ]
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)

    assert "".join(llm.stream_response("prompt")) == "Assigned to Person B yesterday."


def test_stream_rejects_empty_and_translates_transport_error(monkeypatch):
    monkeypatch.setattr(llm, "get_chat_model", lambda: FakeChatModel())
    with pytest.raises(llm.LLMError, match="empty response"):
        list(llm.stream_response("prompt"))

    monkeypatch.setattr(
        llm,
        "get_chat_model",
        lambda: FakeChatModel(error=ConnectionError("private host")),
    )
    with pytest.raises(llm.LLMError, match="could not reach"):
        list(llm.stream_response("prompt"))


def test_multimodal_uses_gemini_image_content(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"png-data")
    model = FakeChatModel(FakeMessage("a chart"))
    monkeypatch.setattr(llm, "get_chat_model", lambda: model)

    assert llm.generate_multimodal_response("Describe it", [str(image)]) == "a chart"
    message = model.invocations[0][0]
    image_block = message.content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"png-data").decode("ascii")
    )


def test_multimodal_validates_image_paths(tmp_path):
    with pytest.raises(ValueError, match="No valid image"):
        llm.generate_multimodal_response(
            "Describe it",
            [str(tmp_path / "missing.png")],
        )

    text_file = tmp_path / "not-an-image.txt"
    text_file.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported format"):
        llm.generate_multimodal_response("Describe it", [str(text_file)])


def test_gemini_configuration_and_provider_metadata(monkeypatch):
    assert llm.get_text_model() == "gemini-test-model"
    assert llm.get_vision_model() == "gemini-test-model"
    assert llm.get_request_timeout() == llm.DEFAULT_REQUEST_TIMEOUT_SECONDS

    info = llm.describe_active_provider()
    assert info["provider"] == "gemini"
    assert info["model"] == "gemini-test-model"
    assert info["vision_provider"] == "gemini"
    assert llm.validate_llm_config()["configured"] is True

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "45")
    assert llm.get_request_timeout() == 45


@pytest.mark.parametrize("timeout", ["invalid", "0", "-1"])
def test_invalid_timeout_is_reported(monkeypatch, timeout):
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", timeout)
    with pytest.raises(llm.LLMError, match="GEMINI_TIMEOUT_SECONDS"):
        llm.get_request_timeout()


def test_missing_key_and_non_gemini_graph_provider_are_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.validate_llm_config()

    with pytest.raises(llm.LLMError, match="Gemini only"):
        llm.build_graph_extraction_llm("ollama")


def test_model_cache_rebuilds_for_configuration_change(monkeypatch):
    import langchain_google_genai

    created = []

    def fake_model(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        fake_model,
    )

    first = llm.get_chat_model()
    assert llm.get_chat_model() is first
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "another-model")
    second = llm.get_chat_model()

    assert second is not first
    assert len(created) == 2
    assert created[0]["api_key"] == "unit-test-key"
    assert created[0]["request_timeout"] == 300
