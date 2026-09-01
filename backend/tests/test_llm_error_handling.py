"""
Comprehensive LLM Error Handling, Retry, and Timeout Tests.

Tests the production-grade provider-independent error handling:
  1.  Exception hierarchy
  2.  All exceptions inherit from LLMError
  3.  LLMConfig timeout/retry configuration
  4.  Retryable timeout
  5.  Retryable rate-limit error
  6.  Retryable temporary provider error
  7.  Authentication error is not retried
  8.  Configuration error is not retried
  9.  Retry count is respected
  10. Successful retry returns LLMResponse
  11. Exhausted retries raise common exception
  12. Provider-specific exceptions do not escape adapters
  13. MockLLM remains deterministic
  14. Factory still creates providers correctly
  15. chat.py remains provider-agnostic
  16. No provider SDK imports in application/business layer
  17. No silent fallback to MockLLM
  18. Existing API behavior remains intact
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.llm.base import BaseLLM, LLMResponse
from app.llm.config import LLMConfig
from app.llm.types import LLMRequest
from app.llm.exceptions import (
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMProviderError,
)
from app.llm.retry import with_retry, is_retryable
from app.llm.mock import MockLLM
from app.llm.factory import create_llm


# ============================================================
# 1. Exception hierarchy
# ============================================================

def test_exception_hierarchy():
    """LLMError is the base of all common LLM exceptions."""
    assert issubclass(LLMConfigurationError, LLMError)
    assert issubclass(LLMAuthenticationError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMProviderError, LLMError)


# ============================================================
# 2. All exceptions inherit from LLMError
# ============================================================

def test_all_exceptions_inherit_from_base():
    """Every common exception is an instance of LLMError."""
    for exc_cls in (
        LLMConfigurationError,
        LLMAuthenticationError,
        LLMRateLimitError,
        LLMTimeoutError,
        LLMProviderError,
    ):
        exc = exc_cls("test")
        assert isinstance(exc, LLMError)
        assert isinstance(exc, Exception)
        assert str(exc) == "test"


# ============================================================
# 3. LLMConfig timeout/retry configuration
# ============================================================

def test_config_timeout_retry_defaults():
    """app.config exposes timeout/retry settings with defaults."""
    import app.config as cfg

    assert isinstance(cfg.LLM_TIMEOUT_SECONDS, float)
    assert cfg.LLM_TIMEOUT_SECONDS > 0

    assert isinstance(cfg.LLM_MAX_RETRIES, int)
    assert cfg.LLM_MAX_RETRIES >= 0

    assert isinstance(cfg.LLM_RETRY_DELAY_SECONDS, float)
    assert cfg.LLM_RETRY_DELAY_SECONDS >= 0


def test_config_timeout_retry_env_override():
    """Environment variables override the defaults."""
    with patch.dict(os.environ, {
        "LLM_TIMEOUT_SECONDS": "30",
        "LLM_MAX_RETRIES": "5",
        "LLM_RETRY_DELAY_SECONDS": "0.5",
    }):
        # Re-import to pick up patched env
        import importlib
        import app.config
        importlib.reload(app.config)

        assert app.config.LLM_TIMEOUT_SECONDS == 30.0
        assert app.config.LLM_MAX_RETRIES == 5
        assert app.config.LLM_RETRY_DELAY_SECONDS == 0.5

    # Restore
    import importlib
    import app.config
    importlib.reload(app.config)


# ============================================================
# 4. Retryable timeout
# ============================================================

def test_retryable_timeout():
    """LLMTimeoutError is retryable."""
    exc = LLMTimeoutError("timed out")
    assert is_retryable(exc)


# ============================================================
# 5. Retryable rate-limit error
# ============================================================

def test_retryable_rate_limit():
    """LLMRateLimitError is retryable."""
    exc = LLMRateLimitError("rate limited")
    assert is_retryable(exc)


# ============================================================
# 6. Retryable temporary provider error
# ============================================================

def test_retryable_provider_error():
    """LLMProviderError is retryable."""
    exc = LLMProviderError("server error")
    assert is_retryable(exc)


# ============================================================
# 7. Authentication error is not retried
# ============================================================

def test_auth_error_not_retryable():
    """LLMAuthenticationError must NOT be retried."""
    exc = LLMAuthenticationError("bad key")
    assert not is_retryable(exc)


def test_config_error_not_retryable():
    """LLMConfigurationError must NOT be retried."""
    exc = LLMConfigurationError("bad config")
    assert not is_retryable(exc)


# ============================================================
# 8. Configuration error is not retried (retry raises immediately)
# ============================================================

def test_config_error_raises_immediately():
    """LLMConfigurationError propagates without any retry."""
    call_count = 0

    def failing_fn():
        nonlocal call_count
        call_count += 1
        raise LLMConfigurationError("bad config")

    with pytest.raises(LLMConfigurationError):
        with_retry(failing_fn, max_retries=3, delay=0)

    assert call_count == 1


def test_auth_error_raises_immediately():
    """LLMAuthenticationError propagates without any retry."""
    call_count = 0

    def failing_fn():
        nonlocal call_count
        call_count += 1
        raise LLMAuthenticationError("unauthorized")

    with pytest.raises(LLMAuthenticationError):
        with_retry(failing_fn, max_retries=3, delay=0)

    assert call_count == 1


# ============================================================
# 9. Retry count is respected
# ============================================================

def test_retry_count_respected():
    """Transient errors are retried exactly max_retries times."""
    call_count = 0

    def always_timeout():
        nonlocal call_count
        call_count += 1
        raise LLMTimeoutError("timeout")

    with pytest.raises(LLMTimeoutError):
        with_retry(always_timeout, max_retries=2, delay=0)

    # 1 initial + 2 retries = 3 total attempts
    assert call_count == 3


def test_retry_count_zero():
    """max_retries=0 means no retries (only 1 attempt)."""
    call_count = 0

    def always_timeout():
        nonlocal call_count
        call_count += 1
        raise LLMTimeoutError("timeout")

    with pytest.raises(LLMTimeoutError):
        with_retry(always_timeout, max_retries=0, delay=0)

    assert call_count == 1


# ============================================================
# 10. Successful retry returns LLMResponse
# ============================================================

def test_successful_retry():
    """A transient error followed by success returns LLMResponse."""
    call_count = 0

    def succeeds_on_second():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMTimeoutError("timeout")
        return LLMResponse(text="success", model="m", provider="p")

    result = with_retry(succeeds_on_second, max_retries=2, delay=0)
    assert isinstance(result, LLMResponse)
    assert result.text == "success"
    assert call_count == 2


# ============================================================
# 11. Exhausted retries raise common exception
# ============================================================

def test_exhausted_retries_raises():
    """After all retries exhausted, the last common exception is raised."""
    call_count = 0

    def always_rate_limited():
        nonlocal call_count
        call_count += 1
        raise LLMRateLimitError("rate limited")

    with pytest.raises(LLMRateLimitError) as exc_info:
        with_retry(always_rate_limited, max_retries=2, delay=0)

    assert call_count == 3
    assert "rate limited" in str(exc_info.value)


# ============================================================
# 12. Provider-specific exceptions do not escape adapters
# ============================================================

def test_gemini_adapter_translates_errors():
    """GeminiLLM._translate_error maps SDK errors to common errors."""
    from app.llm.gemini import GeminiLLM

    # Timeout
    exc = TimeoutError("request timeout")
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMTimeoutError)

    # Non-APIError
    exc = RuntimeError("something went wrong")
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)


def test_gemini_adapter_translates_api_errors():
    """GeminiLLM._translate_error maps google-genai APIError subclasses."""
    from app.llm.gemini import GeminiLLM
    from google.genai.errors import ClientError, ServerError, APIError

    # ClientError with 401 → authentication error
    exc = ClientError(401, {"error": {"status": "UNAUTHENTICATED"}}, None)
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMAuthenticationError)

    # ClientError with 429 → rate limit error
    exc = ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}, None)
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMRateLimitError)

    # ServerError with 500 → provider error
    exc = ServerError(500, {"error": {"status": "INTERNAL"}}, None)
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)

    # ClientError with 400 → provider error
    exc = ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}}, None)
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)


def test_gemini_adapter_httpx_timeout():
    """GeminiLLM translates httpx.TimeoutException to LLMTimeoutError."""
    from app.llm.gemini import GeminiLLM
    import httpx

    exc = httpx.TimeoutException("read timed out")
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMTimeoutError)

    exc = httpx.ConnectError("connection refused")
    result = GeminiLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)


def test_groq_adapter_translates_errors():
    """GroqLLM._translate_error maps groq SDK errors."""
    from app.llm.groq import GroqLLM
    from groq import (
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
        InternalServerError,
        BadRequestError,
    )

    assert isinstance(
        GroqLLM._translate_error(APITimeoutError("timeout")),
        LLMTimeoutError,
    )
    assert isinstance(
        GroqLLM._translate_error(AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )),
        LLMAuthenticationError,
    )
    assert isinstance(
        GroqLLM._translate_error(RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )),
        LLMRateLimitError,
    )
    assert isinstance(
        GroqLLM._translate_error(InternalServerError(
            message="server error",
            response=MagicMock(status_code=500, headers={}),
            body=None,
        )),
        LLMProviderError,
    )
    assert isinstance(
        GroqLLM._translate_error(BadRequestError(
            message="bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )),
        LLMProviderError,
    )


def test_openrouter_adapter_translates_errors():
    """OpenRouterLLM._translate_error maps openai SDK errors."""
    from app.llm.openrouter import OpenRouterLLM
    from openai import (
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
        InternalServerError,
        BadRequestError,
    )

    assert isinstance(
        OpenRouterLLM._translate_error(APITimeoutError("timeout")),
        LLMTimeoutError,
    )
    assert isinstance(
        OpenRouterLLM._translate_error(AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )),
        LLMAuthenticationError,
    )
    assert isinstance(
        OpenRouterLLM._translate_error(RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )),
        LLMRateLimitError,
    )
    assert isinstance(
        OpenRouterLLM._translate_error(InternalServerError(
            message="server error",
            response=MagicMock(status_code=500, headers={}),
            body=None,
        )),
        LLMProviderError,
    )
    assert isinstance(
        OpenRouterLLM._translate_error(BadRequestError(
            message="bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )),
        LLMProviderError,
    )


def test_mistral_adapter_translates_errors():
    """MistralLLM._translate_error maps mistral SDK errors."""
    from app.llm.mistral import MistralLLM
    from mistralai.client.errors import SDKError
    import httpx

    # httpx.TimeoutException → LLMTimeoutError
    exc = httpx.TimeoutException("timeout")
    result = MistralLLM._translate_error(exc)
    assert isinstance(result, LLMTimeoutError)

    # httpx.ConnectError → LLMProviderError
    exc = httpx.ConnectError("connection refused")
    result = MistralLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)

    # SDKError with 429 → LLMRateLimitError
    mock_response = MagicMock()
    mock_response.status_code = 429
    exc = SDKError("rate limited", mock_response, None)
    result = MistralLLM._translate_error(exc)
    assert isinstance(result, LLMRateLimitError)

    # SDKError with 401 → LLMAuthenticationError
    mock_response = MagicMock()
    mock_response.status_code = 401
    exc = SDKError("unauthorized", mock_response, None)
    result = MistralLLM._translate_error(exc)
    assert isinstance(result, LLMAuthenticationError)

    # SDKError with 500 → LLMProviderError
    mock_response = MagicMock()
    mock_response.status_code = 500
    exc = SDKError("server error", mock_response, None)
    result = MistralLLM._translate_error(exc)
    assert isinstance(result, LLMProviderError)


# ============================================================
# 13. MockLLM remains deterministic
# ============================================================

def test_mock_llm_deterministic():
    """MockLLM continues to produce deterministic responses."""
    llm = MockLLM(
        config=LLMConfig(provider="mock", model="mock-model", api_key=""),
    )

    request = LLMRequest(
        messages=[{"role": "user", "content": "hello"}],
    )
    resp = llm.generate(request)
    assert isinstance(resp, LLMResponse)
    assert "hello" in resp.text
    assert resp.provider == "mock"

    # Same input → same output
    resp2 = llm.generate(request)
    assert resp.text == resp2.text


# ============================================================
# 14. Factory still creates providers correctly
# ============================================================

def test_factory_creates_mock():
    """Factory creates MockLLM correctly."""
    llm = create_llm(provider="mock", api_key="", model="test")
    assert isinstance(llm, MockLLM)
    assert isinstance(llm, BaseLLM)


def test_factory_creates_gemini():
    """Factory creates GeminiLLM correctly."""
    from app.llm.gemini import GeminiLLM

    llm = create_llm(provider="gemini", api_key="fake-key", model="test-model")
    assert isinstance(llm, GeminiLLM)
    assert isinstance(llm, BaseLLM)


def test_factory_creates_groq():
    """Factory creates GroqLLM correctly."""
    from app.llm.groq import GroqLLM

    llm = create_llm(provider="groq", api_key="fake-key", model="test-model")
    assert isinstance(llm, GroqLLM)
    assert isinstance(llm, BaseLLM)


def test_factory_creates_mistral():
    """Factory creates MistralLLM correctly."""
    from app.llm.mistral import MistralLLM

    llm = create_llm(provider="mistral", api_key="fake-key", model="test-model")
    assert isinstance(llm, MistralLLM)
    assert isinstance(llm, BaseLLM)


def test_factory_creates_openrouter():
    """Factory creates OpenRouterLLM correctly."""
    from app.llm.openrouter import OpenRouterLLM

    llm = create_llm(provider="openrouter", api_key="fake-key", model="test-model")
    assert isinstance(llm, OpenRouterLLM)
    assert isinstance(llm, BaseLLM)


# ============================================================
# 15. chat.py remains provider-agnostic
# ============================================================

def test_chat_py_provider_agnostic():
    """chat.py does not import any provider-specific SDK."""
    import app.chat as chat_mod

    source = open(chat_mod.__file__).read()

    assert "from google" not in source
    assert "from groq" not in source
    assert "from mistralai" not in source
    assert "from openai" not in source

    assert "from .llm.factory import create_llm" in source
    assert "from .llm.types import LLMRequest" in source
    assert "from .llm.retry import with_retry" in source


# ============================================================
# 16. No provider SDK imports in application/business layer
# ============================================================

def test_no_sdk_imports_in_application():
    """Application layer files do not import provider SDKs."""
    for filepath in [
        "app/chat.py",
        "app/config.py",
    ]:
        full = os.path.join(os.path.dirname(__file__), "..", filepath)
        if not os.path.exists(full):
            continue
        source = open(full).read()
        assert "from google" not in source, f"{filepath} imports google SDK"
        assert "from groq" not in source, f"{filepath} imports groq SDK"
        assert "from mistralai" not in source, f"{filepath} imports mistral SDK"
        assert "from openai" not in source, f"{filepath} imports openai SDK"


# ============================================================
# 17. No silent fallback to MockLLM
# ============================================================

def test_no_silent_fallback():
    """Real provider failures remain real failures; no fallback to MockLLM."""
    try:
        with patch("app.llm.factory.LLM_API_KEY", None):
            create_llm(provider="gemini", api_key=None, model="test")
        assert False, "Should raise ValueError, not fall back to mock"
    except ValueError:
        pass

    try:
        with patch("app.llm.factory.LLM_API_KEY", None):
            create_llm(provider="groq", api_key=None, model="test")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_no_silent_fallback_in_retry():
    """Retry never falls back to MockLLM; it re-raises the last error."""
    call_count = 0

    def always_fails():
        nonlocal call_count
        call_count += 1
        raise LLMProviderError("provider down")

    with pytest.raises(LLMProviderError):
        with_retry(always_fails, max_retries=2, delay=0)

    assert call_count == 3  # 1 initial + 2 retries


# ============================================================
# 18. Existing API behavior remains intact
# ============================================================

def test_api_behavior_unchanged():
    """The FastAPI app can be imported and basic endpoints exist."""
    from app.main import app

    routes = [r.path for r in app.routes]
    assert "/" in routes
    assert "/cases/{case_id}/chat" in routes
    assert "/chats/{chat_id}/messages" in routes
    assert "/cases/{case_id}/chats" in routes


# ============================================================
# 19. Provider adapters have _translate_error
# ============================================================

def test_all_real_providers_have_translate_error():
    """Every real provider adapter has a _translate_error method."""
    from app.llm.gemini import GeminiLLM
    from app.llm.groq import GroqLLM
    from app.llm.mistral import MistralLLM
    from app.llm.openrouter import OpenRouterLLM

    for cls in (GeminiLLM, GroqLLM, MistralLLM, OpenRouterLLM):
        assert hasattr(cls, "_translate_error"), (
            f"{cls.__name__} missing _translate_error"
        )
        assert callable(cls._translate_error)


# ============================================================
# 20. Retry mixed errors
# ============================================================

def test_retry_mixed_transient_errors():
    """Retry works across different transient error types."""
    call_count = 0

    def mixed_errors():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMTimeoutError("timeout")
        if call_count == 2:
            raise LLMRateLimitError("rate limited")
        return LLMResponse(text="ok", model="m", provider="p")

    result = with_retry(mixed_errors, max_retries=3, delay=0)
    assert result.text == "ok"
    assert call_count == 3


# ============================================================
# 21. Non-retryable error stops retry chain
# ============================================================

def test_non_retryable_stops_retry_chain():
    """A non-retryable error mid-chain stops retries immediately."""
    call_count = 0

    def mixed_non_retryable():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMTimeoutError("timeout")
        raise LLMAuthenticationError("auth failed")

    with pytest.raises(LLMAuthenticationError):
        with_retry(mixed_non_retryable, max_retries=3, delay=0)

    # First call: timeout → retry. Second call: auth → stop.
    assert call_count == 2


# ============================================================
# 22. Retry with successful response on first try
# ============================================================

def test_retry_no_error():
    """No retry needed when first attempt succeeds."""
    call_count = 0

    def succeeds():
        nonlocal call_count
        call_count += 1
        return LLMResponse(text="ok", model="m", provider="p")

    result = with_retry(succeeds, max_retries=3, delay=0)
    assert result.text == "ok"
    assert call_count == 1


# ============================================================
# 23. Exception chaining (__cause__)
# ============================================================

def test_exception_chaining():
    """Provider adapters wrap SDK exceptions as common LLMError subclasses."""
    from app.llm.gemini import GeminiLLM
    from app.llm.groq import GroqLLM
    from app.llm.mistral import MistralLLM
    from app.llm.openrouter import OpenRouterLLM

    # Each adapter's _translate_error must accept arbitrary exceptions
    # and return a common LLMError subclass, never a raw SDK exception.
    for cls in (GeminiLLM, GroqLLM, MistralLLM, OpenRouterLLM):
        exc = cls._translate_error(RuntimeError("generic"))
        assert isinstance(exc, LLMError), (
            f"{cls.__name__}._translate_error did not return LLMError"
        )

    # Gemini: TimeoutError → LLMTimeoutError
    assert isinstance(
        GeminiLLM._translate_error(TimeoutError("t")), LLMTimeoutError
    )

    # Groq: groq.APITimeoutError → LLMTimeoutError
    from groq import APITimeoutError as GroqTimeout
    assert isinstance(
        GroqLLM._translate_error(GroqTimeout("t")), LLMTimeoutError
    )

    # OpenRouter/openai: openai.APITimeoutError → LLMTimeoutError
    from openai import APITimeoutError as OaiTimeout
    assert isinstance(
        OpenRouterLLM._translate_error(OaiTimeout(request=MagicMock())), LLMTimeoutError
    )


# ============================================================
# 24. Main.py imports LLM error types
# ============================================================

def test_main_py_imports_llm_exceptions():
    """main.py imports common LLM exceptions for HTTP mapping."""
    import app.main as main_mod
    source = open(main_mod.__file__).read()
    assert "from .llm.exceptions import" in source
    assert "LLMAuthenticationError" in source
    assert "LLMRateLimitError" in source
    assert "LLMTimeoutError" in source
    assert "LLMProviderError" in source


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    tests = [
        test_exception_hierarchy,
        test_all_exceptions_inherit_from_base,
        test_config_timeout_retry_defaults,
        test_config_timeout_retry_env_override,
        test_retryable_timeout,
        test_retryable_rate_limit,
        test_retryable_provider_error,
        test_auth_error_not_retryable,
        test_config_error_not_retryable,
        test_config_error_raises_immediately,
        test_auth_error_raises_immediately,
        test_retry_count_respected,
        test_retry_count_zero,
        test_successful_retry,
        test_exhausted_retries_raises,
        test_gemini_adapter_translates_errors,
        test_gemini_adapter_translates_api_errors,
        test_gemini_adapter_httpx_timeout,
        test_groq_adapter_translates_errors,
        test_openrouter_adapter_translates_errors,
        test_mistral_adapter_translates_errors,
        test_mock_llm_deterministic,
        test_factory_creates_mock,
        test_factory_creates_gemini,
        test_factory_creates_groq,
        test_factory_creates_mistral,
        test_factory_creates_openrouter,
        test_chat_py_provider_agnostic,
        test_no_sdk_imports_in_application,
        test_no_silent_fallback,
        test_no_silent_fallback_in_retry,
        test_api_behavior_unchanged,
        test_all_real_providers_have_translate_error,
        test_retry_mixed_transient_errors,
        test_non_retryable_stops_retry_chain,
        test_retry_no_error,
        test_exception_chaining,
        test_main_py_imports_llm_exceptions,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("LLM ERROR HANDLING TEST SUITE")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASSED: {test_fn.__name__}")
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {test_fn.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if errors:
        print()
        print("FAILURES:")
        for name, err in errors:
            print(f"  {name}: {err}")

    sys.exit(1 if failed else 0)
