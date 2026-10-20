"""Central Gemini text, streaming, and multimodal generation."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger("app.llm")


class LLMError(Exception):
    """Raised when a configured model cannot generate a usable response."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "llm_error",
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        provider: str = "gemini",
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage | None = None
    duration_ms: float | None = None
    finish_reason: str | None = None


DEFAULT_TEXT_MODEL = "gemini-3.8-flash"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
PROVIDER_NAME = "gemini"
MAX_PROVIDER_RETRIES = 2
MAX_RETRY_ELAPSED_SECONDS = 12.0
MAX_RETRY_DELAY_SECONDS = 8.0

_model = None
_model_cache_key: tuple[str, float, str] | None = None
_last_request: dict = {
    "duration_ms": None,
    "usage": None,
    "provider": None,
    "model": None,
}
_ollama_model = None
_ollama_cache_key: tuple[str, str] | None = None

_PLACEHOLDER_PATTERN = re.compile(
    r"\[(?P<category>[A-Z][A-Z0-9_]*)_(?P<number>\d+)\]"
)
_PLACEHOLDER_LABELS = {
    "PERSON": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "GPE": "Place",
    "LOC": "Place",
    "LOCATION": "Place",
    "DATE": "Date",
    "EMAIL": "Email address",
    "PHONE": "Phone number",
    "ADDRESS": "Address",
}


def get_text_model() -> str:
    """Return the shared Gemini generation model."""

    return (
        os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL).strip()
        or DEFAULT_TEXT_MODEL
    )


def get_vision_model() -> str:
    """Return the shared multimodal Gemini model."""

    return get_text_model()


def get_request_timeout() -> float:
    """Return the configured Gemini request timeout in seconds."""

    raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    try:
        timeout = float(raw)
    except ValueError as error:
        raise LLMError(
            "GEMINI_TIMEOUT_SECONDS must be a positive number."
        ) from error

    if timeout <= 0:
        raise LLMError("GEMINI_TIMEOUT_SECONDS must be a positive number.")

    return timeout


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "GEMINI_API_KEY is not configured. Set it in the root .env file."
        )
    return key


def is_configured() -> bool:
    """Report whether the required API key is present without connecting."""

    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def get_chat_model():
    """Return the lazily-created shared LangChain Gemini chat model."""

    global _model, _model_cache_key

    model_name = get_text_model()
    timeout = get_request_timeout()
    key = _api_key()
    cache_key = (model_name, timeout, key)

    if _model is None or _model_cache_key != cache_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=key,
            temperature=0,
            request_timeout=timeout,
            retries=0,
        )
        _model_cache_key = cache_key

    return _model


def build_graph_extraction_llm(provider: str | None = None):
    """Share the configured Gemini model with ``LLMGraphTransformer``."""

    if provider not in (None, "gemini"):
        raise LLMError("Graph extraction supports Gemini only.")
    return get_chat_model()


def describe_active_provider() -> dict:
    """Describe the configured provider without returning credentials."""

    model = get_text_model()
    return {
        "provider": PROVIDER_NAME,
        "model": model,
        "vision_provider": PROVIDER_NAME,
        "vision_model": model,
        "backend": PROVIDER_NAME,
        "selection_source": (
            "env" if os.getenv("GEMINI_TEXT_MODEL", "").strip() else "default"
        ),
        "quota_fallback": (
            "ollama"
            if os.getenv("LLM_QUOTA_FALLBACK", "").strip().casefold()
            == "ollama"
            else None
        ),
    }


def validate_llm_config(force: bool = False) -> dict:
    """Validate Gemini configuration locally without a startup API call."""

    del force
    _api_key()
    timeout = get_request_timeout()
    model = get_text_model()
    if not model:
        raise LLMError("GEMINI_TEXT_MODEL cannot be empty.")

    return {
        "configured": True,
        "provider": PROVIDER_NAME,
        "model": model,
        "model_available": None,
        "vision_supported": None,
        "timeout_seconds": timeout,
        "warning": None,
    }


def reset_client() -> None:
    """Clear the cached Gemini model and request metadata."""

    global _model, _model_cache_key, _ollama_model, _ollama_cache_key
    _model = None
    _model_cache_key = None
    _ollama_model = None
    _ollama_cache_key = None
    reset_request_metadata()


def get_last_request_metadata() -> dict:
    return dict(_last_request)


def reset_request_metadata() -> None:
    _last_request.update(
        {
            "duration_ms": None,
            "usage": None,
            "provider": None,
            "model": None,
        }
    )


def _record_metadata(
    duration_ms: float,
    usage: TokenUsage | None,
    *,
    provider: str = PROVIDER_NAME,
    model: str | None = None,
) -> None:
    _last_request["duration_ms"] = round(duration_ms, 1)
    _last_request["provider"] = provider
    _last_request["model"] = model or (
        get_text_model() if provider == PROVIDER_NAME else os.getenv("OLLAMA_TEXT_MODEL")
    )
    _last_request["usage"] = (
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage is not None
        else None
    )


def _clean_token_count(value) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _usage_from_message(message) -> TokenUsage | None:
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None

    input_tokens = _clean_token_count(usage.get("input_tokens"))
    output_tokens = _clean_token_count(usage.get("output_tokens"))
    total_tokens = _clean_token_count(usage.get("total_tokens"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _content_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        return "".join(text_parts)

    return ""


def _alphabetic_label(number: int) -> str:
    """Turn a placeholder counter into a readable, non-reversible label."""

    label = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label or "A"


def normalize_masked_placeholders(text: str) -> str:
    """Replace internal mask tokens with readable, non-sensitive aliases."""

    def replace(match: re.Match[str]) -> str:
        category = match.group("category")
        label = _PLACEHOLDER_LABELS.get(
            category,
            category.replace("_", " ").title(),
        )
        number = int(match.group("number"))
        return f"{label} {_alphabetic_label(number)}"

    return _PLACEHOLDER_PATTERN.sub(replace, text)


def _messages(prompt: str, system: str | None) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system and system.strip():
        messages.append(SystemMessage(content=system.strip()))
    messages.append(HumanMessage(content=prompt))
    return messages


def _status_code(error: Exception) -> int | None:
    for value in (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            status = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    return None


def _retry_after_seconds(error: Exception) -> float | None:
    raw = getattr(error, "retry_after", None)
    if raw is None:
        headers = getattr(
            getattr(error, "response", None),
            "headers",
            None,
        )
        if headers is not None:
            raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (retry_at - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


def _translate_error(error: Exception, action: str) -> LLMError:
    """Map upstream failures to actionable, secret-free errors."""

    if isinstance(error, LLMError):
        return error

    status = _status_code(error)
    error_name = type(error).__name__.casefold()
    error_text = str(error).casefold()
    quota_error = (
        status == 429
        or any(
            token in error_name or token in error_text
            for token in (
                "resourceexhausted",
                "resource_exhausted",
                "quota",
                "rate limit",
                "rate_limit",
            )
        )
    )

    if action == "Response stream" and any(
        marker in error_text
        for marker in ("streaming not supported", "stream is not supported")
    ):
        return LLMError(
            "Response stream failed: the configured Gemini model does not "
            "support streaming."
        )

    if status in {401, 403} or any(
        token in error_name for token in ("unauthorized", "permissiondenied")
    ):
        return LLMError(
            f"{action} failed: Gemini authentication was rejected. "
            "Check GEMINI_API_KEY and API access.",
            code="gemini_authentication_failed",
        )
    if quota_error:
        return LLMError(
            f"{action} failed: Gemini quota or rate limit was exceeded.",
            code="gemini_quota_exhausted",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(error),
        )
    if isinstance(error, (TimeoutError,)) or any(
        token in error_name for token in ("timeout", "deadlineexceeded")
    ):
        return LLMError(
            f"{action} failed: the Gemini request timed out.",
            code="gemini_timeout",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(error),
        )
    if status == 404:
        return LLMError(
            f"{action} failed: the configured Gemini model is unavailable.",
            code="gemini_model_not_found",
        )
    if status == 400:
        return LLMError(
            f"{action} failed: Gemini rejected the request; check the "
            "model configuration and request format.",
            code="gemini_invalid_request",
        )
    if isinstance(error, ConnectionError) or any(
        token in error_name
        for token in ("connecterror", "networkerror", "serviceunavailable")
    ):
        return LLMError(
            f"{action} failed: could not reach the Gemini service.",
            code="gemini_network_error",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(error),
        )
    if status is not None and status >= 500:
        return LLMError(
            f"{action} failed: Gemini returned a temporary service error.",
            code="gemini_service_unavailable",
            retryable=status in {500, 502, 503, 504},
            retry_after_seconds=_retry_after_seconds(error),
        )

    return LLMError(
        f"{action} failed: Gemini request error ({type(error).__name__}).",
        code="gemini_request_error",
    )


def _retry_gemini(operation, action: str):
    started = time.monotonic()
    retries = 0
    while True:
        try:
            return operation()
        except Exception as error:
            translated = _translate_error(error, action)
            if (
                not translated.retryable
                or retries >= MAX_PROVIDER_RETRIES
            ):
                raise translated from error

            exponential = min(2**retries, MAX_RETRY_DELAY_SECONDS)
            delay = random.uniform(exponential * 0.8, exponential * 1.2)
            if translated.retry_after_seconds is not None:
                delay = max(delay, translated.retry_after_seconds)
            elapsed = time.monotonic() - started
            if (
                delay > MAX_RETRY_DELAY_SECONDS
                or elapsed + delay > MAX_RETRY_ELAPSED_SECONDS
            ):
                raise translated from error
            retries += 1
            logger.warning(
                "%s transient failure; retry=%s code=%s delay_seconds=%.1f",
                action,
                retries,
                translated.code,
                delay,
            )
            time.sleep(delay)


def _configured_ollama_model():
    """Return a healthy configured local text model or fail without fallback."""

    global _ollama_model, _ollama_cache_key
    if os.getenv("LLM_QUOTA_FALLBACK", "").strip().casefold() != "ollama":
        raise LLMError(
            "Ollama quota fallback is not enabled.",
            code="ollama_fallback_disabled",
            provider="ollama",
        )
    base_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    model_name = os.getenv("OLLAMA_TEXT_MODEL", "").strip()
    if not base_url or not model_name:
        raise LLMError(
            "Ollama fallback requires OLLAMA_BASE_URL and OLLAMA_TEXT_MODEL.",
            code="ollama_not_configured",
            provider="ollama",
        )
    key = (base_url, model_name)
    if _ollama_model is not None and _ollama_cache_key == key:
        return _ollama_model

    try:
        import httpx

        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        payload = response.json()
        names = {
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        if model_name not in names:
            raise LLMError(
                "The configured Ollama model is not available.",
                code="ollama_model_not_found",
                provider="ollama",
            )

        from langchain_ollama import ChatOllama

        _ollama_model = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            client_kwargs={"timeout": 60.0},
        )
        _ollama_cache_key = key
        return _ollama_model
    except LLMError:
        raise
    except Exception as error:
        raise LLMError(
            "The configured Ollama service is unavailable.",
            code="ollama_unavailable",
            retryable=True,
            provider="ollama",
        ) from error


def _invoke_text_model(messages, action: str):
    started = time.perf_counter()
    try:
        response = _retry_gemini(
            lambda: get_chat_model().invoke(messages),
            action,
        )
        return response, PROVIDER_NAME, get_text_model(), started
    except LLMError as error:
        if error.code != "gemini_quota_exhausted":
            raise
        try:
            model = _configured_ollama_model()
            response = model.invoke(messages)
        except LLMError as fallback_error:
            if fallback_error.code == "ollama_fallback_disabled":
                raise error
            raise LLMError(
                "Gemini quota or rate limit was exceeded; Ollama fallback "
                f"is unavailable ({fallback_error.code}).",
                code=error.code,
                retryable=True,
                retry_after_seconds=error.retry_after_seconds,
            ) from fallback_error
        except Exception as ollama_error:
            raise LLMError(
                "Ollama fallback failed; retry the request later.",
                code="ollama_generation_failed",
                retryable=True,
                provider="ollama",
            ) from ollama_error
        return (
            response,
            "ollama",
            os.getenv("OLLAMA_TEXT_MODEL", "").strip(),
            started,
        )


def _require_text(message, action: str) -> str:
    content = _content_text(getattr(message, "content", None))
    if not content.strip():
        raise LLMError(
            f"{action} failed: the configured text model returned an "
            "empty response."
        )
    return normalize_masked_placeholders(content)


def generate_response(prompt: str, system: str | None = None) -> str:
    """Generate a text response using the shared Gemini model."""

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    try:
        response, provider, model_name, started = _invoke_text_model(
            _messages(prompt, system),
            "Text generation",
        )
    except LLMError:
        raise

    answer = _require_text(response, "Text generation")
    _record_metadata(
        (time.perf_counter() - started) * 1000,
        _usage_from_message(response),
        provider=provider,
        model=model_name,
    )
    return answer


def _stream_text_chunks(stream) -> Iterator[str]:
    """Yield live text while safely normalizing tokens split across chunks."""

    pending = ""
    for chunk in stream:
        content = _content_text(getattr(chunk, "content", None))
        if not content:
            continue
        pending += content

        while pending:
            opening = pending.find("[")
            if opening < 0:
                yield pending
                pending = ""
                break
            if opening:
                yield pending[:opening]
                pending = pending[opening:]

            closing = pending.find("]", 1)
            if closing < 0:
                if len(pending) > 128:
                    yield pending[0]
                    pending = pending[1:]
                    continue
                break

            token = pending[: closing + 1]
            yield normalize_masked_placeholders(token)
            pending = pending[closing + 1 :]

    if pending:
        yield normalize_masked_placeholders(pending)


def stream_response(
    prompt: str,
    system: str | None = None,
) -> Iterator[str]:
    """Yield real Gemini streaming text deltas."""

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    started = time.perf_counter()
    retry_started = time.monotonic()
    retries = 0
    emitted_any = False
    usage = None

    while True:
        attempt_usage = None
        try:
            stream = get_chat_model().stream(_messages(prompt, system))

            def track_usage():
                nonlocal attempt_usage
                for raw_chunk in stream:
                    usage_from_chunk = _usage_from_message(raw_chunk)
                    if usage_from_chunk is not None:
                        attempt_usage = usage_from_chunk
                    yield raw_chunk

            for chunk in _stream_text_chunks(track_usage()):
                emitted_any = True
                yield chunk
            usage = attempt_usage
            break
        except Exception as error:
            translated = _translate_error(error, "Response stream")
            if not translated.retryable or emitted_any or retries >= MAX_PROVIDER_RETRIES:
                raise translated from error

            exponential = min(2**retries, MAX_RETRY_DELAY_SECONDS)
            delay = random.uniform(exponential * 0.8, exponential * 1.2)
            if translated.retry_after_seconds is not None:
                delay = max(delay, translated.retry_after_seconds)
            elapsed = time.monotonic() - retry_started
            if (
                delay > MAX_RETRY_DELAY_SECONDS
                or elapsed + delay > MAX_RETRY_ELAPSED_SECONDS
            ):
                raise translated from error
            retries += 1
            logger.warning(
                "Response stream transient failure; retry=%s code=%s "
                "delay_seconds=%.1f",
                retries,
                translated.code,
                delay,
            )
            time.sleep(delay)

    if not emitted_any:
        raise LLMError("Response stream failed: Gemini returned an empty response.")

    _record_metadata((time.perf_counter() - started) * 1000, usage)


def generate_multimodal_response(
    message: str,
    image_paths: list[str],
    system: str | None = None,
) -> str:
    """Answer a question using Gemini text and image input."""

    paths = [Path(path) for path in image_paths if path]
    valid_images = [path for path in paths if path.is_file()]
    if not valid_images:
        raise ValueError("No valid image files were found.")
    if not message or not message.strip():
        raise ValueError("Prompt cannot be empty.")

    from langchain_core.messages import HumanMessage, SystemMessage

    content: list[dict] = [{"type": "text", "text": message}]
    for image_path in valid_images:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("An image file has an unsupported format.")
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                },
            }
        )

    messages = []
    if system and system.strip():
        messages.append(SystemMessage(content=system.strip()))
    messages.append(HumanMessage(content=content))

    started = time.perf_counter()
    try:
        response = _retry_gemini(
            lambda: get_chat_model().invoke(messages),
            "Image question answering",
        )
    except LLMError:
        raise

    answer = _require_text(response, "Image question answering")
    _record_metadata(
        (time.perf_counter() - started) * 1000,
        _usage_from_message(response),
        provider=PROVIDER_NAME,
        model=get_vision_model(),
    )
    return answer
