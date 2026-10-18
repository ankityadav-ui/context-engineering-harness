# Single-backend LLM integration (Phase 5): Ollama only.
#
# Design rules enforced here:
# - Text and image (multimodal) requests both go to Ollama with the
#   configured model tag (default "gemma4:31b"). The configured tag is
#   never silently substituted with another model.
# - There is no provider factory, no runtime provider switching, and no
#   OpenRouter code anywhere in the application.
# - Host resolution: OLLAMA_HOST when set, otherwise Ollama Cloud
#   (https://ollama.com) when an Ollama API key is present in .env,
#   otherwise the local server (http://localhost:11434).
# - Trusted system instructions are sent as a real `system` message.
#   User content (including retrieved document context) never travels
#   in the system message, and retrieved text can never override the
#   trusted instructions.
# - Request duration and token usage are recorded only when the backend
#   reports them. Nothing is ever estimated or fabricated.

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

# Load the root .env file before reading configuration.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class LLMError(Exception):
    """Raised when the LLM cannot generate a response."""


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


# --------------------------------------------------
# Configuration (read from the environment; validated at startup)
# --------------------------------------------------

DEFAULT_MODEL = "gemma4:31b"
DEFAULT_LOCAL_HOST = "http://localhost:11434"
CLOUD_HOST = "https://ollama.com"

# Generation can legitimately take minutes (long answers, image input),
# so the request timeout is generous. The startup availability probe must
# not block application startup on an unreachable host, so it uses a
# short connect/read timeout instead.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
PROBE_TIMEOUT_SECONDS = 5.0


def get_request_timeout() -> float:
    """Return the request timeout in seconds (OLLAMA_TIMEOUT_SECONDS)."""

    raw = os.getenv("OLLAMA_TIMEOUT_SECONDS", "").strip()

    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    return value if value > 0 else DEFAULT_REQUEST_TIMEOUT_SECONDS


def get_text_model() -> str:
    """Return the configured Ollama model tag (text + vision)."""

    return (
        os.getenv("OLLAMA_TEXT_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL
    )


def get_vision_model() -> str:
    """Return the model used for image requests.

    OLLAMA_VISION_MODEL is an optional override. When it is unset the
    configured text model is used, so the documented single-model setup
    (gemma4:31b for text and vision) needs only OLLAMA_TEXT_MODEL.
    """

    return (
        os.getenv("OLLAMA_VISION_MODEL", "").strip()
        or get_text_model()
    )


def get_api_key() -> str:
    """Return the Ollama API key (both spellings are accepted).

    The key is never logged, returned by an endpoint, or embedded in an
    error message.
    """

    return (
        os.getenv("Ollama_API_KEY", "").strip()
        or os.getenv("OLLAMA_API_KEY", "").strip()
    )


def get_host() -> str:
    """Return the Ollama host: OLLAMA_HOST, else cloud, else local."""

    explicit = os.getenv("OLLAMA_HOST", "").strip()

    if explicit:
        return explicit.rstrip("/")

    return CLOUD_HOST if get_api_key() else DEFAULT_LOCAL_HOST


def backend_kind() -> str:
    """Describe the active backend: "cloud", "local", or "custom"."""

    host = get_host()

    if host.rstrip("/") == CLOUD_HOST:
        return "cloud"

    if host.rstrip("/") == DEFAULT_LOCAL_HOST:
        return "local"

    return "custom"


def _requires_api_key() -> bool:
    """Whether the active host is Ollama Cloud (key required)."""

    return backend_kind() == "cloud"


def is_configured() -> bool:
    """Report whether an Ollama backend and model tag are configured."""

    if not get_text_model():
        return False

    if _requires_api_key() and not get_api_key():
        return False

    return True


# --------------------------------------------------
# Request metadata (reported when the backend supplies it)
# --------------------------------------------------

_last_request: dict = {
    "duration_ms": None,
    "usage": None,
}


def get_last_request_metadata() -> dict:
    """Return duration/token usage of the most recent request.

    Values are ``None`` when the serving backend did not report them.
    No usage or timing data is ever estimated or fabricated.
    """

    return dict(_last_request)


def _record_metadata(
    duration_ms: float,
    usage: TokenUsage | None,
) -> None:
    _last_request["duration_ms"] = round(duration_ms, 1)
    _last_request["usage"] = (
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage is not None
        else None
    )


def reset_request_metadata() -> None:
    """Clear the recorded request metadata (used by tests)."""

    _last_request["duration_ms"] = None
    _last_request["usage"] = None


def describe_active_provider() -> dict:
    """Describe the single serving backend (no secrets)."""

    return {
        "provider": "ollama",
        "model": get_text_model(),
        "vision_provider": "ollama",
        "vision_model": get_vision_model(),
        "backend": backend_kind(),
        "selection_source": "env",
    }


# --------------------------------------------------
# Ollama client (created lazily, cached per host/key)
# --------------------------------------------------

_client = None
_client_key: tuple[str, str] | None = None


def _build_client(timeout: float):
    """Create an Ollama client for the current host/key configuration."""

    import ollama

    host = get_host()
    key = get_api_key()

    kwargs: dict = {"host": host, "timeout": timeout}

    if key:
        # Explicit header: the environment spelling used by this project
        # (Ollama_API_KEY) is not read by the SDK itself.
        kwargs["headers"] = {"Authorization": f"Bearer {key}"}

    return ollama.Client(**kwargs)


def _close_quietly(client) -> None:
    """Close a client without turning a cleanup failure into an error."""

    close = getattr(client, "close", None)

    if callable(close):
        try:
            close()
        except Exception:
            pass


def _get_client():
    """Return the Ollama client used for generation.

    A client assigned directly to the module-level `_client` (used by
    tests) has no cached configuration key. It is adopted for the
    current configuration instead of being replaced with a real network
    client, so tests stay hermetic and never contact a live backend.
    """

    global _client, _client_key

    cache_key = (get_host(), get_api_key())

    if _client is not None and _client_key is None:
        _client_key = cache_key
        return _client

    if _client is None or _client_key != cache_key:
        _close_quietly(_client)

        _client = _build_client(get_request_timeout())
        _client_key = cache_key

    return _client


# The availability probe uses its own short-timeout client so a network
# problem cannot make application startup hang.
_probe_client = None
_probe_client_key = None


def _get_probe_client():
    """Return the short-timeout client used by the startup probe."""

    global _probe_client, _probe_client_key

    cache_key = (get_host(), get_api_key())

    if _probe_client is None or _probe_client_key != cache_key:
        _close_quietly(_probe_client)

        _probe_client = _build_client(PROBE_TIMEOUT_SECONDS)
        _probe_client_key = cache_key

    return _probe_client


def reset_client() -> None:
    """Drop the cached clients (used by tests and after config changes)."""

    global _client, _client_key, _probe_client, _probe_client_key

    _close_quietly(_client)
    _close_quietly(_probe_client)

    _client = None
    _client_key = None
    _probe_client = None
    _probe_client_key = None

    # The validation cache is deliberately kept: it describes the
    # configuration, not a live connection, and it is keyed by
    # (host, key, model), so a configuration change invalidates it
    # automatically. Use reset_validation_cache() to force a re-probe.


# --------------------------------------------------
# Token usage extraction
# --------------------------------------------------

def _usage_from_response(response) -> TokenUsage | None:
    """Extract token usage when the backend reported it.

    Only the last streaming chunk can carry raw completion counters,
    so this helper reads whatever the response object exposes without
    guessing missing values.
    """

    def _clean(value):
        return value if isinstance(value, int) and value >= 0 else None

    prompt_tokens = None
    output_tokens = None

    # Prefer a structured usage object when the backend provides one.
    usage = getattr(response, "usage", None)

    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_eval_count", None)
        output_tokens = getattr(usage, "eval_count", None)

    if prompt_tokens is None:
        prompt_tokens = getattr(response, "prompt_eval_count", None)

    if output_tokens is None:
        output_tokens = getattr(response, "eval_count", None)

    input_tokens = _clean(prompt_tokens)
    output_tokens = _clean(output_tokens)

    if input_tokens is None and output_tokens is None:
        return None

    total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


# --------------------------------------------------
# Model discovery helpers
# --------------------------------------------------

def _model_names(listing) -> set[str]:
    """Normalize model tags reported by the Ollama list endpoint.

    The SDK returns a response object with a ``models`` attribute, while
    the tests supply a plain sequence of model objects; both shapes are
    accepted.
    """

    names: set[str] = set()

    items = getattr(listing, "models", None)

    if items is None:
        items = listing if isinstance(listing, (list, tuple)) else []

    for item in items or []:
        raw = getattr(item, "model", None) or getattr(item, "name", None)

        if not raw:
            continue

        tag = str(raw).removeprefix("models/").strip()
        names.add(tag if ":" in tag else f"{tag}:latest")

    return names


def _normalize_tag(tag: str) -> str:
    cleaned = tag.strip()

    return cleaned if ":" in cleaned else f"{cleaned}:latest"


# --------------------------------------------------
# Error translation
# --------------------------------------------------

def _translate_error(
    error: Exception,
    action: str,
    model: str | None = None,
) -> LLMError:
    """Turn an SDK/network failure into one clear, secret-free error.

    Only exception types and HTTP status codes are reported. The
    upstream message is never copied into the error, because it can
    contain request details or credentials.
    """

    if isinstance(error, LLMError):
        return error

    # The SDK raises ollama.ResponseError for non-2xx responses.
    if type(error).__name__ == "ResponseError":
        status = getattr(error, "status_code", None)

        if status == 404:
            return LLMError(
                f"Model {model or get_text_model()!r} is not available "
                "on the configured Ollama backend. No substitute model "
                "is used."
            )

        return LLMError(
            f"{action} failed: the Ollama backend returned status "
            f"{status or 'unknown'}."
        )

    if isinstance(error, ConnectionError):
        return LLMError(
            f"{action} failed: could not reach the Ollama backend at "
            f"{get_host()}. Check that Ollama is running or that the "
            "API key and network are valid."
        )

    return LLMError(f"{action} failed: {type(error).__name__}")


# --------------------------------------------------
# Startup validation
# --------------------------------------------------

# Definitive probe results are cached per configuration so repeatedly
# starting the app in one process (and the automated test suite) does not
# repeat the network round trip. A transient probe failure is never
# cached, so a backend that comes back is detected on the next start.
_validation_cache: dict | None = None
_validation_cache_key: tuple | None = None


def _cache_validation(cache_key: tuple, result: dict) -> None:
    """Store a definitive validation result."""

    global _validation_cache, _validation_cache_key

    _validation_cache = dict(result)
    _validation_cache_key = cache_key


def reset_validation_cache() -> None:
    """Forget the cached probe result (used by tests and config changes)."""

    global _validation_cache, _validation_cache_key

    _validation_cache = None
    _validation_cache_key = None


def validate_llm_config(force: bool = False) -> dict:
    """Validate the Ollama configuration and model availability.

    Called at API startup. Raises LLMError when the configuration cannot
    work at all (empty model tag). Cloud and local hosts are reported
    separately; the configured model tag is checked against the
    backend's model list, and an unavailable model is reported as an
    error rather than silently substituted.

    The result is cached per (host, key, model) so repeatedly starting the
    app in one process does not repeat the network probe. Pass
    ``force=True`` (or call :func:`reset_validation_cache`) to probe again.
    """

    model = get_text_model()

    cache_key = (get_host(), bool(get_api_key()), model)

    if (
        not force
        and _validation_cache is not None
        and _validation_cache_key == cache_key
    ):
        return dict(_validation_cache)

    if not model:
        raise LLMError("OLLAMA_TEXT_MODEL cannot be empty.")

    result: dict = {
        "configured": True,
        "model": model,
        "backend": backend_kind(),
        "model_available": None,
        "vision_supported": None,
        "warning": None,
    }

    warnings: list[str] = []

    try:
        available = _model_names(_get_probe_client().list())
    except Exception as error:
        # Availability check is best-effort: a transient listing
        # failure must not block the API from starting, but it is
        # reported loudly instead of hidden.
        result["warning"] = (
            f"Model availability could not be checked: "
            f"{type(error).__name__}"
        )
        # Transient failure: deliberately not cached.
        return result

    if _normalize_tag(model) not in available:
        result["model_available"] = False
        warnings.append(
            f"Model {model!r} was not found for the configured Ollama "
            "backend. Requests will fail until it is available. No "
            "substitute model is applied."
        )

        result["warning"] = " ".join(warnings)
        _cache_validation(cache_key, result)
        return result

    result["model_available"] = True

    # Capability check: image questions are only supported when the
    # model advertises the vision capability.
    try:
        info = _get_probe_client().show(model=model)
        capabilities = [
            str(capability).lower()
            for capability in (getattr(info, "capabilities", None) or [])
        ]

        if capabilities:
            result["vision_supported"] = "vision" in capabilities

            if "vision" not in capabilities:
                warnings.append(
                    f"Model {model!r} does not advertise image input "
                    "support; image questions will be rejected by the "
                    "backend."
                )
    except Exception:
        # Capability probing is optional; unknown stays unknown.
        result["vision_supported"] = None

    if warnings:
        result["warning"] = " ".join(warnings)

    _cache_validation(cache_key, result)

    return result


# --------------------------------------------------
# Message construction
# --------------------------------------------------

def _build_messages(prompt: str, system: str | None) -> list[dict]:
    """Build chat messages with trusted instructions in `system`."""

    messages: list[dict] = []

    if system is not None and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    messages.append({"role": "user", "content": prompt})

    return messages


# --------------------------------------------------
# Text generation
# --------------------------------------------------

def generate_response(prompt: str, system: str | None = None) -> str:
    """Generate a text response for the given user content.

    ``system`` carries the trusted instructions (the conversation's
    custom prompt plus the built-in security rules). ``prompt`` carries
    user content and any retrieved context.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    started = time.perf_counter()

    try:
        response = _get_client().chat(
            model=get_text_model(),
            messages=_build_messages(prompt, system),
        )

    except Exception as error:
        raise _translate_error(error, "Text generation") from error

    duration_ms = (time.perf_counter() - started) * 1000

    content = getattr(
        getattr(response, "message", None), "content", None
    )

    if not content or not str(content).strip():
        raise LLMError("The model returned an empty response.")

    _record_metadata(duration_ms, _usage_from_response(response))

    return str(content)


# --------------------------------------------------
# Streaming text generation
# --------------------------------------------------

def stream_response(
    prompt: str,
    system: str | None = None,
) -> Iterator[str]:
    """Yield text deltas as Ollama generates them.

    A failed or empty stream raises LLMError, but never invents text.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    started = time.perf_counter()

    try:
        stream = _get_client().chat(
            model=get_text_model(),
            messages=_build_messages(prompt, system),
            stream=True,
        )

    except TypeError as error:
        raise LLMError(
            "The configured Ollama backend does not support streaming."
        ) from error

    except Exception as error:
        raise _translate_error(error, "Stream") from error

    emitted_any = False
    final_usage = None

    try:
        for chunk in stream:
            # The real ollama SDK yields ChatResponse objects directly;
            # the text fragment is a message, not a top-level field.
            # Ollama stream chunks expose content directly on chunk.message.
            message = getattr(chunk, "message", None)
            delta = getattr(message, "content", None)

            if delta and isinstance(delta, str):
                emitted_any = True
                yield delta

            # The last chunk carries the completion counters.
            final_usage = _usage_from_response(chunk)

    except Exception as error:
        raise _translate_error(error, "Response stream") from error

    finally:
        # Release the upstream connection on completion, failure, and
        # client cancellation (GeneratorExit).
        close = getattr(stream, "close", None)

        if callable(close):
            close()

    _record_metadata(
        (time.perf_counter() - started) * 1000,
        final_usage,
    )

    if not emitted_any:
        raise LLMError("The model returned an empty response.")


# --------------------------------------------------
# Multimodal generation (same Ollama model, image input)
# --------------------------------------------------

def generate_multimodal_response(
    message: str,
    image_paths: list[str],
    system: str | None = None,
) -> str:
    """Generate a response for user content that includes images.

    Uses the configured Ollama model. Empty responses, missing image
    files, unsupported image operations, and connection failures raise
    LLMError (or ValueError when no valid image file exists).
    """

    valid_images = [
        str(path)
        for path in (Path(p) for p in image_paths)
        if path and Path(path).is_file()
    ]

    if not valid_images:
        raise ValueError("No valid image files were found.")

    started = time.perf_counter()

    try:
        # Images travel inside the user message (the SDK chat API has
        # no top-level images argument).
        messages = _build_messages(message, system)
        messages[-1] = {**messages[-1], "images": valid_images}

        response = _get_client().chat(
            model=get_vision_model(),
            messages=messages,
        )

    except Exception as error:
        raise _translate_error(
            error,
            "Image question answering",
            model=get_vision_model(),
        ) from error

    content = getattr(
        getattr(response, "message", None), "content", None
    )

    if not content or not str(content).strip():
        raise LLMError("The model returned an empty response.")

    _record_metadata(
        (time.perf_counter() - started) * 1000,
        _usage_from_response(response),
    )

    return str(content)