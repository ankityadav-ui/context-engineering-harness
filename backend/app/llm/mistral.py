from mistralai.client import Mistral

from ..config import LLM_TIMEOUT_SECONDS
from .base import BaseLLM, LLMResponse
from .config import LLMConfig
from .exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .types import LLMRequest


class MistralLLM(BaseLLM):
    """Mistral implementation of the common BaseLLM interface.

    Translates mistralai SDK exceptions into common LLMError
    subclasses so that application code never sees
    provider-specific errors.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # mistralai uses timeout_ms (milliseconds)
        timeout_ms = int(LLM_TIMEOUT_SECONDS * 1000)

        self.client = Mistral(
            api_key=config.api_key,
            timeout_ms=timeout_ms,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:

        kwargs = {
            "model": self.model,
            "messages": request.messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        try:
            response = self.client.chat.complete(
                **kwargs
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            provider=self.config.provider,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        """Translate a mistralai SDK exception into a common
        LLMError subclass.

        The mistralai SDK does not expose fine-grained HTTP
        error classes like other providers. Most HTTP errors
        arrive as ``SDKError`` with a ``raw_response`` that
        carries the status code.  Timeout/connection errors
        arrive as ``httpx`` exceptions.
        """
        import httpx
        from mistralai.client.errors import SDKError

        # Timeout / connection errors from httpx
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(str(exc))
        if isinstance(exc, httpx.ConnectError):
            return LLMProviderError(str(exc))

        # mistralai SDKError carries raw_response with status_code
        if isinstance(exc, SDKError):
            raw = getattr(exc, "raw_response", None)
            status = (
                getattr(raw, "status_code", 0) if raw else 0
            ) or 0

            if status == 429:
                return LLMRateLimitError(str(exc))
            if status in (401, 403):
                return LLMAuthenticationError(str(exc))

            return LLMProviderError(str(exc))

        # Any other MistralError subclass
        return LLMProviderError(str(exc))