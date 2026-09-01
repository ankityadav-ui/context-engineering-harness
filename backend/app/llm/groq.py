from groq import Groq
from groq import (
    APITimeoutError,
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    BadRequestError,
    APIStatusError,
    GroqError,
)

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


class GroqLLM(BaseLLM):
    """Groq implementation of the common BaseLLM interface.

    Translates groq SDK exceptions into common LLMError
    subclasses so that application code never sees
    provider-specific errors.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self.client = Groq(
            api_key=config.api_key,
            timeout=LLM_TIMEOUT_SECONDS,
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
            response = self.client.chat.completions.create(
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
        """Translate a groq SDK exception into a common
        LLMError subclass."""

        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError(str(exc))

        if isinstance(exc, AuthenticationError):
            return LLMAuthenticationError(str(exc))

        if isinstance(exc, RateLimitError):
            return LLMRateLimitError(str(exc))

        if isinstance(exc, InternalServerError):
            return LLMProviderError(str(exc))

        if isinstance(exc, (BadRequestError, APIStatusError)):
            return LLMProviderError(str(exc))

        if isinstance(exc, APIConnectionError):
            return LLMProviderError(str(exc))

        if isinstance(exc, GroqError):
            return LLMProviderError(str(exc))

        return LLMProviderError(str(exc))