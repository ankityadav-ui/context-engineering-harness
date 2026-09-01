from google import genai
from google.genai import types

from .base import BaseLLM, LLMResponse
from .config import LLMConfig
from .exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .types import LLMRequest


class GeminiLLM(BaseLLM):
    """
    Gemini implementation of the common BaseLLM interface.

    Translates google-genai SDK exceptions into common
    LLMError subclasses so that application code never
    sees provider-specific errors.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        api_key: str = "",
        model: str = "",
    ):
        super().__init__(
            config,
            api_key=api_key,
            model=model,
        )

        self.client = genai.Client(
            api_key=self.api_key,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        system_instruction = None
        contents = []

        for message in request.messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                system_instruction = content

            elif role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=content
                            )
                        ],
                    )
                )

            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                text=content
                            )
                        ],
                    )
                )

        config_kwargs = {}

        if system_instruction:
            config_kwargs["system_instruction"] = (
                system_instruction
            )

        if request.temperature is not None:
            config_kwargs["temperature"] = (
                request.temperature
            )

        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = (
                request.max_tokens
            )

        config = None

        if config_kwargs:
            config = types.GenerateContentConfig(
                **config_kwargs
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc

        return LLMResponse(
            text=response.text or "",
            model=self.model,
            provider=self.config.provider,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        """Translate a google-genai SDK exception into a common
        LLMError subclass."""
        import socket

        import httpx
        from google.genai.errors import (
            APIError,
            ClientError,
            ServerError,
        )

        # Timeout / connection errors
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return LLMTimeoutError(str(exc))
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(str(exc))
        if isinstance(exc, httpx.ConnectError):
            return LLMProviderError(str(exc))

        if not isinstance(exc, APIError):
            return LLMProviderError(str(exc))

        # google-genai uses HTTP status codes to decide ClientError
        # vs ServerError: 4xx → ClientError, 5xx → ServerError.
        # Rate limit (429) falls under ClientError (4xx range).
        code = getattr(exc, "code", 0) or 0

        if code == 429:
            return LLMRateLimitError(str(exc))
        if code in (401, 403):
            return LLMAuthenticationError(str(exc))
        if isinstance(exc, ServerError):
            return LLMProviderError(str(exc))
        if isinstance(exc, ClientError):
            return LLMProviderError(str(exc))

        # Generic APIError fallback
        return LLMProviderError(str(exc))