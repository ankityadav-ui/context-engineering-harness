from collections.abc import Callable

from .base import BaseLLM
from .config import LLMConfig
from .gemini import GeminiLLM
from .groq import GroqLLM
from .mistral import MistralLLM
from .mock import MockLLM
from .openrouter import OpenRouterLLM

from ..config import (
    LLM_PROVIDER,
    LLM_API_KEY,
    LLM_MODEL,
)


LLMProviderFactory = Callable[[LLMConfig], BaseLLM]


def _create_mock(config: LLMConfig) -> BaseLLM:
    return MockLLM(config)


def _create_gemini(config: LLMConfig) -> BaseLLM:
    return GeminiLLM(config)


def _create_groq(config: LLMConfig) -> BaseLLM:
    return GroqLLM(config)


def _create_mistral(config: LLMConfig) -> BaseLLM:
    return MistralLLM(config)


def _create_openrouter(config: LLMConfig) -> BaseLLM:
    return OpenRouterLLM(config)


LLM_PROVIDERS: dict[str, LLMProviderFactory] = {
    "mock": _create_mock,
    "gemini": _create_gemini,
    "groq": _create_groq,
    "mistral": _create_mistral,
    "openrouter": _create_openrouter,
}


PROVIDERS_REQUIRING_API_KEY = {
    "gemini",
    "groq",
    "mistral",
    "openrouter",
}


def create_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> BaseLLM:
    """
    Create an LLM implementation using the configured provider.

    Application code depends only on BaseLLM.
    Provider-specific construction is isolated here.
    """

    provider_name = (
        provider or LLM_PROVIDER
    ).lower().strip()

    selected_api_key = api_key or LLM_API_KEY
    selected_model = model or LLM_MODEL

    provider_factory = LLM_PROVIDERS.get(
        provider_name
    )

    if provider_factory is None:
        supported = ", ".join(
            sorted(LLM_PROVIDERS)
        )

        raise ValueError(
            f"Unsupported LLM provider: {provider_name}. "
            f"Supported providers: {supported}"
        )

    if (
        provider_name in PROVIDERS_REQUIRING_API_KEY
        and not selected_api_key
    ):
        raise ValueError(
            f"{provider_name.capitalize()} "
            f"API key is not configured"
        )

    config = LLMConfig(
        provider=provider_name,
        model=selected_model,
        api_key=selected_api_key or "",
    )

    return provider_factory(config)