from .base import BaseLLM
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


def create_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> BaseLLM:

    provider = (provider or LLM_PROVIDER).lower().strip()
    api_key = api_key or LLM_API_KEY
    model = model or LLM_MODEL

    if provider == "mock":
        return MockLLM(
            api_key=api_key or "",
            model=model,
        )

    if provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is not configured")

        return GeminiLLM(
            api_key=api_key,
            model=model,
        )

    if provider == "groq":
        if not api_key:
            raise ValueError("Groq API key is not configured")

        return GroqLLM(
            api_key=api_key,
            model=model,
        )

    if provider == "mistral":
        if not api_key:
            raise ValueError("Mistral API key is not configured")

        return MistralLLM(
            api_key=api_key,
            model=model,
        )
    if provider == "openrouter":
        if not api_key:
            raise ValueError("OpenRouter API key is not configured")

        return OpenRouterLLM(
            api_key=api_key,
            model=model,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )