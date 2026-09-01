from openai import OpenAI

from .base import BaseLLM


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter implementation of the common BaseLLM interface.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        super().__init__(
            api_key=api_key,
            model=model,
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(
            **kwargs
        )

        return response.choices[0].message.content