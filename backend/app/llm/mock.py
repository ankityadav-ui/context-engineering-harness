from .base import BaseLLM


class MockLLM(BaseLLM):
    """
    Mock LLM used for testing the provider architecture.
    """

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:

        last_message = messages[-1]["content"]

        return f"Mock response to: {last_message}"