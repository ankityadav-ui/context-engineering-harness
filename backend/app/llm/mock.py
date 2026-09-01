from .base import BaseLLM, LLMResponse
from .config import LLMConfig
from .types import LLMRequest


class MockLLM(BaseLLM):
    """
    Mock LLM used for testing the provider architecture.

    Supports both the new LLMConfig-based construction and
    the legacy api_key/model constructor.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        api_key: str = "",
        model: str = "mock-model",
    ):
        super().__init__(
            config,
            api_key=api_key,
            model=model,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        last_message = request.messages[-1]["content"]

        return LLMResponse(
            text=f"Mock response to: {last_message}",
            model=self.model,
            provider=self.config.provider,
        )