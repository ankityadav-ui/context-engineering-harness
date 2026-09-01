from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Common interface for all LLM providers.

    The application talks only to this interface.
    Provider-specific SDK logic stays inside each
    provider implementation.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: Chat messages using the common format.
            temperature: Optional generation temperature.
            max_tokens: Optional maximum output tokens.

        Returns:
            Generated text.
        """

        raise NotImplementedError