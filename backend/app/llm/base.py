from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import LLMConfig
from .types import LLMRequest


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-independent response returned by every LLM.
    """

    text: str
    model: str | None = None
    provider: str | None = None


class BaseLLM(ABC):
    """
    Common interface for all LLM providers.

    Supports both:

        BaseLLM(config)

    and the legacy:

        BaseLLM(api_key="...", model="...")

    The compatibility layer allows the existing application
    and tests to continue working while the provider architecture
    remains model-agnostic.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        api_key: str = "",
        model: str = "",
    ):
        if config is None:
            config = LLMConfig(
                provider=self.__class__.__name__.replace(
                    "LLM",
                    "",
                ).lower(),
                model=model,
                api_key=api_key,
            )

        self.config = config

        self.api_key = config.api_key
        self.model = config.model

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a provider-independent response.

        Provider-specific SDK response objects must never
        leave the provider adapter.
        """

        raise NotImplementedError