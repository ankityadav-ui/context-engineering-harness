from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    """
    Provider-independent request for LLM generation.
    """

    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None