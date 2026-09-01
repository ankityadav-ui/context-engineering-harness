from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    """
    Provider-independent configuration for an LLM.

    The application uses this object instead of passing
    provider-specific configuration around independently.
    """

    provider: str
    model: str
    api_key: str = ""

    # Optional provider-specific configuration.
    # Providers can read only the values they understand.
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            self.provider.lower().strip(),
        )