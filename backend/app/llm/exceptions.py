"""
Provider-independent exception hierarchy for the LLM layer.

Every provider adapter translates its SDK-specific exceptions into
these common exceptions.  Application code catches only these types
and never sees provider-specific SDK errors.

Hierarchy:

    LLMError
    ├── LLMConfigurationError
    ├── LLMAuthenticationError
    ├── LLMRateLimitError
    ├── LLMTimeoutError
    └── LLMProviderError
"""


class LLMError(Exception):
    """Base exception for all LLM failures."""


class LLMConfigurationError(LLMError):
    """Invalid or missing LLM configuration."""


class LLMAuthenticationError(LLMError):
    """Authentication / API-key failure."""


class LLMRateLimitError(LLMError):
    """Provider rate-limit / quota failure."""


class LLMTimeoutError(LLMError):
    """LLM request timed out."""


class LLMProviderError(LLMError):
    """Generic provider, server, or API failure."""
