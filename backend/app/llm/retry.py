"""
Provider-independent retry utility for transient LLM failures.

Retries are attempted ONLY for transient errors:
  - LLMTimeoutError
  - LLMRateLimitError
  - LLMProviderError (temporary/server failures)

Non-retryable errors propagate immediately:
  - LLMConfigurationError
  - LLMAuthenticationError

The application never sees raw SDK exceptions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from .exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that are safe to retry.
_RETRYABLE: tuple[type[LLMError], ...] = (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderError,
)


def is_retryable(exc: LLMError) -> bool:
    """Return True when *exc* represents a transient failure."""
    return isinstance(exc, _RETRYABLE)


def with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 2,
    delay: float = 1.0,
) -> T:
    """Call *fn* with retries for transient LLM failures.

    Parameters
    ----------
    fn:
        Zero-argument callable that performs the LLM request.
    max_retries:
        Maximum number of **retries** after the initial attempt.
        max_retries=2 → up to 3 total attempts.
    delay:
        Seconds to wait between retries.

    Raises
    ------
    LLMError
        Re-raises the last common exception after all retries
        are exhausted, or immediately for non-retryable errors.
    """
    last_exc: LLMError | None = None

    for attempt in range(1 + max_retries):
        try:
            return fn()
        except LLMConfigurationError:
            # Never retry configuration errors.
            raise
        except LLMAuthenticationError:
            # Never retry authentication failures.
            raise
        except LLMError as exc:
            last_exc = exc
            if not is_retryable(exc):
                raise
            if attempt < max_retries:
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs …",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

    # All retries exhausted.
    raise last_exc  # type: ignore[misc]
