"""Fail-closed masking workflow for the Streamlit chat composer.

The backend owns detection, masking, and the secondary safety check
(``POST /masking/text``). This module only decides whether the approved
output may leave the browser:

- ``safety_passed`` must be ``True`` and the approved text non-empty,
  otherwise :class:`MaskingRejected` is raised and nothing is sent.
- A transport/HTTP failure raises :class:`MaskingUnavailable`; the
  original text is never used as a fallback.

No function here returns or logs the original sensitive text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import requests

import api_client


@dataclass(frozen=True)
class MaskingOutcome:
    """Backend-confirmed masking result for one outgoing message."""

    masked_text: str
    detected_count: int = 0
    detected_categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedChatMessage:
    """The text actually sent to chat, plus the confirmed masking report."""

    content: str
    masking: MaskingOutcome | None = None


class MaskingError(Exception):
    """Base class for failures that must stop a message from being sent."""


class MaskingRejected(MaskingError):
    """The backend's safety validation did not approve the masked text."""


class MaskingUnavailable(MaskingError):
    """The masking service could not be reached or returned an invalid body."""


def _optional_int(value: object) -> int:
    """Return a non-negative count, tolerating a malformed payload."""

    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, count)


def _categories(value: object) -> list[str]:
    """Return clean category labels from a backend response."""

    if not isinstance(value, list):
        return []

    return [str(item) for item in value if str(item).strip()]


def mask_for_chat(
    text: str,
    api_url: str | None = None,
) -> MaskingOutcome:
    """Mask one outgoing chat message, or raise without a fallback.

    Calls the existing ``POST /masking/text`` endpoint with the exact
    request schema (``{"text": ...}``).
    """

    if not isinstance(text, str) or not text.strip():
        raise MaskingRejected("There was no message content to mask.")

    try:
        result = api_client.mask_text(text, api_url=api_url)

    except (requests.RequestException, api_client.APIError, ValueError):
        # Never fall back to the original text when masking cannot run.
        raise MaskingUnavailable(
            "The masking service could not be reached, so the message "
            "was not sent."
        ) from None

    if not isinstance(result, dict):
        raise MaskingUnavailable(
            "The masking service returned an unreadable result, so the "
            "message was not sent."
        )

    if result.get("safety_passed") is not True:
        reasons = [
            str(reason)
            for reason in result.get("safety_reasons") or []
            if str(reason).strip()
        ]

        raise MaskingRejected(
            " ".join(reasons)
            or "The masking safety check did not pass, so the message "
            "was not sent."
        )

    masked_text = result.get("masked_text")

    if not isinstance(masked_text, str) or not masked_text.strip():
        raise MaskingRejected(
            "The masking service returned no approved text, so the "
            "message was not sent."
        )

    return MaskingOutcome(
        masked_text=masked_text,
        detected_count=_optional_int(result.get("detected_count")),
        detected_categories=_categories(
            result.get("detected_categories")
        ),
    )


def prepare_chat_message(
    text: str,
    masking_enabled: bool,
    *,
    api_url: str | None = None,
    mask: Callable[..., MaskingOutcome] | None = None,
) -> PreparedChatMessage:
    """Return the exact text to send, applying masking when enabled.

    With masking disabled the pre-existing chat contract is unchanged:
    the original text is sent and the masking endpoint is not called.
    """

    if not masking_enabled:
        return PreparedChatMessage(content=text, masking=None)

    masker = mask or mask_for_chat
    outcome = masker(text, api_url=api_url)

    return PreparedChatMessage(content=outcome.masked_text, masking=outcome)
