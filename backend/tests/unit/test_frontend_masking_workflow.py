"""Tests for the Streamlit masking workflow.

The HTTP layer is mocked: these tests verify the request contract of the
existing masking endpoint and that enabled masking changes the text sent
to chat, while every failure mode refuses to send anything.
"""

import sys
from pathlib import Path

import pytest
import requests

FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import api_client  # noqa: E402
import masking_workflow  # noqa: E402


RAW_TEXT = "Email john@example.com about the case."


class FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload

        return self._payload


def masked_payload(**overrides):
    payload = {
        "masked_text": "Email [EMAIL_1] about the case.",
        "safety_passed": True,
        "detected_count": 1,
        "detected_categories": ["EMAIL"],
        "safety_reasons": [],
    }
    payload.update(overrides)
    return payload


def test_mask_for_chat_uses_existing_endpoint_and_exact_request_schema(
    monkeypatch,
):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(masked_payload())

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    outcome = masking_workflow.mask_for_chat(
        RAW_TEXT,
        api_url="http://backend:8000/",
    )

    assert captured["url"] == "http://backend:8000/masking/text"
    assert captured["json"] == {"text": RAW_TEXT}

    assert outcome.masked_text == "Email [EMAIL_1] about the case."
    assert outcome.detected_count == 1
    assert outcome.detected_categories == ["EMAIL"]


def test_mask_for_chat_rejects_when_safety_did_not_pass(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(
            masked_payload(
                safety_passed=False,
                safety_reasons=["Possible unmasked EMAIL detected."],
            )
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(masking_workflow.MaskingRejected) as error:
        masking_workflow.mask_for_chat(RAW_TEXT)

    assert "Possible unmasked EMAIL detected." in str(error.value)
    # The original sensitive value is never put into the error message.
    assert "john@example.com" not in str(error.value)


def test_mask_for_chat_rejects_empty_approved_text(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(masked_payload(masked_text="   "))

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(masking_workflow.MaskingRejected) as error:
        masking_workflow.mask_for_chat(RAW_TEXT)

    assert "no approved text" in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("backend down"),
        FakeResponse({"detail": "Masking failed."}, status_code=500),
    ],
)
def test_mask_for_chat_fails_closed_when_service_is_unavailable(
    monkeypatch,
    failure,
):
    if isinstance(failure, Exception):

        def fake_post(*_args, **_kwargs):
            raise failure

    else:

        def fake_post(*_args, **_kwargs):
            return failure

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(masking_workflow.MaskingUnavailable) as error:
        masking_workflow.mask_for_chat(RAW_TEXT)

    assert "john@example.com" not in str(error.value)


def test_prepare_chat_message_sends_masked_text_when_enabled():
    calls = []

    def fake_mask(text, api_url=None):
        calls.append({"text": text, "api_url": api_url})

        return masking_workflow.MaskingOutcome(
            masked_text="Email [EMAIL_1] about the case.",
            detected_count=1,
            detected_categories=["EMAIL"],
        )

    prepared = masking_workflow.prepare_chat_message(
        RAW_TEXT,
        masking_enabled=True,
        api_url="http://backend:8000",
        mask=fake_mask,
    )

    assert calls == [
        {"text": RAW_TEXT, "api_url": "http://backend:8000"}
    ]
    assert prepared.content == "Email [EMAIL_1] about the case."
    assert prepared.masking is not None
    assert prepared.masking.detected_count == 1


def test_prepare_chat_message_keeps_disabled_contract_unchanged():
    calls = []

    def fake_mask(text, api_url=None):  # pragma: no cover - must not run
        calls.append(text)
        raise AssertionError("Masking must not run when disabled.")

    prepared = masking_workflow.prepare_chat_message(
        RAW_TEXT,
        masking_enabled=False,
        mask=fake_mask,
    )

    assert calls == []
    assert prepared.content == RAW_TEXT
    assert prepared.masking is None


def test_prepare_chat_message_propagates_rejection():
    def rejecting_mask(_text, api_url=None):
        raise masking_workflow.MaskingRejected("Blocked.")

    with pytest.raises(masking_workflow.MaskingRejected):
        masking_workflow.prepare_chat_message(
            RAW_TEXT,
            masking_enabled=True,
            mask=rejecting_mask,
        )
