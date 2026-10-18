"""Tests for the existing POST /masking/text contract.

The masking engine and safety validator are mocked, so these tests only
verify the HTTP contract the Streamlit masking workflow relies on: the
request schema, the approved-output fields, and fail-closed errors that
never echo the original text.
"""

from types import SimpleNamespace

import app.main as main_module


RAW_TEXT = "Contact John Doe at john@example.com for details."


def _masked_output(masked_text="Contact [PERSON_1] at [EMAIL_1] for details."):
    return {
        "result": SimpleNamespace(
            masked_text=masked_text,
            entities=[
                SimpleNamespace(entity_type="PERSON"),
                SimpleNamespace(entity_type="EMAIL"),
            ],
        ),
        "safety": {"passed": True, "reasons": []},
    }


def test_masking_text_returns_confirmed_masked_output(
    api_client,
    monkeypatch,
):
    captured = {}

    def fake_mask_and_validate(text, document_id):
        captured["text"] = text
        captured["document_id"] = document_id
        return _masked_output()

    monkeypatch.setattr(
        main_module,
        "mask_and_validate",
        fake_mask_and_validate,
    )

    response = api_client.post(
        "/masking/text",
        json={"text": RAW_TEXT},
    )

    assert response.status_code == 200

    body = response.json()

    assert captured == {
        "text": RAW_TEXT,
        "document_id": "text-input",
    }
    assert body["masked_text"] == (
        "Contact [PERSON_1] at [EMAIL_1] for details."
    )
    assert body["safety_passed"] is True
    assert body["detected_count"] == 2
    assert body["detected_categories"] == ["EMAIL", "PERSON"]
    assert body["safety_reasons"] == []

    # Raw values are never returned to the client.
    assert "John Doe" not in response.text
    assert "john@example.com" not in response.text


def test_masking_text_returns_safety_rejection_without_original_text(
    api_client,
    monkeypatch,
):
    def fake_mask_and_validate(text, document_id):
        return {
            "result": SimpleNamespace(
                masked_text=text,
                entities=[],
            ),
            "safety": {
                "passed": False,
                "reasons": ["Possible unmasked EMAIL detected."],
            },
        }

    monkeypatch.setattr(
        main_module,
        "mask_and_validate",
        fake_mask_and_validate,
    )

    response = api_client.post(
        "/masking/text",
        json={"text": RAW_TEXT},
    )

    # The endpoint reports the decision; the caller must fail closed.
    assert response.status_code == 200

    body = response.json()

    assert body["safety_passed"] is False
    assert body["safety_reasons"] == ["Possible unmasked EMAIL detected."]
    # Unapproved candidate output is withheld, so a caller cannot send it
    # by accident or display a missed value as if it were masked.
    assert body["masked_text"] is None
    assert "john@example.com" not in response.text


def test_masking_text_fails_closed_on_service_error(api_client, monkeypatch):
    def failing_mask_and_validate(**_kwargs):
        raise RuntimeError(f"internal failure while masking {RAW_TEXT}")

    monkeypatch.setattr(
        main_module,
        "mask_and_validate",
        failing_mask_and_validate,
    )

    response = api_client.post(
        "/masking/text",
        json={"text": RAW_TEXT},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Masking failed. No result was approved."
    )

    # Neither the original text nor the internal exception is echoed.
    assert "john@example.com" not in response.text
    assert "internal failure" not in response.text


def test_masking_text_rejects_blank_and_missing_text(api_client):
    blank = api_client.post("/masking/text", json={"text": "   "})

    assert blank.status_code == 400
    assert blank.json()["detail"] == "Text cannot be empty."

    missing = api_client.post("/masking/text", json={})

    assert missing.status_code == 422
