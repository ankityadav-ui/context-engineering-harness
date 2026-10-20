from types import SimpleNamespace

from app.rag import document_service


def test_existing_placeholders_are_distinguished_from_newly_masked_values(
    monkeypatch,
):
    def fake_mask_and_validate(text, document_id, registry, include_header_name):
        del document_id, include_header_name
        if "New Sensitive Name" in text:
            placeholder = registry.placeholder_for(
                "PERSON",
                "New Sensitive Name",
            )
            text = text.replace("New Sensitive Name", placeholder)
        return {
            "result": SimpleNamespace(masked_text=text),
            "safety": {"passed": True, "reasons": []},
        }

    monkeypatch.setattr(
        document_service,
        "mask_and_validate",
        fake_mask_and_validate,
    )

    units = [
        {
            "content_type": "text",
            "content": (
                "Existing [PERSON_7] appears twice: [PERSON_7]. "
                "New Sensitive Name is also present."
            ),
        }
    ]

    stats = document_service.mask_content_units(
        units,
        document_id="doc-mask-test",
    )

    assert stats["masked_entities"] == 1
    assert stats["existing_placeholders"] == 1
    assert stats["existing_placeholder_occurrences"] == 2
    assert "[PERSON_7]" in units[0]["content"]
    assert "[PERSON_1]" in units[0]["content"]
